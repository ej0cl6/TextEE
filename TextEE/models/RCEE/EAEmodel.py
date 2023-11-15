import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, XLMRobertaConfig, BertModel, RobertaModel, XLMRobertaModel
from .pattern import patterns
import ipdb

class RCEEEAEModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        
        # base encoder
        if self.config.pretrained_model_name.startswith('bert-'):
            self.tokenizer.bos_token = self.tokenizer.cls_token
            self.tokenizer.eos_token = self.tokenizer.sep_token
            self.base_config = BertConfig.from_pretrained(self.config.pretrained_model_name, 
                                                          cache_dir=self.config.cache_dir)
            self.base_model = BertModel.from_pretrained(self.config.pretrained_model_name, 
                                                        cache_dir=self.config.cache_dir, 
                                                        output_hidden_states=True)
        elif self.config.pretrained_model_name.startswith('roberta-'):
            self.base_config = RobertaConfig.from_pretrained(self.config.pretrained_model_name, 
                                                             cache_dir=self.config.cache_dir)
            self.base_model = RobertaModel.from_pretrained(self.config.pretrained_model_name, 
                                                           cache_dir=self.config.cache_dir, 
                                                           output_hidden_states=True)
        elif self.config.pretrained_model_name.startswith('xlm-'):
            self.base_config = XLMRobertaConfig.from_pretrained(self.config.pretrained_model_name, 
                                                                cache_dir=self.config.cache_dir)
            self.base_model = XLMRobertaModel.from_pretrained(self.config.pretrained_model_name, 
                                                              cache_dir=self.config.cache_dir, 
                                                              output_hidden_states=True)
        else:
            raise ValueError(f"pretrained_model_name is not supported.")
        
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.base_model_dim = self.base_config.hidden_size
        self.base_model_dropout = nn.Dropout(p=self.config.base_model_dropout)
        
        # local classifiers
        self.dropout = nn.Dropout(p=self.config.linear_dropout)
        feature_dim = self.base_model_dim

        self.args_start_ffn = Linears([feature_dim, 2],
                                   dropout_prob=self.config.linear_dropout, 
                                   bias=self.config.linear_bias, 
                                   activation=self.config.linear_activation)
        
        self.args_end_ffn = Linears([feature_dim, 2],
                                   dropout_prob=self.config.linear_dropout, 
                                   bias=self.config.linear_bias, 
                                   activation=self.config.linear_activation)
            
    def token_lens_to_offsets(self, token_lens):
        """Map token lengths to first word piece indices, used by the sentence
        encoder.
        :param token_lens (list): token lengths (word piece numbers)
        :return (list): first word piece indices (offsets)
        """
        max_token_num = max([len(x) for x in token_lens])
        offsets = []
        for seq_token_lens in token_lens:
            seq_offsets = [0]
            for l in seq_token_lens[:-1]:
                seq_offsets.append(seq_offsets[-1] + l)
            offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
        return offsets
            
    def token_lens_to_idxs(self, token_lens):
        """Map token lengths to a word piece index matrix (for torch.gather) and a
        mask tensor.
        For example (only show a sequence instead of a batch):
        token lengths: [1,1,1,3,1]
        =>
        indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
        masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]
        Next, we use torch.gather() to select vectors of word pieces for each token,
        and average them as follows (incomplete code):
        outputs = torch.gather(bert_outputs, 1, indices) * masks
        outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
        outputs = bert_outputs.sum(2)
        :param token_lens (list): token lengths.
        :return: a index matrix and a mask tensor.
        """
        max_token_num = max([len(x) for x in token_lens])
        max_token_len = max([max(x) for x in token_lens])
        idxs, masks = [], []
        for seq_token_lens in token_lens:
            seq_idxs, seq_masks = [], []
            offset = 0
            for token_len in seq_token_lens:
                seq_idxs.extend([i + offset for i in range(token_len)]
                                + [-1] * (max_token_len - token_len))
                seq_masks.extend([1.0 / token_len] * token_len
                                 + [0.0] * (max_token_len - token_len))
                offset += token_len
            seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
            seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
            idxs.append(seq_idxs)
            masks.append(seq_masks)
        return idxs, masks, max_token_num, max_token_len
    
    def process_data(self, batch):
        enc_idxs = []
        enc_attn = []
        start_labels = []
        end_labels = []
        token_lens = []
        token_nums = []
        triggers = []
        question_offsets = []
        max_token_num = max(batch.batch_token_num)
        
        for tokens, pieces, trigger, arguments, token_len, token_num in zip(batch.batch_tokens, batch.batch_pieces, batch.batch_trigger, 
                                                                      batch.batch_arguments, batch.batch_token_lens, batch.batch_token_num):
            
            questions = patterns[self.config.dataset][self.config.question_type][trigger[2]]
            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)
            
            for candidate in sorted(questions.keys()):
                
                question = questions[candidate]
                question = question.replace("[trigger]", trigger[3])
                
                question_pieces = self.tokenizer.tokenize(question, is_split_into_words=True)
                question_idx = self.tokenizer.convert_tokens_to_ids(question_pieces)
                
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_idx + \
                          [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
                
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                
                question_offsets.append(len(question_idx) + 2)
                
                start_label = [0] * (token_num+1)  # +1 for CLS token
                end_label = [0] * (token_num+1)    # +1 for CLS token
                
                no_answer_flag = True
                for argument in arguments:
                    if argument[2] == candidate:
                        start, end = argument[0], argument[1]
                        start_label[start+1] = 1
                        end_label[end] = 1
                        no_answer_flag = False
                        
                if no_answer_flag:
                    start_label[0] = 1
                    end_label[0] = 1
                    
                token_lens.append(token_len)
                token_nums.append(token_num)
                triggers.append(trigger)
                start_labels.append(start_label + [-100] * (max_token_num-len(tokens)))
                end_labels.append(end_label + [-100] * (max_token_num-len(tokens)))
                
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        start_labels = torch.cuda.LongTensor(start_labels)
        end_labels = torch.cuda.LongTensor(end_labels)
        return enc_idxs, enc_attn, start_labels, end_labels, token_lens, torch.cuda.LongTensor(token_nums), triggers, question_offsets
        
    def encode(self, piece_idxs, attention_masks, token_lens, question_offsets):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_base_model_outputs = self.base_model(piece_idxs, attention_mask=attention_masks)
        base_model_outputs = all_base_model_outputs[0]
        if self.config.multi_piece_strategy == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim)
            # add question offsets
            new_offsets = offsets.clone()
            for i, question_offset in enumerate(question_offsets):
                new_offsets[i] = offsets[i] + question_offset
            base_model_outputs = torch.gather(base_model_outputs, 1, new_offsets)
        elif self.config.multi_piece_strategy == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.base_model_dim)
            # add question offsets
            new_idxs = idxs.clone()
            for i, question_offset in enumerate(question_offsets):
                new_idxs[i] = idxs[i] + question_offset
            masks = base_model_outputs.new(masks).unsqueeze(-1)
            base_model_outputs = torch.gather(base_model_outputs, 1, new_idxs) * masks
            base_model_outputs = base_model_outputs.view(batch_size, token_num, token_len, self.base_model_dim)
            base_model_outputs = base_model_outputs.sum(2)
        else:
            raise ValueError(f'Unknown multi-piece token handling strategy: {self.config.multi_piece_strategy}')
        
        # add cls embedding
        cls_model_outputs = all_base_model_outputs[0][:, :1, :]
        base_model_outputs = torch.concat((cls_model_outputs, base_model_outputs), dim=1)
        base_model_outputs = self.base_model_dropout(base_model_outputs)
        return base_model_outputs

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, start_labels, end_labels, token_lens, token_nums, triggers, question_offsets = self.process_data(batch)
        
        # encoding
        base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens, question_offsets)
        
        start_logits = self.args_start_ffn(base_model_outputs)
        end_logits = self.args_end_ffn(base_model_outputs)
        
        start_loss = F.cross_entropy(start_logits.view(-1, start_logits.size(-1)), start_labels.view(-1))
        end_loss = F.cross_entropy(end_logits.view(-1, end_logits.size(-1)), end_labels.view(-1))
        
        loss = start_loss + end_loss
        
        return loss
    
    def get_spans(self, start_logits, end_logits, token_nums, argument_threshold=0.3):
        
        start_probs = F.softmax(start_logits, dim=2)[:, :, 1].cpu().numpy()
        end_probs = F.softmax(end_logits, dim=2)[:, :, 1].cpu().numpy()
        
        arguments = []
        for start_prob, end_prob, token_num in zip(start_probs, end_probs, token_nums):
            arguments_ = []
            for si, sp in enumerate(start_prob[1:]):
                for ei, ep in enumerate(end_prob[1:]):
                    if si >= token_num or ei >= token_num: # index not in the sentence
                        continue
                    if ei < si: # not a valid span
                        continue
                    if ei - si + 1 > self.config.max_answer_length: # span is too long
                        continue
                    if sp < start_prob[0] or ep < end_prob[0]: # position score < cls score
                        continue
                    
                    if sp + ep > argument_threshold:
                        arguments_.append((si, ei+1, sp+ep))
                    
            arguments.append(arguments_)
            
        return arguments
        
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, _, token_lens, token_nums, triggers, question_offsets = self.process_data(batch)
            
            # encoding
            base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens, question_offsets)
            
            start_logits = self.args_start_ffn(base_model_outputs)
            end_logits = self.args_end_ffn(base_model_outputs)
            
            arguments = self.get_spans(start_logits, end_logits, token_nums, argument_threshold=self.config.argument_threshold)
            
            
            cnt = 0
            new_arguments = []
            for b_idx, trigger in enumerate(batch.batch_trigger):
                questions = patterns[self.config.dataset][self.config.question_type][trigger[2]]
                new_sub_arguments = []
                for candidate in sorted(questions.keys()):
                    new_sub_arguments.extend([[a[0], a[1], candidate] for a in arguments[cnt]])
                    cnt += 1
                new_arguments.append(new_sub_arguments)
            assert cnt == enc_idxs.size(0)
            arguments = new_arguments
                
        self.train()
        return arguments

class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i > 0:
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            inputs = layer(inputs)
            outputs.append(inputs)
        return outputs[-1]
