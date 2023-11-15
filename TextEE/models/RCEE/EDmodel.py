import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, RobertaConfig, XLMRobertaConfig, BertModel, RobertaModel, XLMRobertaModel
import ipdb

class RCEEEDModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        self.generate_tagging_vocab()
        
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

        self.trigger_label_ffn = Linears([feature_dim, len(self.type_stoi["trigger"])],
                                      dropout_prob=self.config.linear_dropout, 
                                      bias=self.config.linear_bias, 
                                      activation=self.config.linear_activation)
            
    def generate_tagging_vocab(self):
        trigger_type_itos = ['O'] + [t for t in sorted(self.type_set["trigger"])]
        trigger_type_stoi = {t: i for i, t in enumerate(trigger_type_itos)}
        self.type_itos = {"trigger": trigger_type_itos}
        self.type_stoi = {"trigger": trigger_type_stoi}
        
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
        trigger_seqidxs = []
        token_lens = []
        token_nums = []
        max_token_num = max(batch.batch_token_num)
        
        for tokens, pieces, triggers, token_len, token_num in zip(batch.batch_tokens, batch.batch_pieces, batch.batch_triggers, 
                                                                      batch.batch_token_lens, batch.batch_token_num):

            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)
            
            if self.config.question_type == "[EVENT]":
                question = "[EVENT]"
                question_pieces = self.tokenizer.tokenize(question, is_split_into_words=True)
                question_idx = self.tokenizer.convert_tokens_to_ids(question_pieces)
                
                enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + question_idx + \
                          [self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
            
                enc_idxs.append(enc_idx)
                enc_attn.append([1]*len(enc_idx))
                self.question_offset = len(question_idx) + 2
                
                token_lens.append(token_len)
                token_nums.append(token_num)
                
                labels = ['O'] * token_num
                for trigger in triggers:
                    if labels[trigger[0]] == 'O':
                        labels[trigger[0]] = trigger[2]
                
                trigger_seqidxs.append([self.type_stoi["trigger"][s] for s in labels] + [-100] * (max_token_num-len(tokens)))

        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        trigger_seqidxs = torch.cuda.LongTensor(trigger_seqidxs)
        return enc_idxs, enc_attn, trigger_seqidxs, token_lens, torch.cuda.LongTensor(token_nums)
        
    def encode(self, piece_idxs, attention_masks, token_lens):
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
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + self.question_offset
            base_model_outputs = torch.gather(base_model_outputs, 1, offsets)
        elif self.config.multi_piece_strategy == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.base_model_dim) + self.question_offset
            masks = base_model_outputs.new(masks).unsqueeze(-1)
            base_model_outputs = torch.gather(base_model_outputs, 1, idxs) * masks
            base_model_outputs = base_model_outputs.view(batch_size, token_num, token_len, self.base_model_dim)
            base_model_outputs = base_model_outputs.sum(2)
        else:
            raise ValueError(f'Unknown multi-piece token handling strategy: {self.config.multi_piece_strategy}')
        base_model_outputs = self.base_model_dropout(base_model_outputs)
        return base_model_outputs

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, trigger_seqidxs, token_lens, token_nums = self.process_data(batch)
        
        # encoding
        base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        logits = self.trigger_label_ffn(base_model_outputs)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), trigger_seqidxs.view(-1))
        
        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, token_lens, token_nums = self.process_data(batch)
            
            # encoding
            base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            logits = self.trigger_label_ffn(base_model_outputs)
            preds = logits.argmax(dim=-1)
            
            preds = preds.cpu().numpy()
            pred_triggers = []
            for pred, token_num in zip(preds, token_nums):
                pred_trigger = []
                for i, t in enumerate(pred):
                    if i >= token_num:
                        break
                    if t == 0:
                        continue
                    pred_trigger.append((i, i+1, self.type_itos["trigger"][t]))
                pred_triggers.append(pred_trigger)
        self.train()
        return pred_triggers


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
