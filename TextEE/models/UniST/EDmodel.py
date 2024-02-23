import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertConfig, RobertaConfig, XLMRobertaConfig, BertModel, RobertaModel, XLMRobertaModel
import ipdb
        
class UniSTModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.type_set = type_set
        self.label_list = [x for x in sorted(self.type_set["trigger"])]
        self.label_strings = [self.preprocess_label(x) for x in self.label_list]
        
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
        
    def preprocess_label(self, label):
        label = label.lower()
        label = label.replace("_", " ")
        label = label.replace("-", " ")
        label = label.replace(":", " ")
        return label
        
    def process_data(self, batch):
        
        sent_texts = []
        pos_texts = []
        neg_texts = []
        for tokens, triggers in zip(batch.batch_tokens, batch.batch_triggers):
            sample_idxs = np.arange(len(triggers))
            np.random.shuffle(sample_idxs)
            triggers_ = [triggers[i] for i in sample_idxs[:self.config.max_sample_trigger]]
            for trigger in triggers_:
                text = tokens[:trigger[0]] + ["<t>"] + tokens[trigger[0]:trigger[1]]+ ["</t>"] + tokens[trigger[1]:]
                text = " ".join(text)
                sent_texts.append(text)
                
                pos_texts.append(self.preprocess_label(trigger[2]))
                
                neg_text = trigger[2]
                while neg_text == trigger[2]:
                    neg_text = np.random.choice(list(self.type_set["trigger"]))
                    
                neg_texts.append(self.preprocess_label(neg_text))
                
        if len(sent_texts) == 0:
            return None, None, None, None, None, None
        
        sent_res = self.tokenizer(sent_texts, padding=True, return_tensors="pt")
        sent_input_ids = sent_res["input_ids"].cuda()
        sent_attention_mask = sent_res["attention_mask"].cuda()
        
        pos_res = self.tokenizer(pos_texts, padding=True, return_tensors="pt")
        pos_input_ids = pos_res["input_ids"].cuda()
        pos_attention_mask = pos_res["attention_mask"].cuda()

        neg_res = self.tokenizer(neg_texts, padding=True, return_tensors="pt")
        neg_input_ids = neg_res["input_ids"].cuda()
        neg_attention_mask = neg_res["attention_mask"].cuda()
        
        return sent_input_ids, sent_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask
    
    def embed(self, input_ids, attention_mask=None):        
        outputs = self.base_model(input_ids, attention_mask=attention_mask, return_dict=True)
        embeddings = outputs["pooler_output"]
        return embeddings
    
    def dist_fn(self, sent_embeddings, label_embeddings):
        return 1.0 - F.cosine_similarity(sent_embeddings, label_embeddings)
        
    def forward(self, batch):
        # process data
        sent_input_ids, sent_attention_mask, pos_input_ids, pos_attention_mask, neg_input_ids, neg_attention_mask = self.process_data(batch)
        if sent_input_ids is None:
            return None
        
        sent_embeddings = self.embed(sent_input_ids, sent_attention_mask)
        pos_embeddings = self.embed(pos_input_ids, pos_attention_mask)
        neg_embeddings = self.embed(neg_input_ids, neg_attention_mask)
        
        loss_fn = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_fn, margin=self.config.margin)
        loss = loss_fn(sent_embeddings, pos_embeddings, neg_embeddings)

        return loss
    
    def predict(self, batch, batch_pred_spans):
        self.eval()
        with torch.no_grad():
            sent_texts = []
            for pred_spans, tokens in zip(batch_pred_spans, batch.batch_tokens):
                for pred_span in pred_spans:
                    sent_text = tokens[:pred_span[0]] + ["<t>"] + tokens[pred_span[0]:pred_span[1]]+ ["</t>"] + tokens[pred_span[1]:]
                    sent_text = " ".join(sent_text)
                    sent_texts.append(sent_text)
            n_sent_text = len(sent_texts)
            if n_sent_text == 0:
                return [[] for _ in range(len(batch_pred_spans))]
            
            sent_res = self.tokenizer(sent_texts, padding=True, return_tensors="pt")
            sent_input_ids = sent_res["input_ids"].cuda()
            sent_attention_mask = sent_res["attention_mask"].cuda()
            sent_embeddings = self.embed(sent_input_ids, sent_attention_mask)
            
            label_res = self.tokenizer(self.label_strings, padding=True, return_tensors="pt")
            label_input_ids = label_res["input_ids"].cuda()
            label_attention_mask = label_res["attention_mask"].cuda()
            label_embeddings = self.embed(label_input_ids, label_attention_mask)
            
            dists = np.zeros((n_sent_text, len(self.label_list)))
            
            for i in range(len(sent_embeddings)):
                sent_embedding = sent_embeddings[i].expand(label_embeddings.shape)
                dist = self.dist_fn(sent_embedding, label_embeddings).detach().cpu().numpy()
                dists[i] = dist
                
            preds = dists.argmin(axis=1)
            batch_pred_triggers = []
            idx = 0
            for pred_spans in batch_pred_spans:
                pred_triggers = []
                for pred_span in pred_spans:
                    pred_trigger = (pred_span[0], pred_span[1], self.label_list[preds[idx]])
                    pred_triggers.append(pred_trigger)
                    idx += 1
                batch_pred_triggers.append(pred_triggers)
                
        self.train()
            
        return batch_pred_triggers

class SpanModel(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
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
        self.base_model_dropout = nn.Dropout(p=self.config.span_base_model_dropout)
        
        # local classifiers
        self.dropout = nn.Dropout(p=self.config.span_linear_dropout)
        feature_dim = self.base_model_dim

        self.span_label_ffn = Linears([feature_dim, self.config.span_linear_hidden_num, len(self.label_stoi)],
                                      dropout_prob=self.config.span_linear_dropout, 
                                      bias=self.config.span_linear_bias, 
                                      activation=self.config.span_linear_activation)
        if self.config.span_use_crf:
            self.span_crf = CRF(self.label_stoi, bioes=False)
            
    def generate_tagging_vocab(self):
        prefix = ['B', 'I']
        span_label_stoi = {'O': 0}
        for t in ["Span"]:
            for p in prefix:
                span_label_stoi['{}-{}'.format(p, t)] = len(span_label_stoi)

        self.label_stoi = span_label_stoi
        self.type_stoi = {t: i for i, t in enumerate(["Span"])}
        
    def get_span_seqlabels(self, spans, token_num, specify_span=None):
        labels = ['O'] * token_num
        count = 0
        for span in spans:
            start, end = span[0], span[1]
            if end > token_num:
                continue
            span_type = span[2]

            if specify_span is not None:
                if span_type != specify_span:
                    continue

            if any([labels[i] != 'O' for i in range(start, end)]):
                count += 1
                continue

            labels[start] = 'B-{}'.format(span_type)
            for i in range(start + 1, end):
                labels[i] = 'I-{}'.format(span_type)
                
        return labels
    
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
    
    def tag_paths_to_spans(self, paths, token_nums, vocab):
        """
        Convert predicted tag paths to a list of spans (entity mentions or event
        triggers).
        :param paths: predicted tag paths.
        :return (list): a list (batch) of lists (sequence) of spans.
        """
        batch_mentions = []
        itos = {i: s for s, i in vocab.items()}
        for i, path in enumerate(paths):
            mentions = []
            cur_mention = None
            path = path.tolist()[:token_nums[i].item()]
            for j, tag in enumerate(path):
                tag = itos[tag]
                if tag == 'O':
                    prefix = tag = 'O'
                else:
                    prefix, tag = tag.split('-', 1)
                if prefix == 'B':
                    if cur_mention:
                        mentions.append(cur_mention)
                    cur_mention = [j, j + 1, tag]
                elif prefix == 'I':
                    if cur_mention is None:
                        # treat it as B-*
                        cur_mention = [j, j + 1, tag]
                    elif cur_mention[-1] == tag:
                        cur_mention[1] = j + 1
                    else:
                        # treat it as B-*
                        mentions.append(cur_mention)
                        cur_mention = [j, j + 1, tag]
                else:
                    if cur_mention:
                        mentions.append(cur_mention)
                    cur_mention = None
            if cur_mention:
                mentions.append(cur_mention)
            batch_mentions.append(mentions)
            
        return batch_mentions
    
    def process_data(self, batch):
        enc_idxs = []
        enc_attn = []
        span_seqidxs = []
        token_lens = []
        token_nums = []
        max_token_num = max(batch.batch_token_num)
        
        for tokens, pieces, spans, token_len, token_num in zip(batch.batch_tokens, batch.batch_pieces, batch.batch_spans, 
                                                                      batch.batch_token_lens, batch.batch_token_num):
            
            piece_id = self.tokenizer.convert_tokens_to_ids(pieces)
            enc_idx = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)] + piece_id + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)]
            
            enc_idxs.append(enc_idx)
            enc_attn.append([1]*len(enc_idx))  
            
            span_seq = self.get_span_seqlabels(spans, len(tokens))
            token_lens.append(token_len)
            token_nums.append(token_num)
            if self.config.span_use_crf:
                span_seqidxs.append([self.label_stoi[s] for s in span_seq] + [0] * (max_token_num-len(tokens)))
            else:
                span_seqidxs.append([self.label_stoi[s] for s in span_seq] + [-100] * (max_token_num-len(tokens)))
        max_len = max([len(enc_idx) for enc_idx in enc_idxs])
        enc_idxs = torch.LongTensor([enc_idx + [self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)]*(max_len-len(enc_idx)) for enc_idx in enc_idxs])
        enc_attn = torch.LongTensor([enc_att + [0]*(max_len-len(enc_att)) for enc_att in enc_attn])
        enc_idxs = enc_idxs.cuda()
        enc_attn = enc_attn.cuda()
        span_seqidxs = torch.cuda.LongTensor(span_seqidxs)
        return enc_idxs, enc_attn, span_seqidxs, token_lens, torch.cuda.LongTensor(token_nums)
        
    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_base_model_outputs = self.base_model(piece_idxs, attention_mask=attention_masks)
        base_model_outputs = all_base_model_outputs[0]
        if self.config.span_multi_piece_strategy == 'first':
            # select the first piece for multi-piece words
            offsets = token_lens_to_offsets(token_lens)
            offsets = piece_idxs.new(offsets) # batch x max_token_num
            # + 1 because the first vector is for [CLS]
            offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
            base_model_outputs = torch.gather(base_model_outputs, 1, offsets)
        elif self.config.span_multi_piece_strategy == 'average':
            # average all pieces for multi-piece words
            idxs, masks, token_num, token_len = self.token_lens_to_idxs(token_lens)
            idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.base_model_dim) + 1
            masks = base_model_outputs.new(masks).unsqueeze(-1)
            base_model_outputs = torch.gather(base_model_outputs, 1, idxs) * masks
            base_model_outputs = base_model_outputs.view(batch_size, token_num, token_len, self.base_model_dim)
            base_model_outputs = base_model_outputs.sum(2)
        else:
            raise ValueError(f'Unknown multi-piece token handling strategy: {self.config.multi_piece_strategy}')
        base_model_outputs = self.base_model_dropout(base_model_outputs)
        return base_model_outputs

    def span_id(self, base_model_outputs, token_nums, target=None, predict=False):
        loss = 0.0
        entities = None
        entity_label_scores = self.span_label_ffn(base_model_outputs)
        if self.config.span_use_crf:
            entity_label_scores_ = self.span_crf.pad_logits(entity_label_scores)
            if predict:
                _, entity_label_preds = self.span_crf.viterbi_decode(entity_label_scores_,
                                                                        token_nums)
                entities = self.tag_paths_to_spans(entity_label_preds, 
                                                   token_nums, 
                                                   self.label_stoi)
            else: 
                entity_label_loglik = self.span_crf.loglik(entity_label_scores_, 
                                                           target, 
                                                           token_nums)
                loss -= entity_label_loglik.mean()
        else:
            if predict:
                entity_label_preds = torch.argmax(entity_label_scores, dim=-1)
                entities = tag_paths_to_spans(entity_label_preds, 
                                              token_nums, 
                                              self.label_stoi)
            else:
                loss = F.cross_entropy(entity_label_scores.view(-1, self.span_label_num), target.view(-1))

        return loss, entities

    def forward(self, batch):
        # process data
        enc_idxs, enc_attn, span_seqidxs, token_lens, token_nums = self.process_data(batch)
        
        # encoding
        base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
        span_id_loss, _ = self.span_id(base_model_outputs, token_nums, span_seqidxs, predict=False)
        loss = span_id_loss

        return loss
    
    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            # process data
            enc_idxs, enc_attn, _, token_lens, token_nums = self.process_data(batch)
            
            # encoding
            base_model_outputs = self.encode(enc_idxs, enc_attn, token_lens)
            _, spans = self.span_id(base_model_outputs, token_nums, predict=True)
        self.train()
        return spans

    

def log_sum_exp(tensor, dim=0, keepdim: bool = False):
    """LogSumExp operation used by CRF."""
    m, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - m
    else:
        stable_vec = tensor - m.unsqueeze(dim)
    return m + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()

def sequence_mask(lens, max_len=None):
    """Generate a sequence mask tensor from sequence lengths, used by CRF."""
    batch_size = lens.size(0)
    if max_len is None:
        max_len = lens.max().item()
    ranges = torch.arange(0, max_len, device=lens.device).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp
    return mask
        
class CRF(nn.Module):
    def __init__(self, label_vocab, bioes=False):
        super(CRF, self).__init__()

        self.label_vocab = label_vocab
        self.label_size = len(label_vocab) + 2
        self.bioes = bioes

        self.start = self.label_size - 2
        self.end = self.label_size - 1
        transition = torch.randn(self.label_size, self.label_size)
        self.transition = nn.Parameter(transition)
        self.initialize()

    def initialize(self):
        self.transition.data[:, self.end] = -100.0
        self.transition.data[self.start, :] = -100.0

        for label, label_idx in self.label_vocab.items():
            if label.startswith('I-') or label.startswith('E-'):
                self.transition.data[label_idx, self.start] = -100.0
            if label.startswith('B-') or label.startswith('I-'):
                self.transition.data[self.end, label_idx] = -100.0

        for label_from, label_from_idx in self.label_vocab.items():
            if label_from == 'O':
                label_from_prefix, label_from_type = 'O', 'O'
            else:
                label_from_prefix, label_from_type = label_from.split('-', 1)

            for label_to, label_to_idx in self.label_vocab.items():
                if label_to == 'O':
                    label_to_prefix, label_to_type = 'O', 'O'
                else:
                    label_to_prefix, label_to_type = label_to.split('-', 1)

                if self.bioes:
                    is_allowed = any(
                        [
                            label_from_prefix in ['O', 'E', 'S']
                            and label_to_prefix in ['O', 'B', 'S'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix in ['I', 'E']
                            and label_from_type == label_to_type
                        ]
                    )
                else:
                    is_allowed = any(
                        [
                            label_to_prefix in ['B', 'O'],

                            label_from_prefix in ['B', 'I']
                            and label_to_prefix == 'I'
                            and label_from_type == label_to_type
                        ]
                    )
                if not is_allowed:
                    self.transition.data[
                        label_to_idx, label_from_idx] = -100.0

    def pad_logits(self, logits):
        """Pad the linear layer output with <SOS> and <EOS> scores.
        :param logits: Linear layer output (no non-linear function).
        """
        batch_size, seq_len, _ = logits.size()
        pads = logits.new_full((batch_size, seq_len, 2), -100.0,
                               requires_grad=False)
        logits = torch.cat([logits, pads], dim=2)
        return logits

    def calc_binary_score(self, labels, lens):
        batch_size, seq_len = labels.size()

        # A tensor of size batch_size * (seq_len + 2)
        labels_ext = labels.new_empty((batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=(seq_len + 2)).long()
        pad_stop = labels.new_full((1,), self.end, requires_grad=False)
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transition
        trn_exp = trn.unsqueeze(0).expand(batch_size, self.label_size,
                                          self.label_size)
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*lbl_r.size(), self.label_size)
        # score of jumping to a tag
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr

        return score

    def calc_unary_score(self, logits, labels, lens):
        """Checked"""
        labels_exp = labels.unsqueeze(-1)
        scores = torch.gather(logits, 2, labels_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        return scores

    def calc_gold_score(self, logits, labels, lens):
        """Checked"""
        unary_score = self.calc_unary_score(logits, labels, lens).sum(
            1).squeeze(-1)
        binary_score = self.calc_binary_score(labels, lens).sum(1).squeeze(-1)
        return unary_score + binary_score

    def calc_norm_score(self, logits, lens):
        batch_size, _, _ = logits.size()
        alpha = logits.new_full((batch_size, self.label_size), -100.0)
        alpha[:, self.start] = 0
        lens_ = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  self.label_size,
                                                  self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand_as(alpha_exp)
            mat = logit_exp + alpha_exp + trans_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (lens_ > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            lens_ = lens_ - 1

        alpha = alpha + self.transition[self.end].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def loglik(self, logits, labels, lens):
        norm_score = self.calc_norm_score(logits, lens)
        gold_score = self.calc_gold_score(logits, labels, lens)
        return gold_score - norm_score

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, _, n_labels = logits.size()
        vit = logits.new_full((batch_size, self.label_size), -100.0)
        vit[:, self.start] = 0
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transition.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transition[self.end].unsqueeze(
                0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def calc_conf_score_(self, logits, labels):
        batch_size, _, _ = logits.size()

        logits_t = logits.transpose(1, 0)
        scores = [[] for _ in range(batch_size)]
        pre_labels = [self.start] * batch_size
        for i, logit in enumerate(logits_t):
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   self.label_size,
                                                   self.label_size)
            trans_exp = self.transition.unsqueeze(0).expand(batch_size,
                                                            self.label_size,
                                                            self.label_size)
            score = logit_exp + trans_exp
            score = score.view(-1, self.label_size * self.label_size) \
                .softmax(1)
            for j in range(batch_size):
                cur_label = labels[j][i]
                cur_score = score[j][cur_label * self.label_size + pre_labels[j]]
                scores[j].append(cur_score)
                pre_labels[j] = cur_label
        return scores
    
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
