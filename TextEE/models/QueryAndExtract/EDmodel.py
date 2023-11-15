import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel
import numpy as np
from .metadata import Metadata
from .utils import prepare_bert_sequence, pad_sequences, bio_to_ids, prepare_sequence
import ipdb

class QueryAndExtractEDModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_pad_value = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.type_set = type_set 
        self.metadata = Metadata(config.metadata_path, self.config.dataset, type_set)
        self.meta_info = self.metadata.metadata

        self.event_count = len(self.meta_info.trigger_set)
        
        # load pre-trained embedding layer
        if config.pretrained_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(config.pretrained_model_name, output_attentions=True)
        elif config.pretrained_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(config.pretrained_model_name, output_attentions=True)
        else:
            raise NotImplementedError
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.embedding_dim = self.bert.config.hidden_size

        if config.use_pos_tag:
            self.linear = nn.Sequential(
                nn.Dropout(p=self.config.dropout),
                nn.Linear(self.embedding_dim*3+17, 2),
            )
        else:
            self.linear = nn.Sequential(
                nn.Dropout(p=self.config.dropout),
                nn.Linear(self.embedding_dim*3, 2),
            )
        self.sigmoid = nn.Sigmoid()
        self.W = nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim))

        self.last_k_hidden = config.last_k_hidden
        self.sqrt_d = np.sqrt(self.embedding_dim)
        self.cos = nn.CosineSimilarity(dim=-1)

        # loss
        weights = torch.ones(2).cuda()
        weights[0] = config.non_weight
        weights[1] = config.trigger_training_weight
        self.criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1, reduction='mean')

    def process_data(self, batch):
        
        # unpack data
        word_to_ix = self.tokenizer.convert_tokens_to_ids

        # general information: sent_len, bert_sent_len, first_index
        bert_sentence_lengths = [len(s) for s in batch.batch_bert_sent]
        max_bert_seq_length = int(max(bert_sentence_lengths))
        sentence_lengths = [len(x) for x in batch.batch_tokens]
        max_seq_length = int(max(sentence_lengths))
        bert_tokens = prepare_bert_sequence(batch.batch_bert_sent, word_to_ix, self.tokenizer.pad_token, max_bert_seq_length)
        # general information: pad_sequence
        idxs_to_collect_sent = pad_sequences(batch.batch_sent_idx_to_collect, dtype="long", truncating="post", padding="post")
        idxs_to_collect_sent = torch.Tensor(idxs_to_collect_sent)
        idxs_to_collect_event = pad_sequences(batch.batch_event_idx_to_collect, dtype="long", truncating="post", padding="post")
        idxs_to_collect_event = torch.Tensor(idxs_to_collect_event)

        sent_lengths = torch.Tensor(sentence_lengths).unsqueeze(1)
        pos_tags = prepare_sequence(batch.batch_pos_tag, self.metadata.pos2id, -1, max_seq_length)
        bert_sentence_lengths = torch.Tensor(bert_sentence_lengths)

        # trigger
        event_bio = batch.batch_trigger_bio
        for i in range(len(event_bio)):
            if len(event_bio[i]) == 1:
                event_bio[i] = event_bio[i][0]
            else:
                event_bio[i] = [min(np.array(event_bio[i])[:, j]) for j in range(len(event_bio[i][0]))]
        event_tags = bio_to_ids(event_bio, self.meta_info.triggers_to_ids, is_trigger=True)
    
        # unpack and truncate data
        embedding_length = int(torch.max(bert_sentence_lengths).item())
        bert_sentence_in = bert_tokens[:, :embedding_length]
        idxs_to_collect_sent = idxs_to_collect_sent[:, :embedding_length]
        sent_lengths = torch.sum(idxs_to_collect_sent, dim=1).int()
        max_sent_len = int(torch.max(sent_lengths).item())
        trigger_tags = event_tags[:, :max_sent_len]
        pos_tags = pos_tags[:, :max_sent_len]

        # send data to gpu
        tmp = [bert_sentence_in, idxs_to_collect_sent, idxs_to_collect_event,
            sent_lengths, pos_tags, trigger_tags]
        return [x.long().cuda() for x in tmp]

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect_sent, idxs_to_collect_event):
        """
        Pick first subword embeddings with the indices list idxs_to_collect
        :param all_embeddings:
        :param idxs_to_collect:
        :param sentence_lengths:
        :return:
        """
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        # Other two mode need to be taken care of the issue
        # that the last index becomes the [SEP] after argument types
        sent_embeddings = []
        event_embeddings = []

        for i in range(N):
            to_collect_sent, to_collect_event = idxs_to_collect_sent[i], idxs_to_collect_event[i]
            collected_sent = all_embeddings[i, torch.nonzero(to_collect_sent,as_tuple=False).squeeze(-1)]  # collecting a slice of tensor
            collected_event = all_embeddings[i, torch.nonzero(to_collect_event,as_tuple=False).squeeze(-1)]  # collecting a slice of tensor
            sent_embeddings.append(collected_sent)
            event_embeddings.append(collected_event)

        max_sent_len = torch.max(torch.sum(idxs_to_collect_sent, dim=-1))
        max_event_len = torch.max(torch.sum(idxs_to_collect_event, dim=-1))

        for i in range(N):
            try:
                assert max_sent_len >= len(sent_embeddings[i])
                assert max_event_len >= len(event_embeddings[i])
            except AssertionError:
                import ipdb
                ipdb.set_trace()
            sent_embeddings[i] = torch.cat((sent_embeddings[i], torch.zeros(max_sent_len - len(sent_embeddings[i]), self.embedding_dim).cuda()))
            event_embeddings[i] = torch.cat((event_embeddings[i], torch.zeros(max_event_len - len(event_embeddings[i]), self.embedding_dim).cuda()))
        sent_embeddings = torch.stack(sent_embeddings)
        event_embeddings = torch.stack(event_embeddings)
        return sent_embeddings, event_embeddings

    def forward(self, batch):
        # process data
        sentence_batch, idxs_to_collect_sent, idxs_to_collect_event, sent_lengths, pos_tag, triggers = self.process_data(batch)
        # bert embeddings
        attention_mask = (sentence_batch != self.tokenizer_pad_value) * 1
        token_type_ids = attention_mask.detach().clone()
        # find the indices of [SEP] after the event query
        if self.config.pretrained_model_name.startswith('bert-'):
            sep_idx = (sentence_batch == self.tokenizer.sep_token_id).nonzero(as_tuple=True)[1].reshape(sentence_batch.shape[0], -1)[:, 1]
            for i in range(sentence_batch.shape[0]):
                token_type_ids[i, :sep_idx[i]+1] = 0
            all_embeddings, _, hidden_layer_att = self.bert(sentence_batch,
                                                            token_type_ids=token_type_ids,
                                                            attention_mask=attention_mask,
                                                            return_dict=False)
        elif self.config.pretrained_model_name.startswith('roberta-'):
            all_embeddings, _, hidden_layer_att = self.bert(sentence_batch,
                                                            attention_mask=attention_mask,
                                                            return_dict=False)
        else:
            raise NotImplementedError
        all_embeddings = all_embeddings * attention_mask.unsqueeze(-1)

        # pick embeddings of sentence and event type
        sent_embeddings, event_embeddings = self.get_fist_subword_embeddings(all_embeddings,
                                                                             idxs_to_collect_sent,
                                                                             idxs_to_collect_event)

        # # use BERT hidden logits to calculate contextual embedding
        avg_layer_att = self.get_last_k_hidden_att(hidden_layer_att, idxs_to_collect_sent, self.last_k_hidden)
        context_emb = avg_layer_att.matmul(sent_embeddings)

        map_numerator = 1.0/torch.sum(idxs_to_collect_event>0, dim=-1).float().unsqueeze(-1).unsqueeze(-1).cuda()
        logits = self.cos(sent_embeddings.unsqueeze(2).matmul(self.W), event_embeddings.unsqueeze(1))
        sent2event_att = logits.matmul(event_embeddings) * map_numerator

        if self.config.use_pos_tag:
            pos_tag[pos_tag<0] = 16
            pos_tag = torch.nn.functional.one_hot(pos_tag, num_classes=17)
            _logits = self.linear(torch.cat((sent_embeddings, sent2event_att, context_emb, pos_tag[:, :context_emb.shape[1]]), dim=-1))
        else:
            _logits = self.linear(torch.cat((sent_embeddings, sent2event_att, context_emb), dim=-1))

        # Loss
        triggers = triggers.flatten()
        feats = torch.flatten(_logits, end_dim=-2)
        targets = triggers[triggers != self.event_count + 1]
        targets = (targets < self.event_count) * 1
        feats = feats[triggers != self.event_count + 1]
        loss = self.criterion(feats, targets)
        return _logits, sent_lengths, loss

    def predict(self, batch):
        logits, sent_lengths, _ = self.forward(batch)
        # get predictions from logits
        pred = ((logits[:,:,1] - logits[:,:,0] - self.config.trigger_threshold)>0)*1
        pred = [pred[k,:sent_lengths[k]] for k in range(self.event_count)]
        this_pred = list(set(self.pred_to_event_mention(pred, batch.batch_event_type, self.event_count)))
        return this_pred
            
    def pred_to_event_mention(self, pred, ids_to_triggers, event_count):
        ret = []
        for i in range(event_count):

            if not torch.any(pred[i]>0.5):
                continue

            temp = torch.cat([torch.tensor([0]).cuda(), pred[i], torch.tensor([0]).cuda()])
            is_event, begin, end = 0, None, None
            for j in range(len(temp)):
                if temp[j] and not is_event:
                    begin = j-1
                    is_event = 1
                if not temp[j] and is_event:
                    end = j-1
                    is_event = 0
                    ret.append((begin, end, ids_to_triggers[i]))
        return ret
    
    def select_hidden_att(self, hidden_att, idxs_to_collect):
        """
        Pick attentions from hidden layers
        :param hidden_att: of dimension (batch_size, embed_length, embed_length)
        :return:
        """
        N = hidden_att.shape[0]
        sent_len = torch.max(torch.sum(idxs_to_collect, dim=-1))
        hidden_att = torch.mean(hidden_att, 1)
        hidden_att_selected = torch.zeros(N, sent_len, sent_len)
        hidden_att_selected = hidden_att_selected.cuda()

        for i in range(N):
            to_collect = idxs_to_collect[i]
            to_collect = torch.nonzero(to_collect, as_tuple=False).squeeze(-1)
            collected = hidden_att[i, to_collect][:,to_collect]  # collecting a slice of tensor
            hidden_att_selected[i, :len(to_collect), :len(to_collect)] = collected

        return hidden_att_selected/(torch.sum(hidden_att_selected, dim=-1, keepdim=True)+1e-9)

    def get_last_k_hidden_att(self, hidden_layer_att, idxs_to_collect, k=3):
        tmp = 0
        for i in range(k):
            tmp += self.select_hidden_att(hidden_layer_att[-i], idxs_to_collect)
        avg_layer_att = tmp/k
        return avg_layer_att