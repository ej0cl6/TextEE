import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, RobertaModel
import numpy as np
from .metadata import Metadata
from .utils import pad_seq
from keras_preprocessing.sequence import pad_sequences
import ipdb

class QueryAndExtractEARLModel(nn.Module):
    def __init__(self, config, tokenizer, type_set):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.tokenizer_pad_value = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        self.type_set = type_set

        self.metadata = Metadata(config.metadata_path, self.config.dataset, type_set)
        self.meta_info = self.metadata.metadata
        self.arg_roles = len(self.meta_info.arg_set) 
        # ipdb.set_trace() # print arg roles num

        if config.pretrained_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(config.pretrained_model_name, output_attentions=True, output_hidden_states=True)
        elif config.pretrained_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(config.pretrained_model_name, output_attentions=True, output_hidden_states=True)
        else:
            raise NotImplementedError
        self.bert.resize_token_embeddings(len(self.tokenizer))
        
        self.embedding_dim = self.bert.config.hidden_size
        self.n_hid = config.n_hid
        self.dropout = config.dropout

        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.embedding_dim *= 2

        self.linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim*6, self.n_hid),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.n_hid, 1)
        )
        self.sqrt_d = np.sqrt(self.embedding_dim)
        # [Change]: add trigger aware
        self.triggerAware_linear = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim*3, self.embedding_dim)
        )
             
        # loss
        weights = torch.ones(self.arg_roles + 1).cuda()
        weights[-1] = self.config.non_weight
        self.criterion = torch.nn.CrossEntropyLoss(weight=weights, ignore_index=self.arg_roles + 1)
            
    def process_data(self, batch):
        sentence_batch = pad_seq(batch.sentence_batch, pad_value=self.tokenizer_pad_value).long()
        idxs_to_collect = pad_seq(batch.idxs_to_collect).long()
        is_triggers = pad_seq(batch.is_triggers)
        bert_sentence_lens = torch.tensor(batch.bert_sentence_lens).long().unsqueeze(1)
        arg_weight_matrices = torch.stack(list(map(torch.Tensor, batch.arg_weight_matrices))).cuda()
        arg_mapping = torch.Tensor(pad_sequences(batch.arg_mapping, maxlen=arg_weight_matrices.shape[-1], dtype="long",
                                                        truncating="post", padding="post",
                                                        value=self.meta_info.args_to_ids['[PAD]'])).long().cuda()
        entity_mapping = pad_seq(batch.entity_mapping, dtype='float32')
        arg_tags = pad_seq(batch.arg_tags, self.meta_info.args_to_ids['[PAD]'])
        return sentence_batch, idxs_to_collect, is_triggers, bert_sentence_lens, arg_weight_matrices, arg_mapping, entity_mapping, arg_tags

    def get_fist_subword_embeddings(self, all_embeddings, idxs_to_collect, bert_sentence_lengths, max_len_arg=None):
        """
        Pick first subword embeddings with the indices list idxs_to_collect
        :param all_embeddings:
        :param idxs_to_collect:
        :param targets:
        :param bert_sentence_lengths:
        :return:
        """
        sent_embeddings = []
        N = all_embeddings.shape[0]  # it's equivalent to N=len(all_embeddings)

        # Other two mode need to be taken care of the issue
        # that the last index becomes the [SEP] after argument types
        arg_type_embeddings = []
        bert_sentence_lengths = bert_sentence_lengths.long()
        for i in range(N):
            this_idxs_to_collect = idxs_to_collect[i]
            this_idxs_to_collect = this_idxs_to_collect[this_idxs_to_collect>0]
            collected = all_embeddings[i, this_idxs_to_collect[:-2]]  # collecting a slice of tensor
            sent_embeddings.append(collected)
            second_last_sep_index = this_idxs_to_collect[-2]

            # argument type embedding
            if self.config.pretrained_model_name.startswith('bert'):
                arg_type_embedding = all_embeddings[i, second_last_sep_index+1:bert_sentence_lengths[i]-1]
            elif self.config.pretrained_model_name.startswith('roberta'):
                arg_type_embedding = all_embeddings[i, second_last_sep_index+2:bert_sentence_lengths[i]-1]
            else:
                raise ValueError("Not implemented.")
            arg_type_embeddings.append(arg_type_embedding)

        max_sent_len = idxs_to_collect.shape[1] - 2

        for i in range(N):
            try:
                assert max_len_arg >= len(arg_type_embeddings[i])
                assert max_sent_len >= len(sent_embeddings[i])
            except AssertionError:
                import ipdb
                ipdb.set_trace()
            arg_type_embeddings[i] = torch.cat((arg_type_embeddings[i], torch.zeros(max_len_arg - len(arg_type_embeddings[i]), self.embedding_dim).cuda()))
            sent_embeddings[i] = torch.cat((sent_embeddings[i], torch.zeros(max_sent_len - len(sent_embeddings[i]), self.embedding_dim).cuda()))

        sent_embeddings = torch.stack(sent_embeddings)
        arg_type_embeddings = torch.stack(arg_type_embeddings)
        return sent_embeddings, arg_type_embeddings

    @staticmethod
    def get_trigger_embeddings(sent_embeddings, is_triggers):
        """
        Select trigger embedding with the is_trigger mask
        :param sent_embeddings:
        :param is_triggers:
        :return:
        """
        return torch.sum(sent_embeddings*is_triggers.unsqueeze(-1)/torch.sum(is_triggers, dim=1).unsqueeze(-1).unsqueeze(-1), dim=1)

    def get_triggerAwared_entity(self, entity_embeddings, trigger_embeddings):
        non_empty_mask = entity_embeddings.abs().sum(dim=-1).bool()
        entity_cat = torch.cat((entity_embeddings, trigger_embeddings, entity_embeddings*trigger_embeddings), -1)
        return self.triggerAware_linear(entity_cat) * non_empty_mask.unsqueeze(-1)

    def normalize(self, mat, dim):
        return F.normalize(mat, p=1, dim=dim).unsqueeze(-1)

    def forward(self, batch):
        # process data
        sentence_batch, idxs_to_collect, is_triggers, bert_sentence_lengths, arg_weight_matrices, arg_mapping, entity_mapping, arg_tags = self.process_data(batch)

        # get embeddings
        sent_mask = (sentence_batch != self.tokenizer_pad_value) * 1
        all_embeddings, _, hidden_states, hidden_layer_att = self.bert(sentence_batch.long(), attention_mask=sent_mask, return_dict=False)
        
        if self.use_extra_bert:
            extra_bert_outputs = hidden_states[self.extra_bert]
            all_embeddings = torch.cat([all_embeddings, extra_bert_outputs], dim=2)

        sent_embeddings, arg_embeddings = self.get_fist_subword_embeddings(all_embeddings, idxs_to_collect, bert_sentence_lengths, arg_weight_matrices.size(1))
        entity_embeddings = sent_embeddings.permute(0, 2, 1).matmul( entity_mapping).permute(0,2,1)
        trigger_candidates = self.get_trigger_embeddings(sent_embeddings, is_triggers)

        arg_embeddings = arg_embeddings.transpose(1,2).matmul(arg_weight_matrices.float()).transpose(1, 2)
        _trigger = trigger_candidates.unsqueeze(1).repeat(1, entity_embeddings.shape[1], 1)
        # [Change] process trigger embedding
        entity_embeddings = self.get_triggerAwared_entity(entity_embeddings, _trigger)

        # token to argument attention
        token2arg_score = torch.sum(entity_embeddings.unsqueeze(2) * arg_embeddings.unsqueeze(1), dim=-1) * (1 / self.sqrt_d)

        # [Change] attention weights 
        # token2arg_softmax = (token2arg_score/10).unsqueeze(-1) # This is original code.
        # arg2token_softmax = (token2arg_score/10).unsqueeze(-1) # This is original code.
        token2arg_softmax = self.normalize(token2arg_score, dim=2) # normalize args
        arg2token_softmax = self.normalize(token2arg_score, dim=1) # normalize entities

        token_argAwared = torch.sum(arg_embeddings.unsqueeze(1) * token2arg_softmax, dim=2)   # b * sent_len * 768
        arg_tokenAwared = torch.sum(entity_embeddings.unsqueeze(2) * arg2token_softmax, dim=1)  # b *  arg_len * 768

        # bidirectional attention
        A_h2u = token_argAwared.unsqueeze(2).repeat(1,1,arg_embeddings.shape[1],1)
        A_u2h = arg_tokenAwared.unsqueeze(1).repeat(1,entity_embeddings.shape[1],1,1)
        # argumentation embedding
        U_ = arg_embeddings.unsqueeze(1).repeat(1,entity_embeddings.shape[1],1,1)

        # entity-entity attention
        last0_layer_atten = self.select_hidden_att(hidden_layer_att[-1], idxs_to_collect)
        last1_layer_atten = self.select_hidden_att(hidden_layer_att[-2], idxs_to_collect)
        last2_layer_atten = self.select_hidden_att(hidden_layer_att[-3], idxs_to_collect)
        token2token_softmax = (last0_layer_atten + last1_layer_atten + last2_layer_atten)/3
        A_h2h = token2token_softmax.matmul(sent_embeddings).unsqueeze(2).repeat(1, 1, arg_embeddings.shape[1], 1)
        H_ = sent_embeddings.unsqueeze(2).repeat(1, 1, arg_embeddings.shape[1],1)
        A_h2h = A_h2h.permute(0, 2, 3, 1).matmul(entity_mapping.unsqueeze(1)).permute(0, 3, 1, 2)
        H_ = H_.permute(0, 2, 3, 1).matmul(entity_mapping.unsqueeze(1)).permute(0, 3, 1, 2)

        # arg role to arg role attention
        arg2arg_softmax = F.softmax(arg_embeddings.matmul(arg_embeddings.transpose(-1,-2)), dim=-1)
        A_u2u = arg2arg_softmax.matmul(arg_embeddings).unsqueeze(1).repeat(1, entity_embeddings.shape[1], 1, 1)

        latent = torch.cat((H_, U_, A_h2u, A_h2h, A_u2h, A_u2u), dim=-1)
        score = self.linear(latent).squeeze(3) # [bz, entity_num, arg_length]
        score = self.map_arg_to_ids(score, arg_mapping)
        
        # calc loss
        feats = score[:, :arg_tags.shape[1]]
        logits_padded = feats.flatten(start_dim=0, end_dim=-2)
        targets = arg_tags.flatten().long()
        loss = self.criterion(logits_padded, targets)

        return loss, score

    def predict(self, batch):
        _, all_feats = self.forward(batch)
        entity_mappings = pad_seq(batch.entity_mapping, dtype='float32') # process_data() only for entity_mapping
        all_arg_preds = []
        for feats, arguments, entity_mapping in zip(all_feats, batch.batch_arguments, entity_mappings):
            pred = torch.argmax(feats, dim=-1)
            pred += 1
            pred = (pred * torch.round(torch.sum(entity_mapping, dim=0)).long() - 1).long()
            # write arg predictionds
            this_pred_args = []
            for j, argument in enumerate(arguments):
                if pred[j] < self.meta_info.args_to_ids['O']:
                    # format (start_id, end_id, arg_role, arg_text)
                    this_pred_args.append((argument[0], argument[1], self.meta_info.ids_to_args[pred[j].item()], argument[3]))
                else:
                    this_pred_args.append((argument[0], argument[1], None, argument[3]))
            all_arg_preds.append(this_pred_args)
        return all_arg_preds

    def map_arg_to_ids(self, score, arg_mapping):
        """
        Here we put each argument embedding back to its original place.
        In the input [CLS] sentence [SEP] arguments [SEP],
        arguments contains arguments of the specific trigger type.
        Thus we need to put them back to their actual indices
        :param score:
        :param arg_mapping:
        :return:
        """
        b, s, _ = score.shape
        d = self.arg_roles+1
        new_score = -1e6 * torch.ones(b, s, d).cuda()
        for i in range(b):
            ids = arg_mapping[i][arg_mapping[i] < self.arg_roles+1]
            new_score[i, :, ids] = score[i, :, :len(ids)]
        return new_score

    @staticmethod
    def select_hidden_att(hidden_att, ids_to_collect):
        """
        Pick attentions from hidden layers
        :param hidden_att: of dimension (batch_size, embed_length, embed_length)
        :return:
        """
        N = hidden_att.shape[0]
        sent_len = ids_to_collect.shape[1] - 2
        hidden_att = torch.mean(hidden_att, 1)
        hidden_att_selected = torch.zeros(N, sent_len, sent_len).cuda()

        for i in range(N):
            to_collect = ids_to_collect[i]
            to_collect = to_collect[to_collect>0][:-2]
            collected = hidden_att[i, to_collect][:,to_collect]  # collecting a slice of tensor
            hidden_att_selected[i, :len(to_collect), :len(to_collect)] = collected

        return hidden_att_selected/(torch.sum(hidden_att_selected, dim=-1, keepdim=True)+1e-9)