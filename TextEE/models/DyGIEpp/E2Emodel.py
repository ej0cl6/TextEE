import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, RobertaModel, XLMRobertaModel, AutoModel, BertConfig, RobertaConfig, XLMRobertaConfig
from collections import Counter
import copy
from .graph import Graph
from .util import enumerate_spans, graph_add_fake_entity, graph_add_fake_trigger
import ipdb

def get_trigger_position(trigger_idx, sequence_length) -> list:
    """
    trigger_idx[0] = 5, trigger_idx[1] = 6, sequence_length = 10
    returns: [-5, -4, -3, -2, -1, 0, 0, 1, 2, 3]
    """
    return list(range(-trigger_idx[0], 0)) + \
            [0] * (trigger_idx[1] - trigger_idx[0] + 1) + \
            list(range(1, sequence_length - trigger_idx[1]))
            
def get_argument_position(argument_idx, sequence_length) -> list:
    return list(range(-argument_idx[0], 0)) + \
            [0] * (argument_idx[1] - argument_idx[0] + 1) + \
            list(range(1, sequence_length - argument_idx[1]))

def token_lens_to_idxs(token_lens):
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
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens))) # map to BOS/CLS token
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

def graphs_to_node_idxs(graphs, max_entity_dist, max_trigger_dist):
    """
    :param graphs (list): A list of Graph objects.
    :return: entity/trigger index matrix, mask tensor, max number, max length
    """
    ent_start_idxs, ent_end_idxs, entity_dists, entity_masks = [], [], [], []
    tri_start_idxs, tri_end_idxs, trigger_dists, trigger_masks = [], [], [], []
    tri_full_span_idxs, tri_full_span_masks = [], []
    entity_types, trigger_types = [], []
    max_entity_num = max(max(graph.entity_num for graph in graphs), 1)
    max_trigger_num = max(max(graph.trigger_num for graph in graphs), 1)
    max_trigger_len = max(max([t[1] - t[0] for t in graph.triggers] + [1])
                          for graph in graphs)

    for graph in graphs:
        seq_ent_start_idxs, seq_ent_end_idxs, seq_ent_dists = [], [], []
        seq_tri_start_idxs, seq_tri_end_idxs, seq_tri_dists = [], [], []
        seq_tri_full_span_idxs, seq_tri_full_span_masks = [], []
        entity_type, trigger_type = [], []
        for entity in graph.entities:
            seq_ent_start_idxs.append(entity[0])
            seq_ent_end_idxs.append(entity[1]-1)
            entity_type.append(entity[2])
            seq_ent_dists.append(min(max_entity_dist, (entity[1]-entity[0])))
            
        seq_ent_start_idxs.extend([0] * (max_entity_num - len(entity_type)))
        seq_ent_end_idxs.extend([0] * (max_entity_num - len(entity_type)))
        seq_ent_dists.extend([0] * (max_entity_num - len(entity_type)))
        
        entity_masks.append([1]*len(entity_type)+[0]*(max_entity_num - len(entity_type)))
        entity_type.extend([-100]* (max_entity_num - len(entity_type)))
        entity_types.append(entity_type)
        ent_start_idxs.append(seq_ent_start_idxs)
        ent_end_idxs.append(seq_ent_end_idxs)
        entity_dists.append(seq_ent_dists)


        for trigger in graph.triggers:
            seq_tri_start_idxs.append(trigger[0])
            seq_tri_end_idxs.append(trigger[1]-1)
            trigger_type.append(trigger[2])
            seq_tri_dists.append(min(max_trigger_dist, (trigger[1]-trigger[0])))

            trigger_len = trigger[1] - trigger[0]
            seq_tri_full_span_idxs.extend([i for i in range(trigger[0], trigger[1])])
            seq_tri_full_span_idxs.extend([0] * (max_trigger_len - trigger_len))
            seq_tri_full_span_masks.extend([1.0 / trigger_len] * trigger_len)
            seq_tri_full_span_masks.extend([0.0] * (max_trigger_len - trigger_len))

        seq_tri_start_idxs.extend([0] * (max_trigger_num - len(trigger_type)))
        seq_tri_end_idxs.extend([0] * (max_trigger_num - len(trigger_type)))
        seq_tri_dists.extend([0] * (max_trigger_num - len(trigger_type)))
        seq_tri_full_span_idxs.extend([0] * max_trigger_len * (max_trigger_num - len(trigger_type)))
        seq_tri_full_span_masks.extend([0.0] * max_trigger_len * (max_trigger_num - len(trigger_type)))

        trigger_masks.append([1]*len(trigger_type)+[0]*(max_trigger_num - len(trigger_type)))
        trigger_type.extend([-100]* (max_trigger_num - len(trigger_type)))
        trigger_types.append(trigger_type)
        tri_start_idxs.append(seq_tri_start_idxs)
        tri_end_idxs.append(seq_tri_end_idxs)
        trigger_dists.append(seq_tri_dists)
        tri_full_span_idxs.append(seq_tri_full_span_idxs)
        tri_full_span_masks.append(seq_tri_full_span_masks)

    return (
        ent_start_idxs, ent_end_idxs, entity_dists, entity_masks, entity_types,
        tri_start_idxs, tri_end_idxs, trigger_dists, trigger_masks, trigger_types,
        tri_full_span_idxs, tri_full_span_masks, max_trigger_num
    )

class Linears(nn.Module):
    """Multiple linear layers with Dropout."""
    def __init__(self, dimensions, activation='relu', dropout_prob=0.0, bias=True):
        super().__init__()
        assert len(dimensions) > 1
        self.num_layer = len(dimensions) - 1
        self.layers = nn.ModuleList([nn.Linear(dimensions[i], dimensions[i + 1], bias=bias)
                                     for i in range(len(dimensions) - 1)])
        self.activation = getattr(torch, activation)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs):
        outputs = []
        for i, layer in enumerate(self.layers):
            inputs = layer(inputs)
            if i < (self.num_layer-1):
                inputs = self.activation(inputs)
                inputs = self.dropout(inputs)
            outputs.append(inputs)
        return outputs

class DyGIEppE2EModel(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()
        # vocabularies
        self.vocabs = vocabs
        self.config = config
        self.entity_type_stoi = vocabs['entity_type']
        self.trigger_type_stoi = vocabs['event_type']
        self.relation_type_stoi = vocabs['relation_type']
        self.role_type_stoi = vocabs['role_type']
        self.entity_type_itos = vocabs['entity_type_itos']
        self.trigger_type_itos = vocabs['event_type_itos']
        self.relation_type_itos = vocabs['relation_type_itos']
        self.role_type_itos = vocabs['role_type_itos']

        self.entity_type_num = len(self.entity_type_stoi)
        self.trigger_type_num = len(self.trigger_type_stoi)
        self.relation_type_num = len(self.relation_type_stoi)
        self.role_type_num = len(self.role_type_stoi)

        # BERT encoder
        self.pretrained_model_name = config.pretrained_model_name
        self.cache_dir = config.cache_dir
        if self.pretrained_model_name.startswith('bert-'):
            self.bert = BertModel.from_pretrained(self.pretrained_model_name,
                                                  cache_dir=self.cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = BertConfig.from_pretrained(self.pretrained_model_name,
                                              cache_dir=self.cache_dir)
        elif self.pretrained_model_name.startswith('roberta-'):
            self.bert = RobertaModel.from_pretrained(self.pretrained_model_name,
                                                  cache_dir=self.cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = RobertaConfig.from_pretrained(self.pretrained_model_name,
                                                 cache_dir=self.cache_dir)
        elif self.pretrained_model_name.startswith('xlm-'):
            self.bert = XLMRobertaModel.from_pretrained(self.pretrained_model_name,
                                                  cache_dir=self.cache_dir,
                                                  output_hidden_states=True)
            self.bert_config = XLMRobertaConfig.from_pretrained(self.pretrained_model_name,
                                                 cache_dir=self.cache_dir)  
        else:
            raise ValueError
        
        self.bert_dim = self.bert_config.hidden_size
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)

        # local classifiers
        linear_bias = config.linear_bias
        linear_dropout = config.linear_dropout
        entity_hidden_num = config.entity_hidden_num
        trigger_hidden_num = config.trigger_hidden_num
        relation_hidden_num = config.relation_hidden_num
        role_hidden_num = config.role_hidden_num
        
        # Note that based on the scrip provided in https://github.com/dwadden/dygiepp/blob/master/training_config/template.libsonnet#L92 & https://github.com/dwadden/dygiepp/blob/master/doc/config.md#changing-arbitrary-parts-of-the-template. The DyGIE++ is trained without coref and span propagation by default. So we directly ignore the propagation layer and coref module in this re-implementation

        # span id
        self.max_entity_span = config.max_entity_span
        self.min_entity_span = config.min_entity_span
        self.max_trigger_span = config.max_trigger_span
        self.min_trigger_span = config.min_trigger_span

        # span width embedding
        self.ent_span_width_emd = nn.Embedding(config.max_entity_span+1, config.feature_size)
        self.tri_span_width_emd = nn.Embedding(config.max_trigger_span+1, config.feature_size)
        
        # span classification
        # The layer setting is following https://github.com/dwadden/dygiepp/blob/master/training_config/template.libsonnet#L83
        self.entity_type_ffn = Linears([self.bert_dim*2 + config.feature_size, # start + end + width feature
                                       entity_hidden_num, entity_hidden_num,
                                       self.entity_type_num],
                                       dropout_prob=linear_dropout,
                                       bias=linear_bias,
                                       activation=config.linear_activation)

        self.trigger_type_ffn = Linears([self.bert_dim*2 + config.feature_size, # start + end + width feature
                                        trigger_hidden_num, trigger_hidden_num,
                                        self.trigger_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)
        # self.trigger_type_ffn = Linears([self.bert_dim,
        #                                 trigger_hidden_num, trigger_hidden_num,
        #                                 self.trigger_type_num],
        #                                 dropout_prob=linear_dropout,
        #                                 bias=linear_bias,
        #                                 activation=config.linear_activation)

        # relation classifier
        self.relation_type_ffn = Linears([(self.bert_dim*2 + config.feature_size) *3, # based on https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/relation.py#L54
                                        relation_hidden_num, relation_hidden_num,
                                        self.relation_type_num],
                                        dropout_prob=linear_dropout,
                                        bias=linear_bias,
                                        activation=config.linear_activation)

        
        # argument classifier
        self.distance_embedding = nn.Embedding(11, config.feature_size) # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/events.py#L94
        self.role_type_ffn = Linears([(self.bert_dim*2 + config.feature_size) *2 + config.feature_size +2, # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/events.py#L83
                                    role_hidden_num, role_hidden_num,
                                    self.role_type_num],
                                    dropout_prob=linear_dropout,
                                    bias=linear_bias,
                                    activation=config.linear_activation)
        # self.role_type_ffn = Linears([(self.bert_dim*3 + config.feature_size) + config.feature_size +2, # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/events.py#L83
        #                             role_hidden_num, role_hidden_num,
        #                             self.role_type_num],
        #                             dropout_prob=linear_dropout,
        #                             bias=linear_bias,
        #                             activation=config.linear_activation)        

        # loss functions
        self.entity_criteria = torch.nn.CrossEntropyLoss()
        self.trigger_criteria = torch.nn.CrossEntropyLoss()
        self.relation_criteria = torch.nn.CrossEntropyLoss()
        self.role_criteria = torch.nn.CrossEntropyLoss()

    def encode(self, piece_idxs, attention_masks, token_lens):
        """Encode input sequences with BERT
        :param piece_idxs (LongTensor): word pieces indices
        :param attention_masks (FloatTensor): attention mask
        :param token_lens (list): token lengths
        """
        batch_size, _ = piece_idxs.size()
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]

        # average all pieces for multi-piece words
        idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
        idxs = piece_idxs.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
        masks = bert_outputs.new(masks).unsqueeze(-1)
        bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
        bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
        bert_outputs = bert_outputs.sum(2)

        bert_outputs = self.bert_dropout(bert_outputs)

        return bert_outputs

    def span_id(self, batch, vocabs, gold_tri=False, gold_ent=False):
        graphs = []
        for gold_graph, token in zip(batch.graphs, batch.tokens):
            graph = Graph([], [], [], [], vocabs, gold=False)
            
            if gold_ent:
                for ent in gold_graph.entities:
                    if ent[3]:
                        graph.add_entity(ent[0], ent[1], ent[2], gold=True)
            else:
                entity_spans = enumerate_spans(token, offset=0,
                                               max_span_width=self.max_entity_span, 
                                               min_span_width=self.min_entity_span)
                graph_add_fake_entity(entity_spans, graph, vocabs)
            
            if gold_tri:
                for tri in gold_graph.triggers:
                    if tri[3]:
                        graph.add_trigger(tri[0], tri[1], tri[2], gold=True)
            else:
                trigger_spans = enumerate_spans(token, offset=0,
                                                max_span_width=self.max_trigger_span, 
                                                min_span_width=self.min_trigger_span)
                graph_add_fake_trigger(trigger_spans, graph, vocabs)
                
            
            graphs.append(graph)

        return graphs

    def span_classification(self, bert_outputs, graphs):
        (
            ent_start_idxs, ent_end_idxs, entity_dists, entity_masks, entity_types,
            tri_start_idxs, tri_end_idxs, trigger_dists, trigger_masks, trigger_types,
            tri_full_span_idxs, tri_full_span_masks, max_trigger_num
        ) = graphs_to_node_idxs(graphs, self.max_entity_span, self.max_trigger_span)
        entity_types = bert_outputs.new_tensor(data=entity_types).long()
        trigger_types = bert_outputs.new_tensor(data=trigger_types).long()
        batch_size, _, bert_dim = bert_outputs.size()
        
        ent_start_idxs = bert_outputs.new_tensor(ent_start_idxs, dtype=torch.long)
        ent_end_idxs = bert_outputs.new_tensor(ent_end_idxs, dtype=torch.long)
        entity_dists = bert_outputs.new_tensor(entity_dists, dtype=torch.long)
        entity_masks = bert_outputs.new_tensor(entity_masks, dtype=torch.bool)

        # get span representation for span candidates
        ent_start_idxs = ent_start_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        ent_end_idxs = ent_end_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        ent_start_emb = torch.gather(bert_outputs, 1, ent_start_idxs) # batch x max_ent_num x bert_dim
        ent_end_emb = torch.gather(bert_outputs, 1, ent_end_idxs)
        ent_width_emb = self.ent_span_width_emd(entity_dists)
        ent_span_emb = torch.cat([ent_start_emb, ent_end_emb, ent_width_emb], dim=-1)

        tri_start_idxs = bert_outputs.new_tensor(tri_start_idxs, dtype=torch.long)
        tri_end_idxs = bert_outputs.new_tensor(tri_end_idxs, dtype=torch.long)
        trigger_dists = bert_outputs.new_tensor(trigger_dists, dtype=torch.long)
        trigger_masks = bert_outputs.new_tensor(trigger_masks, dtype=torch.bool)

        tri_start_idxs = tri_start_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        tri_end_idxs = tri_end_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        tri_start_emb = torch.gather(bert_outputs, 1, tri_start_idxs) # batch x max_ent_num x bert_dim
        tri_end_emb = torch.gather(bert_outputs, 1, tri_end_idxs)
        tri_width_emb = self.tri_span_width_emd(trigger_dists)
        tri_span_emb = torch.cat([tri_start_emb, tri_end_emb, tri_width_emb], dim=-1)
        
        # tri_full_span_idxs = bert_outputs.new_tensor(tri_full_span_idxs, dtype=torch.long)
        # tri_full_span_masks = bert_outputs.new_tensor(tri_full_span_masks)
        # tri_full_span_idxs = tri_full_span_idxs.unsqueeze(-1).expand(-1, -1, bert_dim)
        # tri_full_span_masks = tri_full_span_masks.unsqueeze(-1).expand(-1, -1, bert_dim)
        # trigger_words = torch.gather(bert_outputs, 1, tri_full_span_idxs)
        # trigger_words = trigger_words * tri_full_span_masks
        # trigger_words = trigger_words.view(batch_size, max_trigger_num, -1, bert_dim)
        # tri_span_emb = trigger_words.sum(2)
        
        # pass to classifer
        entity_type_scores = self.entity_type_ffn(ent_span_emb)[-1]
        trigger_type_scores = self.trigger_type_ffn(tri_span_emb)[-1]
        
        return (
            ent_span_emb, entity_masks, entity_type_scores, entity_types,
            tri_span_emb, trigger_masks, trigger_type_scores, trigger_types
        )

    def relation_classification(self, graphs):
        subj_repres = []
        obj_repres = []
        relation_labels = []
        for graph in graphs:
            for rel in graph.relations:
                subj_idx = graph.entity_map[(rel[0][0], rel[0][1])]
                obj_idx = graph.entity_map[(rel[1][0], rel[1][1])]
                subj_repres.append(graph.entity_emb[subj_idx])
                obj_repres.append(graph.entity_emb[obj_idx])
                relation_labels.append(rel[2])
        if len(subj_repres) > 0:
            subj_repres = torch.stack(subj_repres, dim=0)
            obj_repres = torch.stack(obj_repres, dim=0)
            relation_labels = torch.cuda.LongTensor(relation_labels)
            # the score calculation can be seen 
            # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/relation.py#L224 and
            # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/relation.py#L204
            similarity_embeddings = torch.mul(subj_repres, obj_repres)
            pair_embeddings = torch.cat([subj_repres, obj_repres, similarity_embeddings], dim=-1)
            
            # pass through classifier
            relation_distribution = self.relation_type_ffn(pair_embeddings)[-1]
        else:
            relation_distribution = None
        return relation_distribution, relation_labels

    def argument_classification(self, graphs):
        tri_repres = []
        argu_repres = []
        role_labels = []
        dists = []
        trigger_before_features = []
        trigger_inside_features = []
        for graph in graphs:
            for role in graph.roles:
                tri_idx = graph.trigger_map[(role[0][0], role[0][1])]
                argu_idx = graph.entity_map[(role[1][0], role[1][1])]
                tri_repres.append(graph.trigger_emb[tri_idx])
                argu_repres.append(graph.entity_emb[argu_idx])
                role_labels.append(role[2])
                dist = min(abs(role[0][0] - role[1][0]), abs(role[0][0] - role[1][1]), 10)
                dists.append(torch.cuda.LongTensor([dist]))
                if role[0][0] < role[1][0]:
                    trigger_before_features.append(torch.cuda.FloatTensor([1.0]))
                else:
                    trigger_before_features.append(torch.cuda.FloatTensor([0.0]))
                if (role[1][0] <= role[0][0]) and (role[0][1] <= role[1][1]):
                    trigger_inside_features.append(torch.cuda.FloatTensor([1.0]))
                else:
                    trigger_inside_features.append(torch.cuda.FloatTensor([0.0]))
        if len(tri_repres) > 0:
            tri_repres = torch.stack(tri_repres, dim=0)
            argu_repres = torch.stack(argu_repres, dim=0)
            role_labels = torch.cuda.LongTensor(role_labels)
            dists = torch.cat(dists, dim=0)
            trigger_before_features = torch.stack(trigger_before_features, dim=0)
            trigger_inside_features = torch.stack(trigger_inside_features, dim=0)

            # the score calculation can be seen 
            # https://github.com/dwadden/dygiepp/blob/5d7e7d58367c1ec74e653d9abf299d6b103e20de/dygie/models/events.py#L287
            dists_emb = self.distance_embedding(dists)
            pair_embeddings = torch.cat([tri_repres, argu_repres, dists_emb, trigger_before_features, trigger_inside_features], dim=-1)
            # pass through classifier
            role_distribution = self.role_type_ffn(pair_embeddings)[-1]
        else:
            role_distribution = None
        return role_distribution, role_labels

    def forward(self, batch, logger=None, tag=None, step=None):
        # get representation from PLM for each tokens
        bert_outputs = self.encode(batch.piece_idxs,
                                   batch.attention_masks,
                                   batch.token_lens)
        batch_size, _, _ = bert_outputs.size()

        node_graphs = copy.deepcopy(batch.graphs)
        # span classification (NER & Trigger prediction)
        (
            ent_span_emb, entity_masks, entity_type_scores, entity_types,
            tri_span_emb, trigger_masks, trigger_type_scores, trigger_types
        )  = self.span_classification(bert_outputs, node_graphs)
        # get span cls loss
        entity_types = entity_types.view(-1)
        trigger_types = trigger_types.view(-1)
        entity_type_scores_ = entity_type_scores.view(-1, self.entity_type_num)
        trigger_type_scores_ = trigger_type_scores.view(-1, self.trigger_type_num)
        entity_type_loss = self.entity_criteria(entity_type_scores_, entity_types)
        trigger_type_loss = self.trigger_criteria(trigger_type_scores_, trigger_types)
        if logger:
            logger.scalar_summary(tag+'/entity loss', entity_type_loss, step)
            logger.scalar_summary(tag+'/trigger loss', trigger_type_loss, step)

        # assign node scores so easier for doing span pruning
        node_graphs = self.assign_node_score(node_graphs, entity_masks, entity_type_scores, trigger_masks, trigger_type_scores, ent_span_emb, tri_span_emb, predict=False)

        # span pruning to create graph candidates for relation and role prediction
        node_graphs = self.span_pruning(node_graphs, bert_outputs, batch.token_nums)
    
        # relation classification
        relation_distribution, relation_labels = self.relation_classification(node_graphs)
        
        # argument classification
        role_distribution, role_labels = self.argument_classification(node_graphs)

        loss = self.config.ner_loss_weight*entity_type_loss + self.config.trigger_loss_weight*trigger_type_loss

        # calculate loss
        if len(relation_labels) > 0:
            relation_type_loss = self.relation_criteria(relation_distribution, relation_labels)
            loss += self.config.relation_loss_weight*relation_type_loss
            if logger:
                logger.scalar_summary(tag+'/relation loss', relation_type_loss, step)
        if len(role_labels) > 0:
            role_type_loss = self.role_criteria(role_distribution, role_labels)
            loss += self.config.role_loss_weight*role_type_loss
            if logger:
                logger.scalar_summary(tag+'/role loss', role_type_loss, step)
        return loss
    
    def assign_node_score(self, node_graphs, entity_masks, entity_type_scores, trigger_masks, 
                    trigger_type_scores, entity_embs, trigger_embs, predict=False, gold_tri=False, gold_ent=False):
        for (ent_masks, ent_scores, tri_masks, tri_scores, node_graph, entity_emb, trigger_emb) in \
            zip(entity_masks, entity_type_scores, trigger_masks, trigger_type_scores, node_graphs, entity_embs, trigger_embs):
            entity_list = copy.deepcopy(node_graph.entities)
            trigger_list = copy.deepcopy(node_graph.triggers)
            node_graph.clean_entity()
            node_graph.clean_trigger()

            if gold_tri:
                if gold_ent:
                    cnt = 0
                    for (m, ent_emb) in zip(ent_masks, entity_emb):
                        if m:
                            entity = entity_list[cnt]
                            node_graph.add_entity(entity[0], entity[1], entity[2], ent_emb, 100, gold=True)
                            cnt += 1

                    cnt = 0
                    for (m, tri_emb) in zip(tri_masks, trigger_emb):
                        if m:
                            trigger = trigger_list[cnt]
                            node_graph.add_trigger(trigger[0], trigger[1], trigger[2], tri_emb, 100, gold=True)
                            cnt += 1
                else:
                    # get pruning score
                    ent_pred = torch.argmax(ent_scores, dim=1, keepdim=False)
                    ent_score = torch.sum(ent_scores[:, 1:], dim=1, keepdim=False)

                    cnt = 0
                    for (m, ent_s, ent_emb) in zip(ent_masks, ent_score, entity_emb):
                        if m:
                            entity = entity_list[cnt]
                            if predict:
                                node_graph.add_entity(entity[0], entity[1], ent_pred[cnt].tolist(), ent_emb, ent_s, gold=entity[3])
                            else:
                                node_graph.add_entity(entity[0], entity[1], entity[2], ent_emb, ent_s, gold=entity[3])
                            cnt += 1

                    cnt = 0
                    for (m, tri_emb) in zip(tri_masks, trigger_emb):
                        if m:
                            trigger = trigger_list[cnt]
                            node_graph.add_trigger(trigger[0], trigger[1], trigger[2], tri_emb, 100, gold=True)
                            cnt += 1
            else:
                if gold_ent:
                    # get pruning score
                    tri_pred = torch.argmax(tri_scores, dim=1, keepdim=False)
                    tri_score = torch.sum(tri_scores[:, 1:], dim=1, keepdim=False)

                    cnt = 0
                    for (m, ent_emb) in zip(ent_masks, entity_emb):
                        if m:
                            entity = entity_list[cnt]
                            node_graph.add_entity(entity[0], entity[1], entity[2], ent_emb, 100, gold=True)
                            cnt += 1

                    cnt = 0
                    for (m, tri_s, tri_emb) in zip(tri_masks, tri_score, trigger_emb):
                        if m:
                            trigger = trigger_list[cnt]
                            if predict:
                                node_graph.add_trigger(trigger[0], trigger[1], tri_pred[cnt].tolist(), tri_emb, tri_s, gold=trigger[3])
                            else:
                                node_graph.add_trigger(trigger[0], trigger[1], trigger[2], tri_emb, tri_s, gold=trigger[3])
                            cnt += 1
                else:
                    # get pruning score
                    ent_pred = torch.argmax(ent_scores, dim=1, keepdim=False)
                    ent_score = torch.sum(ent_scores[:, 1:], dim=1, keepdim=False)
                    tri_pred = torch.argmax(tri_scores, dim=1, keepdim=False)
                    tri_score = torch.sum(tri_scores[:, 1:], dim=1, keepdim=False)

                    cnt = 0
                    for (m, ent_s, ent_emb) in zip(ent_masks, ent_score, entity_emb):
                        if m:
                            entity = entity_list[cnt]
                            if predict:
                                node_graph.add_entity(entity[0], entity[1], ent_pred[cnt].tolist(), ent_emb, ent_s, gold=entity[3])
                            else:
                                node_graph.add_entity(entity[0], entity[1], entity[2], ent_emb, ent_s, gold=entity[3])
                            cnt += 1

                    cnt = 0
                    for (m, tri_s, tri_emb) in zip(tri_masks, tri_score, trigger_emb):
                        if m:
                            trigger = trigger_list[cnt]
                            if predict:
                                node_graph.add_trigger(trigger[0], trigger[1], tri_pred[cnt].tolist(), tri_emb, tri_s, gold=trigger[3])
                            else:
                                node_graph.add_trigger(trigger[0], trigger[1], trigger[2], tri_emb, tri_s, gold=trigger[3])
                            cnt += 1
        return node_graphs

    def assign_edge_score(self, node_graphs, relation_distribution, role_distribution):
        rel_cnt = 0
        role_cnt = 0
        if relation_distribution is not None:
            relation_scores, relation_preds = torch.max(relation_distribution, dim=1, keepdim=False)
        if role_distribution is not None:
            role_scores, role_preds = torch.max(role_distribution, dim=1, keepdim=False)

        for graph in node_graphs:
            relation_list = graph.relations
            graph.clean_relation()
            for rel in relation_list:
                graph.add_relation(rel[0], rel[1], relation_preds[rel_cnt].tolist(), relation_scores[rel_cnt].tolist(), gold=False)
                rel_cnt += 1
            
            role_list = graph.roles
            graph.clean_role()
            for role in role_list:
                graph.add_role(role[0], role[1], role_preds[role_cnt].tolist(), role_scores[role_cnt].tolist(), gold=False)
                role_cnt += 1
            
            graph.clean()

        return node_graphs

    def span_pruning(self, node_graphs, bert_outputs, token_nums, gold_tri=False, gold_ent=False):
        for token_num, node_graph in zip(token_nums, node_graphs):
            # relation pruning
            num_spans_to_keep = torch.ceil(token_num.float() * self.config.relation_spans_per_word).long()

            entity_scores = bert_outputs.new_tensor(node_graph.entity_scores, 
                                                    dtype=torch.float)

            topk_entity_idx = entity_scores.topk(num_spans_to_keep, dim=0, largest=True, sorted=False)[1].tolist()
            # add relation candidates
            for i_ in topk_entity_idx:
                for j_ in topk_entity_idx:
                    if i_ != j_:
                        subj = node_graph.entities[i_]
                        obj = node_graph.entities[j_]
                        node_graph.add_relation(
                            (subj[0], subj[1], self.entity_type_itos[subj[2]]),
                            (obj[0], obj[1], self.entity_type_itos[obj[2]]),
                            label=0, gold=False
                        )
            # argument pruning
            num_trigger_spans_to_keep = torch.ceil(token_num.float() * self.config.trigger_spans_per_word).long()
            num_argument_spans_to_keep = torch.ceil(token_num.float() * self.config.argument_spans_per_word).long()
            entity_scores = bert_outputs.new_tensor(node_graph.entity_scores, 
                                                    dtype=torch.float)
            if gold_ent:
                topk_entity_idx = entity_scores.topk(len(node_graph.entities), dim=0, largest=True, sorted=False)[1].tolist()
            else:
                topk_entity_idx = entity_scores.topk(num_argument_spans_to_keep, dim=0, largest=True, sorted=False)[1].tolist()
                
            trigger_scores = bert_outputs.new_tensor(node_graph.trigger_scores, 
                                                    dtype=torch.float)
            if gold_tri:
                topk_trigger_idx = trigger_scores.topk(len(node_graph.triggers), dim=0, largest=True, sorted=False)[1].tolist()
            else:
                topk_trigger_idx = trigger_scores.topk(num_trigger_spans_to_keep, dim=0, largest=True, sorted=False)[1].tolist()
            
            
            # add role candidates
            for i_ in topk_trigger_idx:
                for j_ in topk_entity_idx:
                    tri = node_graph.triggers[i_]
                    ent = node_graph.entities[j_]
                    node_graph.add_role(
                        (tri[0], tri[1], self.trigger_type_itos[tri[2]]),
                        (ent[0], ent[1], self.entity_type_itos[ent[2]]),
                        label=0, gold=False
                    )
        return node_graphs

    def predict(self, batch, gold_tri=False, gold_ent=False):
        self.eval()
        with torch.no_grad():
            bert_outputs = self.encode(batch.piece_idxs,
                                       batch.attention_masks,
                                       batch.token_lens)
            batch_size, _, _ = bert_outputs.size()
            
            # span enumeration
            node_graphs = self.span_id(batch, self.vocabs, gold_tri=gold_tri, gold_ent=gold_ent)
            # span classification
            (
                ent_span_emb, entity_masks, entity_type_scores, entity_types,
                tri_span_emb, trigger_masks, trigger_type_scores, trigger_types
            )  = self.span_classification(bert_outputs, node_graphs)
            
            # assign node scores so easier for doing span pruning
            node_graphs = self.assign_node_score(node_graphs, entity_masks, entity_type_scores, trigger_masks, trigger_type_scores, ent_span_emb, tri_span_emb, predict=True, gold_tri=gold_tri, gold_ent=gold_ent)
            
            # span pruning to create graph candidates for relation and role prediction
            node_graphs = self.span_pruning(node_graphs, bert_outputs, batch.token_nums, gold_tri=gold_tri, gold_ent=gold_ent)

            # relation classification
            relation_distribution, _ = self.relation_classification(node_graphs)
            
            # argument classification
            role_distribution, _ = self.argument_classification(node_graphs)
            
            node_graphs = self.assign_edge_score(node_graphs, relation_distribution, role_distribution)

        self.train()
        return node_graphs