import copy, json, logging
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from itertools import combinations
from .graph import Graph
from .util import enumerate_spans, graph_add_fake_entity, graph_add_fake_trigger
import ipdb

logger = logging.getLogger(__name__)

instance_fields = [
    'doc_id', 'wnd_id', 'tokens', 'pieces', 'piece_idxs', 
    'token_lens', 'attention_mask', 'graph', 'trigger_list'
]

batch_fields = [
    'doc_ids', 'wnd_ids', 'tokens', 'pieces', 'piece_idxs', 
    'token_lens', 'attention_masks', 'graphs', 'token_nums',
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


def preprocess_entity(entities):
    """
    We prevent the situation that there are more than 1 types for exactly same span
    """
    span_map = []
    entities_ = []
    delete_num = 0
    for ent in entities:
        if (ent['start'], ent['end']) not in span_map:
            entities_.append(ent)
            span_map.append((ent['start'], ent['end']))
        else:
            delete_num += 1
    if delete_num:
        logger.info('remove {} entities due to span duplication'.format(delete_num))
    return entities_

def get_relation_list(entities, relations, vocab, 
                      directional=True, symmetric=None):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(entities))]
    relation_list = []
    for relation in relations:
        arg_1 = arg_2 = None
        for arg in relation['arguments']:
            if arg['role'] == 'Arg-1':
                arg_1 = entity_idxs[arg['entity_id']]
            elif arg['role'] == 'Arg-2':
                arg_2 = entity_idxs[arg['entity_id']]
        if arg_1 is None or arg_2 is None:
            continue
        relation_type = relation['relation_type']

        if (not directional and arg_1[0] > arg_2[0]) or \
                (directional and symmetric and (relation_type in symmetric) and (arg_1[0] > arg_2[0])):
            arg_1, arg_2 = arg_2, arg_1
        
        if visited[arg_1[0]][arg_2[0]] == 0:
            # TODO (I-Hung): This will automatically remove multi relation
            # scenario, but we first stick to this setup
            temp = ((arg_1[1]['start'], arg_1[1]['end'], arg_1[1].get('entity_type', 'UNK')),
                    (arg_2[1]['start'], arg_2[1]['end'], arg_2[1].get('entity_type', 'UNK')),
                    vocab[relation_type])
            relation_list.append(temp)

            if not directional:
                temp = ((arg_2[1]['start'], arg_2[1]['end'], arg_2.get('entity_type', 'UNK')),
                        (arg_1[1]['start'], arg_1[1]['end'], arg_1.get('entity_type', 'UNK')),
                        vocab[relation_type])
                relation_list.append(temp)
                visited[arg_2[0]][arg_1[0]] = 1

            visited[arg_1[0]][arg_2[0]] = 1

    relation_list.sort(key=lambda x: (x[0][0], x[1][0]))
    return relation_list

def get_role_list(entities, events, vocab):
    entity_idxs = {entity['id']: (i,entity) for i, entity in enumerate(entities)}
    visited = [[0] * len(entities) for _ in range(len(events))]
    role_list = []
    cnt = 0
    for i, event in enumerate(events):
        for arg in event['arguments']:
            entity_idx = entity_idxs[arg['entity_id']]
            if visited[i][entity_idx[0]] == 0 and arg['role'] in vocab:
                # TODO (I-Hung): This will automatically remove multi role
                # scenario, but we first stick to this setup
                temp = ((event['trigger']['start'], event['trigger']['end'], event['event_type']),
                        (entity_idx[1]['start'], entity_idx[1]['end'], entity_idx[1].get('entity_type', 'UNK')),
                        vocab[arg['role']])
                role_list.append(temp)
                visited[i][entity_idx[0]] = 1
            else:
                cnt += 1 
    role_list.sort(key=lambda x: (x[0][0], x[1][0]))
    if cnt:
        logger.info('{} times of role are removed in gold because of span duplication'.format(cnt))
    return role_list

def clean_events(events):
    cleaned_map = {}
    for event in events:
        key = (event['trigger']['start'], event['trigger']['end'], event['event_type'], event['trigger']['text'])
        if key in cleaned_map:
            # do argument merging
            cleaned_map[key]['arguments'].extend(event['arguments'])
        else:
            cleaned_map[key] = event
    return list(cleaned_map.values())


class IEDataset(Dataset):
    def __init__(self, raw_data, tokenizer, config, max_length=128, test=False):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length
        self.test=test
        
        self.max_entity_span = config.max_entity_span
        self.min_entity_span = config.min_entity_span
        self.max_trigger_span = config.max_trigger_span
        self.min_trigger_span = config.min_trigger_span

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def entity_type_set(self):
        type_set = set()
        for inst in self.data:
            for entity in inst['entity_mentions']:
                type_set.add(entity.get('entity_type', "UNK"))
        return type_set

    @property
    def event_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                type_set.add(event['event_type'])
        return type_set

    @property
    def relation_type_set(self):
        type_set = set()
        for inst in self.data:
            for relation in inst.get('relation_mentions', []):
                type_set.add(relation['relation_type'])
        return type_set

    @property
    def role_type_set(self):
        type_set = set()
        for inst in self.data:
            for event in inst['event_mentions']:
                for arg in event['arguments']:
                    type_set.add(arg['role'])
        return type_set

    def load_data(self):
        overlength_num = 0
        for inst in self.raw_data:
            
            ## added
            pieces = [self.tokenizer.tokenize(t, is_split_into_words=True) for t in inst['tokens']]
            token_lens = [len(x) for x in pieces]
            if 0 in token_lens:
                raise ValueError
            pieces = [p for ps in pieces for p in ps]
            inst['pieces'] = pieces
            inst['token_lens'] = token_lens
            
            inst['entity_mentions'] = inst['extra_info']['entity_mentions']
            inst['relation_mentions'] = inst['extra_info']['relation_mentions']
            inst['event_mentions'] = inst['extra_info']['event_mentions']
            ##

            if not self.test:
                if self.max_length != -1 and len(pieces) > self.max_length - 2:
                    overlength_num += 1
                    continue
            else:
                if len(pieces) > self.max_length - 2:
                    # add token_lens until over-length
                    piece_counter = 0
                    for max_token_include, token_len in enumerate(inst['token_lens']):
                        if piece_counter + token_len >= self.max_length - 2:
                            logger.info('overlength during testing...')
                            break
                        else:
                            piece_counter += token_len
                    inst['pieces'] = inst['pieces'][:piece_counter]
                    inst['token_lens'] = inst['token_lens'][:max_token_include]
                    inst['tokens'] = inst['tokens'][:max_token_include]
            self.data.append(inst)

        if overlength_num:
            logger.info('Discarded {} overlength instances'.format(overlength_num))
        logger.info('Loaded {} DyGIEpp instances from {} E2E instances'.format(len(self), len(self.raw_data)))

    def numberize(self, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        entity_type_stoi = vocabs.get('entity_type', None)
        event_type_stoi = vocabs.get('event_type', None)
        relation_type_stoi = vocabs.get('relation_type', None)
        role_type_stoi = vocabs.get('role_type', None)

        data = []
        for inst in self.data:
            doc_id = inst['doc_id']
            tokens = inst['tokens']
            pieces = inst['pieces']
            wnd_id = inst['wnd_id']
            token_num = len(tokens)
            token_lens = inst['token_lens']

            entities = inst['entity_mentions']
            entities.sort(key=lambda x: x['start'])
            events = inst['event_mentions']
            # events = clean_events(events)
            events.sort(key=lambda x: x['trigger']['start'])

            # Pad word pieces with special tokens
            piece_idxs = self.tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            if sum(token_lens) < self.max_length -2:
                assert sum(token_lens) +2 == len(piece_idxs)
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
            piece_idxs = piece_idxs + [pad_id] * pad_num

            entity_list = [(e['start'], e['end'], 
                            entity_type_stoi[e.get('entity_type', "UNK")])
                           for e in entities]

            trigger_list = [(e['trigger']['start'], e['trigger']['end'],
                             event_type_stoi[e['event_type']])
                            for e in events]

            # Argument role
            role_list = get_role_list(entities, events, role_type_stoi)

            # Relations
            relation_list = get_relation_list(entities, inst.get('relation_mentions', []),relation_type_stoi)

            # Graph
            graph = Graph(
                entities=entity_list,
                triggers=trigger_list,
                relations=relation_list,
                roles=role_list,
                vocabs=vocabs,
                gold=True
            )

            # Add other span from span enumeration
            entity_spans = enumerate_spans(tokens, offset=0,
                                        max_span_width=self.max_entity_span, 
                                        min_span_width=self.min_entity_span)
            trigger_spans = enumerate_spans(tokens, offset=0,
                                        max_span_width=self.max_trigger_span, 
                                        min_span_width=self.min_trigger_span)
            
            graph_add_fake_entity(entity_spans, graph, vocabs)
            graph_add_fake_trigger(trigger_spans, graph, vocabs)
            
            instance = Instance(
                doc_id=doc_id,
                wnd_id=wnd_id,
                tokens=tokens,
                pieces=pieces,
                piece_idxs=piece_idxs,
                token_lens=token_lens,
                attention_mask=attn_mask,
                graph=graph,
                trigger_list=trigger_list
            )
            data.append(instance)
        self.data = data
        
    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_graphs = []
        batch_token_lens = []
        batch_attention_masks = []
        doc_ids = [inst.doc_id for inst in batch]
        wnd_ids = [inst.wnd_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_graphs.append(inst.graph)
            batch_tokens.append(inst.tokens)
            
        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)
        token_nums = torch.cuda.LongTensor(token_nums)
    
        return Batch(
            doc_ids=doc_ids,
            wnd_ids=wnd_ids,
            tokens=[inst.tokens for inst in batch],
            pieces=[inst.pieces for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            graphs=batch_graphs,
            token_nums=token_nums,        
        )
