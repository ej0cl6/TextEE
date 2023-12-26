import torch
import numpy as np
from collections import defaultdict

def token_to_berttokens(sent, tokenizer, omit=True, template=False):
    '''
    Generate Bert subtokens, and return first subword index list
    '''
    if template:
        bert_tokens = [tokenizer.tokenize(x) for x in sent]
    else:
        bert_tokens = [tokenizer.tokenize(x, is_split_into_words=True) for x in sent]
    to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens]
    if template:
        second_sep_idx = [i for i in range(len(sent)) if sent[i] == tokenizer.sep_token][-1]
        bert_tokens_prefix = [tokenizer.tokenize(x) for x in sent[:second_sep_idx+1]]
        to_collect = [[1] + [0 for i in range(len(x[1:]))] for x in bert_tokens_prefix]
    bert_tokens = sum(bert_tokens, [])
    to_collect = sum(to_collect, [])
    if omit:
        omit_dic = {tokenizer.sep_token, tokenizer.cls_token}
        to_collect = [x if y not in omit_dic else 0 for x, y in zip(to_collect, bert_tokens)]
    return bert_tokens, to_collect

def arg_to_token_ids(arg_set, tokenizer):
    """
    Generate a dictionary for argument with token ids
    :param arg_set: sorted arg set
    """
    ret = dict()
    for x in arg_set:
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
        ret[x] = ids
    return ret

from keras_preprocessing.sequence import pad_sequences
def event_id_to_arg_query_and_mask(event_arg_dic, args_to_token_ids, args_to_type_ids):
    """
    For each event, compose an argument query and an argument masking
    :return: a dictionary of the following structure: {event_type: (arg_query_bert, arg_query_mask, arg_ids)}
    """
    ret = dict()
    max_w = max([len(v) for v in event_arg_dic.values()])
    queries = [sum([args_to_token_ids[arg] for arg in v], []) for v in event_arg_dic.values()]
    queries = pad_sequences(queries, dtype="long", truncating="post", padding="post")
    max_h = queries.shape[1]

    i = 0
    for event in sorted(list(event_arg_dic.keys())):
        begin, end = 0, 0
        j = 0
        this_query = []
        # extra slot for [SEP] to indicate 'O'
        this_mask = np.zeros((max_h+1, max_w+1))
        arg_ids = []
        for arg in sorted(list(event_arg_dic[event])):
            this_query.append(args_to_token_ids[arg])
            end += len(args_to_token_ids[arg])
            this_mask[begin:end, j] = 1 / (end - begin)
            j += 1
            begin = end
            arg_ids.append(args_to_type_ids[arg])
        arg_ids.append(args_to_type_ids['O'])
        this_query = np.array(sum(this_query, []), dtype=int)
        this_mask[end, j] = 1
        ret[i] = (this_query, this_mask, arg_ids)
        i += 1
    return ret

def from_entity_identifier_to_entity_matrix(entity_identifier, max_entity_count=40):
    # transform the entity token indicator to an entity mapping matrix
    N = len(entity_identifier)
    entity_matrix = torch.zeros((N, max_entity_count))

    for i in range(len(entity_identifier)):
        if entity_identifier[i] < 0 or entity_identifier[i] >= max_entity_count:
            continue
        else:
            this_entity_span = torch.sum(entity_identifier == entity_identifier[i]).float()
            entity_matrix[i, entity_identifier[i]] = 1. / this_entity_span

    return entity_matrix

def pad_seq(data, pad_value=0, dtype='long'):
    N = len(data)
    for i in range(N):
        data[i] = np.array(data[i])
    maxlen = max([len(x) for x in data])
    data = pad_sequences(data, maxlen=maxlen, dtype=dtype, truncating="post", padding="post", value=pad_value)
    return torch.Tensor(data).cuda()

def get_pos(sentence, spacy_tagger):
    '''
    Get POS tag for input sentence
    :param sentence:
    :return:
    '''
    doc = spacy_tagger(' '.join(list(sentence)))
    ret = []
    for token in doc:
        ret.append(token.pos_)
    return ret

def convert_trigger_to_bio(trigger, token_len):
    res = ['O'] * token_len
    st, ed, event_type, name = trigger
    res[st] = 'B-' + event_type
    if ed - 1 > 0 and ed - 1 != st:
        res[ed - 1] = 'I-' + event_type
    return res

def trigger_bio_to_ids(trigger_bio, event_type, sent_len):
    '''
    Convert list annotation to dictionary
    :param trigger_bio: Trigger list [[trigger_for_event_mention1], [trigger_for_event_mention2]]
    :param event_type: event type list
    :param sent_len:
    :return:
    '''
    ret = defaultdict(list)

    if trigger_bio:
        N = len(trigger_bio)
        for i in range(N):
            this_trigger = set(trigger_bio[i])
            if 'O' in this_trigger:
                this_trigger.remove('O')
            if this_trigger:
                this_trigger = list(this_trigger)[0][2:]
            ret[this_trigger].append(trigger_bio[i])

    no_this_trigger = ['O'] * sent_len
    for i in event_type:
        if not ret[i]:
            ret[i] = [no_this_trigger]

    return ret

def prepare_bert_sequence(seq_batch, to_ix, pad, emb_len):
    padded_seqs = []
    for seq in seq_batch:
        pad_seq = torch.full((emb_len,), to_ix(pad), dtype=torch.int)
        # ids = [to_ix(w) for w in seq]
        ids = to_ix(seq)
        pad_seq[:len(ids)] = torch.tensor(ids, dtype=torch.long)
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)

import re
def bio_to_ids(bio_tags, tags_to_ids, remove_overlap=False, is_trigger=False, is_entity=False):
    if remove_overlap:
        bio_tags = [[x.split('#@#')[0] for x in args] for args in bio_tags]
    if is_trigger:
        arg_remove_bio = [[tags_to_ids[x[2:]] if len(x) > 2 else tags_to_ids[x] for x in args] for args in bio_tags]
    elif is_entity:
        arg_remove_bio = [[tags_to_ids[re.split('[: -]', x[2:])[0]] if len(x) > 2 else tags_to_ids[x] for x in args] for args in bio_tags]
    else:
        arg_remove_bio = [[tags_to_ids[re.split('[-#]', x[2:])[0]] if len(x) > 2 and x[2:] in tags_to_ids else tags_to_ids['O'] for x in args] for args in bio_tags]
    args = torch.Tensor(pad_sequences(arg_remove_bio, dtype="long", truncating="post", padding="post", value=tags_to_ids['[PAD]']))
    return args.long()

def prepare_sequence(seq_batch, to_ix, pad, seqlen, remove_bio_prefix=False):
    padded_seqs = []
    for seq in seq_batch:
        if pad == -1:
            pad_seq = torch.full((seqlen,), pad, dtype=torch.int)
        else:
            pad_seq = torch.full((seqlen,), to_ix[pad], dtype=torch.int)
        if remove_bio_prefix:
            ids = [to_ix[w[2:]] if len(w) > 1 and w[2:] in to_ix else to_ix['O'] for w in seq]
        else:
            ids = [to_ix[w] if w in to_ix else -1 for w in seq ]
        
        # pad_seq[:len(ids)] = torch.Tensor(ids).long()
        pad_seq[:len(ids)] = torch.Tensor(ids[:seqlen]).long()
        padded_seqs.append(pad_seq)
    return torch.stack(padded_seqs)