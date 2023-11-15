import os
import json
import glob
import lxml.etree as et
import numpy as np
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
from tensorboardX import SummaryWriter
import torch

class Logger(object):
    def __init__(self, logdir='./log'):
        self.writer = SummaryWriter(logdir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def text_summary(self, tag, value, step):
        self.writer.add_text(tag, value, step)

def generate_vocabs(datasets):
    """Generate vocabularies from a list of data sets
    :param datasets (list): A list of data sets
    :return (dict): A dictionary of vocabs
    """
    entity_type_set = set()
    event_type_set = set()
    relation_type_set = set()
    role_type_set = set()
    for dataset in datasets:
        entity_type_set.update(dataset.entity_type_set)
        event_type_set.update(dataset.event_type_set)
        relation_type_set.update(dataset.relation_type_set)
        role_type_set.update(dataset.role_type_set)

    # entity and trigger labels
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in sorted(entity_type_set):
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in sorted(event_type_set):
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(sorted(entity_type_set), 1)}
    entity_type_stoi['O'] = 0
    entity_type_itos = {i:k for k, i in entity_type_stoi.items()}

    event_type_stoi = {k: i for i, k in enumerate(sorted(event_type_set), 1)}
    event_type_stoi['O'] = 0
    event_type_itos = {i:k for k, i in event_type_stoi.items()}

    relation_type_stoi = {k: i for i, k in enumerate(sorted(relation_type_set), 1)}
    relation_type_stoi['O'] = 0
    relation_type_itos = {i:k for k, i in relation_type_stoi.items()}

    role_type_stoi = {k: i for i, k in enumerate(sorted(role_type_set), 1)}
    role_type_stoi['O'] = 0
    role_type_itos = {i:k for k, i in role_type_stoi.items()}

    return {
        'entity_type': entity_type_stoi,
        'event_type': event_type_stoi,
        'relation_type': relation_type_stoi,
        'role_type': role_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
        'entity_type_itos': entity_type_itos,
        'event_type_itos': event_type_itos,
        'relation_type_itos': relation_type_itos,
        'role_type_itos': role_type_itos
    }

def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            record = json.loads(line)
            dev = record['dev']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev, test, epoch]

        print('Epoch: {}'.format(best_scores[-1]))
        
        tasks = ['entity', 'trigger', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))

### Below is for span based functions ###
def enumerate_spans(
    sentence,
    offset= 0,
    max_span_width= None,
    min_span_width= 1,
    filter_function= None,
):
    """
    Given a sentence, return all token spans within the sentence. Spans are `exclusive`.
    Additionally, you can provide a maximum and minimum span width, which will be used
    to exclude spans outside of this range.
    Finally, you can provide a function mapping `List[T] -> bool`, which will
    be applied to every span to decide whether that span should be included. This
    allows filtering by length, regex matches, pos tags or any Spacy `Token`
    attributes, for example. TODO (I-Hung): the filter function is not yet supported

    # Parameters
    sentence : `List[T]`, required.
        The sentence to generate spans for. The type is generic, as this function
        can be used with strings, or Spacy `Tokens` or other sequences. In our usage,
        our input is a list of strings.
    offset : `int`, optional (default = `0`)
        A numeric offset to add to all span start and end indices. This is helpful
        if the sentence is part of a larger structure, such as a document, which
        the indices need to respect.
    max_span_width : `int`, optional (default = `None`)
        The maximum length of spans which should be included. Defaults to len(sentence).
    min_span_width : `int`, optional (default = `1`)
        The minimum length of spans which should be included. Defaults to 1.
    filter_function : `Callable[[List[T]], bool]`, optional (default = `None`)
        A function mapping sequences of the passed type T to a boolean value.
        If `True`, the span is included in the returned spans from the
        sentence, otherwise it is excluded..
    """
    max_span_width = max_span_width or len(sentence)
    assert (max_span_width - min_span_width) >= 0
    filter_function = filter_function or (lambda x: True)
    spans = []

    for start_index in range(len(sentence)):
        last_end_index = min(start_index + max_span_width, len(sentence))
        first_end_index = min(start_index + min_span_width - 1, len(sentence))
        for end_index in range(first_end_index, last_end_index):
            start = offset + start_index
            end = offset + end_index + 1
            # TODO (I-Hung): need to add this filter function
            #if filter_function(sentence[slice(start_index, end_index)]):
            #    spans.append([start, end, 0])
            spans.append([start, end, 0]) # default label is 0
    return spans


def graph_add_fake_entity(entities, graph, vocabs, num=None):
    idxs = np.arange(len(entities))
    np.random.shuffle(idxs)
    n_add = 0
    # add fake entity spans to graph
    for idx in idxs:
        entity = entities[idx]
        success = graph.add_entity(start=entity[0], end=entity[1],
                                   label=vocabs['entity_type']['O'],
                                   gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break


def graph_add_fake_trigger(triggers, graph, vocabs, num=None):
    idxs = np.arange(len(triggers))
    np.random.shuffle(idxs)
    n_add = 0
    # add fake trigger spans to graph
    for idx in idxs:
        trigger = triggers[idx]
        success = graph.add_trigger(start=trigger[0], end=trigger[1],
                                    label=vocabs['event_type']['O'],
                                    gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break

          
def graph_add_fake_granular(granulars, graph, vocabs, num=None):
    idxs = np.arange(len(granulars))
    np.random.shuffle(idxs)
    n_add = 0
    # add fake trigger spans to graph
    for idx in idxs:
        granular = granulars[idx]
        success = graph.add_granular(start=granular[0], end=granular[1],
                                    label=vocabs['granular_type']['O'],
                                    gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break


def graph_add_fake_relation(relations, graph, vocabs):
    for relation in relations:
        success = graph.add_relation(ent_1=(relation[0][0], relation[0][1], 'O'), 
                                     ent_2=(relation[1][0], relation[1][1], 'O'),
                                     label=vocabs['relation_type']['O'],
                                     gold=False)


def graph_add_fake_role(roles, graph, vocabs, num=None):
    idxs = np.arange(len(roles))
    np.random.shuffle(idxs)
    n_add = 0
    for idx in idxs:
        role = roles[idx]
        success = graph.add_role(tri=(role[0][0], role[0][1], 'O'), 
                                 ent=(role[1][0], role[1][1], 'O'),
                                 label=vocabs['role_type']['O'],
                                 gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break

def graph_add_fake_ent2ent(relations, graph, vocabs, num=None):
    idxs = np.arange(len(relations))
    np.random.shuffle(idxs)
    n_add = 0
    for idx in idxs:
        relation = relations[idx]
        success = graph.add_ent2ent_relation(obj_1=(relation[0][0], relation[0][1], 'O'), 
                                     obj_2=(relation[1][0], relation[1][1], 'O'),
                                     label=vocabs['ent_ent_relation_type']['O'],
                                     gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break

def graph_add_fake_tri2ent(relations, graph, vocabs, num=None):
    idxs = np.arange(len(relations))
    np.random.shuffle(idxs)
    n_add = 0
    for idx in idxs:
        relation = relations[idx]
        success = graph.add_tri2ent_relation(obj_1=(relation[0][0], relation[0][1], 'O'), 
                                     obj_2=(relation[1][0], relation[1][1], 'O'),
                                     label=vocabs['tri_ent_relation_type']['O'],
                                     gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break

def graph_add_fake_tri2tri(relations, graph, vocabs, num=None):
    idxs = np.arange(len(relations))
    np.random.shuffle(idxs)
    n_add = 0
    for idx in idxs:
        relation = relations[idx]
        success = graph.add_tri2tri_relation(obj_1=(relation[0][0], relation[0][1], 'O'), 
                                     obj_2=(relation[1][0], relation[1][1], 'O'),
                                     label=vocabs['tri_tri_relation_type']['O'],
                                     gold=False)
        n_add += success
        if num is not None and n_add >= num:
            break
