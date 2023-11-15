import os
import json
import glob
import lxml.etree as et
from nltk import word_tokenize, sent_tokenize
from copy import deepcopy
import numpy as np


def generate_vocabs(datasets, coref=False,
                    relation_directional=False,
                    symmetric_relations=None):
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

    # add inverse relation types for non-symmetric relations
    if relation_directional:
        if symmetric_relations is None:
            symmetric_relations = []
        relation_type_set_ = set()
        for relation_type in relation_type_set:
            relation_type_set_.add(relation_type)
            if relation_directional and relation_type not in symmetric_relations:
                relation_type_set_.add(relation_type + '_inv')

    # entity and trigger labels
    prefix = ['B', 'I']
    entity_label_stoi = {'O': 0}
    trigger_label_stoi = {'O': 0}
    for t in entity_type_set:
        for p in prefix:
            entity_label_stoi['{}-{}'.format(p, t)] = len(entity_label_stoi)
    for t in event_type_set:
        for p in prefix:
            trigger_label_stoi['{}-{}'.format(p, t)] = len(trigger_label_stoi)

    entity_type_stoi = {k: i for i, k in enumerate(entity_type_set, 1)}
    entity_type_stoi['O'] = 0

    event_type_stoi = {k: i for i, k in enumerate(event_type_set, 1)}
    event_type_stoi['O'] = 0

    relation_type_stoi = {k: i for i, k in enumerate(relation_type_set, 1)}
    relation_type_stoi['O'] = 0
    if coref:
        relation_type_stoi['COREF'] = len(relation_type_stoi)

    role_type_stoi = {k: i for i, k in enumerate(role_type_set, 1)}
    role_type_stoi['O'] = 0

    mention_type_stoi = {'NAM': 0, 'NOM': 1, 'PRO': 2, 'UNK': 3, 'NEU': 4}

    return {
        'entity_type': entity_type_stoi,
        'event_type': event_type_stoi,
        'relation_type': relation_type_stoi,
        'role_type': role_type_stoi,
        'mention_type': mention_type_stoi,
        'entity_label': entity_label_stoi,
        'trigger_label': trigger_label_stoi,
    }


def load_valid_patterns(path, vocabs):
    event_type_vocab = vocabs['event_type']
    entity_type_vocab = vocabs['entity_type']
    relation_type_vocab = vocabs['relation_type']
    role_type_vocab = vocabs['role_type']

    # valid event-role
    valid_event_role = set()
    event_role = json.load(
        open(os.path.join(path, 'event_role.json'), 'r', encoding='utf-8'))
    for event, roles in event_role.items():
        if event not in event_type_vocab:
            continue
        event_type_idx = event_type_vocab[event]
        for role in roles:
            if role not in role_type_vocab:
                continue
            role_type_idx = role_type_vocab[role]
            valid_event_role.add(event_type_idx * 100 + role_type_idx)

    # valid relation-entity
    valid_relation_entity = set()
    relation_entity = json.load(
        open(os.path.join(path, 'relation_entity.json'), 'r', encoding='utf-8'))
    for relation, entities in relation_entity.items():
        relation_type_idx = relation_type_vocab[relation]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_relation_entity.add(
                relation_type_idx * 100 + entity_type_idx)

    # valid role-entity
    valid_role_entity = set()
    role_entity = json.load(
        open(os.path.join(path, 'role_entity.json'), 'r', encoding='utf-8'))
    for role, entities in role_entity.items():
        if role not in role_type_vocab:
            continue
        role_type_idx = role_type_vocab[role]
        for entity in entities:
            entity_type_idx = entity_type_vocab[entity]
            valid_role_entity.add(role_type_idx * 100 + entity_type_idx)

    return {
        'event_role': valid_event_role,
        'relation_entity': valid_relation_entity,
        'role_entity': valid_role_entity
    }


def read_ltf(path):
    root = et.parse(path, et.XMLParser(
        dtd_validation=False, encoding='utf-8')).getroot()
    doc_id = root.find('DOC').get('id')
    doc_tokens = []
    for seg in root.find('DOC').find('TEXT').findall('SEG'):
        seg_id = seg.get('id')
        seg_tokens = []
        seg_start = int(seg.get('start_char'))
        seg_text = seg.find('ORIGINAL_TEXT').text
        for token in seg.findall('TOKEN'):
            token_text = token.text
            start_char = int(token.get('start_char'))
            end_char = int(token.get('end_char'))
            assert seg_text[start_char - seg_start:
                            end_char - seg_start + 1
                            ] == token_text, 'token offset error'
            seg_tokens.append((token_text, start_char, end_char))
        doc_tokens.append((seg_id, seg_tokens))

    return doc_tokens, doc_id


def read_txt(path, language='english'):
    doc_id = os.path.basename(path)
    data = open(path, 'r', encoding='utf-8').read()
    data = [s.strip() for s in data.split('\n') if s.strip()]
    sents = [l for ls in [sent_tokenize(line, language=language) for line in data]
             for l in ls]
    doc_tokens = []
    offset = 0
    for sent_idx, sent in enumerate(sents):
        sent_id = '{}-{}'.format(doc_id, sent_idx)
        tokens = word_tokenize(sent)
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((sent_id, tokens))
    return doc_tokens, doc_id


def read_json(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = data[0]['doc_id']
    offset = 0
    doc_tokens = []

    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, offset + i, offset + i + 1)
                  for i, token in enumerate(tokens)]
        offset += len(tokens)
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def read_json_single(path):
    with open(path, 'r', encoding='utf-8') as r:
        data = [json.loads(line) for line in r]
    doc_id = os.path.basename(path)
    doc_tokens = []
    for inst in data:
        tokens = inst['tokens']
        tokens = [(token, i, i + 1) for i, token in enumerate(tokens)]
        doc_tokens.append((inst['sent_id'], tokens))
    return doc_tokens, doc_id


def save_result(output_file, gold_graphs, pred_graphs, sent_ids, tokens=None):
    with open(output_file, 'w', encoding='utf-8') as w:
        for i, (gold_graph, pred_graph, sent_id) in enumerate(
                zip(gold_graphs, pred_graphs, sent_ids)):
            output = {'sent_id': sent_id,
                      'gold': gold_graph.to_dict(),
                      'pred': pred_graph.to_dict()}
            if tokens:
                output['tokens'] = tokens[i]
            w.write(json.dumps(output) + '\n')


def mention_to_tab(start, end, entity_type, mention_type, mention_id, tokens, token_ids, score=1):
    tokens = tokens[start:end]
    token_ids = token_ids[start:end]
    span = '{}:{}-{}'.format(token_ids[0].split(':')[0],
                             token_ids[0].split(':')[1].split('-')[0],
                             token_ids[1].split(':')[1].split('-')[1])
    mention_text = tokens[0]
    previous_end = int(token_ids[0].split(':')[1].split('-')[1])
    for token, token_id in zip(tokens[1:], token_ids[1:]):
        start, end = token_id.split(':')[1].split('-')
        start, end = int(start), int(end)
        mention_text += ' ' * (start - previous_end) + token
        previous_end = end
    return '\t'.join([
        'json2tab',
        mention_id,
        mention_text,
        span,
        'NIL',
        entity_type,
        mention_type,
        str(score)
    ])


def json_to_mention_results(input_dir, output_dir, file_name,
                            bio_separator=' '):
    mention_type_list = ['nam', 'nom', 'pro', 'nam+nom+pro']
    file_type_list = ['bio', 'tab']
    writers = {}
    for mention_type in mention_type_list:
        for file_type in file_type_list:
            output_file = os.path.join(output_dir, '{}.{}.{}'.format(file_name,
                                                                     mention_type,
                                                                     file_type))
            writers['{}_{}'.format(mention_type, file_type)
                    ] = open(output_file, 'w')

    json_files = glob.glob(os.path.join(input_dir, '*.json'))
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as r:
            for line in r:
                result = json.loads(line)
                doc_id = result['doc_id']
                tokens = result['tokens']
                token_ids = result['token_ids']
                bio_tokens = [[t, tid, 'O']
                              for t, tid in zip(tokens, token_ids)]
                # separate bio output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    tokens_tmp = deepcopy(bio_tokens)
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            tokens_tmp[start] = 'B-{}'.format(enttype)
                            for token_idx in range(start + 1, end):
                                tokens_tmp[token_idx] = 'I-{}'.format(
                                    enttype)
                    writer = writers['{}_bio'.format(mention_type.lower())]
                    for token in tokens_tmp:
                        writer.write(bio_separator.join(token) + '\n')
                    writer.write('\n')
                # combined bio output
                tokens_tmp = deepcopy(bio_tokens)
                for start, end, enttype, _ in result['graph']['entities']:
                    tokens_tmp[start] = 'B-{}'.format(enttype)
                    for token_idx in range(start + 1, end):
                        tokens_tmp[token_idx] = 'I-{}'.format(enttype)
                writer = writers['nam+nom+pro_bio']
                for token in tokens_tmp:
                    writer.write(bio_separator.join(token) + '\n')
                writer.write('\n')
                # separate tab output
                for mention_type in ['NAM', 'NOM', 'PRO']:
                    writer = writers['{}_tab'.format(mention_type.lower())]
                    mention_count = 0
                    for start, end, enttype, mentype in result['graph']['entities']:
                        if mention_type == mentype:
                            mention_id = '{}-{}'.format(doc_id, mention_count)
                            tab_line = mention_to_tab(
                                start, end, enttype, mentype, mention_id, tokens, token_ids)
                            writer.write(tab_line + '\n')
                # combined tab output
                writer = writers['nam+nom+pro_tab']
                mention_count = 0
                for start, end, enttype, mentype in result['graph']['entities']:
                    mention_id = '{}-{}'.format(doc_id, mention_count)
                    tab_line = mention_to_tab(
                        start, end, enttype, mentype, mention_id, tokens, token_ids)
                    writer.write(tab_line + '\n')
    for w in writers:
        w.close()


def normalize_score(scores):
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        return [0] * len(scores)
    return [(s - min_score) / (max_score - min_score) for s in scores]


def best_score_by_task(log_file, task, max_epoch=1000):
    with open(log_file, 'r', encoding='utf-8') as r:
        config = r.readline()

        best_scores = []
        best_dev_score = 0
        for line in r:
            # print(line)
            record = json.loads(line)
            dev = record['dev']
            dev_goldTri = record['dev_goldTri']
            test = record['test']
            epoch = record['epoch']
            if epoch > max_epoch:
                break
            if dev[task]['f'] > best_dev_score:
                best_dev_score = dev[task]['f']
                best_scores = [dev_goldTri, test, epoch]
        print(best_scores)
        print('Epoch: {}'.format(best_scores[-1]))
        tasks = ['entity', 'mention', 'relation', 'trigger_id', 'trigger',
                 'role_id', 'role']
        for t in tasks:
            print('{}: dev: {:.2f}, test: {:.2f}'.format(t,
                                                         best_scores[0][t][
                                                             'f'] * 100.0,
                                                         best_scores[1][t][
                                                             'f'] * 100.0))

def print_stats(tokens, gold_graphs, pred_graphs):
    vocab = gold_graphs[0].vocabs

    ent_dict = vocab['entity_type']
    evt_dict = vocab['event_type']
    rela_dict = vocab['relation_type']
    role_dict = vocab['role_type']

    print(ent_dict)
    print(evt_dict)
    print(rela_dict)
    print(role_dict)

    ent_vocab = {v: k for k, v in vocab['entity_type'].items()}
    evt_vocab = {v: k for k, v in vocab['event_type'].items()}
    rela_vocab = {v: k for k, v in vocab['relation_type'].items()}
    role_vocab = {v: k for k, v in vocab['role_type'].items()}

    ent_num = len(ent_vocab)
    evt_num = len(evt_vocab)
    rela_num = len(rela_vocab)
    role_num= len(role_vocab)

    ent_mat = np.zeros((ent_num, ent_num), dtype=int)
    evt_mat = np.zeros((evt_num, evt_num), dtype=int)
    rela_mat = np.zeros((rela_num, rela_num), dtype=int)
    role_mat = np.zeros((role_num, role_num), dtype=int)

    for i in range(len(gold_graphs)):
        gold_graph = gold_graphs[i]
        pred_graph = pred_graphs[i]

        # entities
        gold_entites = gold_graph.entities
        pred_entites = pred_graph.entities

        gold_offsets, pred_offsets = [], []
        gold_class, pred_class = [], []

        for gold_entity in gold_entites:
            gold_offsets.append((gold_entity[0], gold_entity[1]))
            gold_class.append(gold_entity[2])
        
        for pred_entity in pred_entites:
            pred_offsets.append((pred_entity[0], pred_entity[1]))
            pred_class.append(pred_entity[2])
        
        for j in range(len(gold_class)):
            gold_ent_j = gold_offsets[j]
            if gold_ent_j not in pred_offsets:
                ent_mat[gold_class[j]][0] += 1
            else:
                pred_index = pred_offsets.index(gold_ent_j)
                ent_mat[gold_class[j]][pred_class[pred_index]] += 1
        
        # triggers
        gold_events = gold_graph.triggers
        pred_events = pred_graph.triggers

        gold_offsets, pred_offsets = [], []
        gold_class, pred_class = [], []

        for gold_event in gold_events:
            gold_offsets.append((gold_event[0], gold_event[1]))
            gold_class.append(gold_event[2])
        
        for pred_event in pred_events:
            pred_offsets.append((pred_event[0], pred_event[1]))
            pred_class.append(pred_event[2])


        for j in range(len(gold_class)):
            gold_evt_j = gold_offsets[j]
            if gold_evt_j not in pred_offsets:
                evt_mat[gold_class[j]][0] += 1
            else:
                pred_index = pred_offsets.index(gold_evt_j)
                evt_mat[gold_class[j]][pred_class[pred_index]] += 1

    print(ent_mat)
    print(evt_mat)


def write_analysis_result(batch_tokens, batch_gold_graphs, batch_pred_graphs, output_dir):
    vocab = batch_gold_graphs[0].vocabs

    ent_vocab = {v: k for k, v in vocab['entity_type'].items()}
    evt_vocab = {v: k for k, v in vocab['event_type'].items()}
    rela_vocab = {v: k for k, v in vocab['relation_type'].items()}
    role_vocab = {v: k for k, v in vocab['role_type'].items()}
    cnt = 0

    with open(output_dir, "w", encoding='utf-8') as f:
        for i in range(len(batch_gold_graphs)):
            tokens_i = batch_tokens[i]

            gold_graph_i = batch_gold_graphs[i]
            pred_graph_i = batch_pred_graphs[i]

            gold_entity_span_list = []
            pred_entity_span_list = []
            gold_trigger_span_list = []
            pred_trigger_span_list = []

            # Entities
            gold_ent_str = "Gold Entities: | "
            for gold_entity in gold_graph_i.entities:
                span_words = tokens_i[gold_entity[0]: gold_entity[1]]
                gold_entity_span_list.append(" ".join(span_words.copy()))
                gold_ent_str += (" ".join(span_words))
                gold_ent_str += '---'
                gold_ent_str += ent_vocab[gold_entity[-1]]
                gold_ent_str += " | "
            gold_ent_str += "\n"

            pred_ent_str = "Pred Entities: | "
            for pred_entity in pred_graph_i.entities:
                span_words = tokens_i[pred_entity[0]: pred_entity[1]]
                pred_entity_span_list.append(" ".join(span_words.copy()))
                pred_ent_str += (" ".join(span_words))
                pred_ent_str += '---'
                pred_ent_str += ent_vocab[pred_entity[-1]]
                pred_ent_str += " | "
            pred_ent_str += "\n"

            # Triggers
            gold_trig_str = "Gold Triggers: | "
            for gold_entity in gold_graph_i.triggers:
                span_words = tokens_i[gold_entity[0]: gold_entity[1]]
                gold_trigger_span_list.append(" ".join(span_words.copy()))
                gold_trig_str += (" ".join(span_words))
                gold_trig_str += '---'
                gold_trig_str += evt_vocab[gold_entity[-1]]
                gold_trig_str += " | "
            gold_trig_str += "\n"

            pred_trig_str = "Pred Triggers: | "
            for pred_entity in pred_graph_i.triggers:
                span_words = tokens_i[pred_entity[0]: pred_entity[1]]
                pred_trigger_span_list.append(" ".join(span_words.copy()))
                pred_trig_str += (" ".join(span_words))
                pred_trig_str += '---'
                pred_trig_str += evt_vocab[pred_entity[-1]]
                pred_trig_str += " | "
            pred_trig_str += "\n"

            trig_gold_str = gold_trig_str
            trig_pred_str = pred_trig_str

            # Roles
            gold_role_str_list = []
            pred_role_str_list = []

            for gold_role in gold_graph_i.roles:
                gold_str = ""
                gold_str += gold_trigger_span_list[gold_role[0]]
                gold_str += " --> "
                gold_str += gold_entity_span_list[gold_role[1]]
                gold_str += " : "
                gold_str += role_vocab[gold_role[2]]
                gold_str += "\n"
                gold_role_str_list.append(gold_str)
            for pred_role in pred_graph_i.roles:
                pred_str = ""
                pred_str += pred_trigger_span_list[pred_role[0]]
                pred_str += " --> "
                pred_str += pred_entity_span_list[pred_role[1]]
                pred_str += " : "
                pred_str += role_vocab[pred_role[2]]
                pred_str += "\n"
                pred_role_str_list.append(pred_str)
                
            
            # Relations
            gold_rela_str_list = []
            pred_rela_str_list = []

            for gold_rela in gold_graph_i.relations:
                gold_rela_str = ""
                gold_rela_str += gold_entity_span_list[gold_rela[0]]
                gold_rela_str += " --> "
                gold_rela_str += gold_entity_span_list[gold_rela[1]]
                gold_rela_str += " : "
                gold_rela_str += rela_vocab[gold_rela[2]]
                gold_rela_str += "\n"

                gold_rela_str_list.append(gold_rela_str)
            
            for pred_rela in pred_graph_i.relations:
                pred_rela_str = ""
                pred_rela_str += pred_entity_span_list[pred_rela[0]]
                pred_rela_str += " --> "
                pred_rela_str += pred_entity_span_list[pred_rela[1]]
                pred_rela_str += " : "
                pred_rela_str += rela_vocab[pred_rela[2]]
                pred_rela_str += "\n"

                pred_rela_str_list.append(pred_rela_str)
            
            if not (set(gold_graph_i.entities) == set(pred_graph_i.entities) and set(gold_graph_i.triggers) == set(pred_graph_i.triggers) and set(pred_role_str_list) == set(gold_role_str_list) and set(pred_rela_str_list) == set(gold_rela_str_list)):
                f.write('\n')
                f.write('\n')
                f.write("=============================================================" + '\n')
                sent_str = "# SENTENCE #: "
                for token in tokens_i:
                    sent_str = sent_str + token + ' '
                f.write(sent_str + '\n')
                f.write("=============================================================" + '\n')

            if set(gold_graph_i.entities) != set(pred_graph_i.entities):
                f.write(gold_ent_str)
                f.write(pred_ent_str)
                f.write("=============================================================" + '\n')

            if set(gold_graph_i.triggers) != set(pred_graph_i.triggers):
                f.write(trig_gold_str)
                f.write(trig_pred_str)
                f.write("=============================================================" + '\n')

            if set(pred_role_str_list) != set(gold_role_str_list):
                f.write("++++ Gold Arguments ++++" + '\n')
                for gold in gold_role_str_list:
                    f.write(gold)
                f.write("-------------------------------------------------------------" + '\n')
                f.write("++++ Predicted Arguments ++++" + '\n')
                for pred in pred_role_str_list:
                    f.write(pred)
                f.write("=============================================================" + '\n')

            if set(pred_rela_str_list) != set(gold_rela_str_list):
                f.write("++++ Gold Relations ++++" + '\n')
                for gold in gold_rela_str_list:
                    f.write(gold)
                f.write("-------------------------------------------------------------" + '\n')
                f.write("++++ Predicted Relations ++++" + '\n')
                for pred in pred_rela_str_list:
                    f.write(pred)
                f.write("=============================================================" + '\n')
            

if __name__ == "__main__":
    best_score_by_task('log/20201108_013209/log.txt', 'trigger_id')
