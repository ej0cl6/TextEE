import argparse
import json
from collections import Counter, defaultdict, OrderedDict
import ipdb
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path_dir", type=str, required=True)
parser.add_argument("-o", "--valid_pattern_path", type=str)
args = parser.parse_args()

# read all files ending with '.oneie.json'
data = []
#for ds in ['train.w1.oneie.json', 'dev.w1.oneie.json', 'test.w1.oneie.json']:
for ds in ['train.json', 'val.json', 'test.json']:
    data_path = os.path.join(args.input_path_dir, ds)
    for line in open(data_path, 'r', encoding='utf-8'):
        data.append(json.loads(line))


# get info
total_instance = len(data)
length_info = Counter()
piece_info = Counter()
entity_span_length_counter = Counter()
trigger_span_length_counter = Counter()
entity_per_instance = Counter()
event_per_instance = Counter()
argument_per_instance = Counter()
argument_per_event = Counter()
eventnum_by_type = Counter()
event_per_eventtype = dict()

#valid patterns
all_entity_type = set()
all_roles = set()
event_role_pair = dict()
event_role_counter = dict()
role_enttype_pair = dict()
role_enttype_counter = dict()

for instance in data:
    length_info[len(instance['tokens'])]+=1
    piece_info[len(instance['pieces'])]+=1
    id2entity = dict()
    entity_per_instance[len(instance['entity_mentions'])] += 1
    event_per_instance[len(instance['event_mentions'])] += 1

    for entity in instance['entity_mentions']:
        start, end = entity['start'], entity['end']
        entity_span_length_counter[end-start]+=1
        id2entity[entity['id']] = entity
        all_entity_type.add(entity.get('entity_type', "UNK"))

    total_argument = 0
    event_per_eventtype_ = Counter()
    for event in instance['event_mentions']:
        start, end = event['trigger']['start'], event['trigger']['end']
        trigger_span_length_counter[end-start]+=1
        eventnum_by_type[event['event_type']] += 1
        event_per_eventtype_[event['event_type']] += 1
        argument_per_event[len(event['arguments'])] += 1
        if event['event_type'] not in event_role_pair.keys():
            event_role_pair[event['event_type']] = []
            event_role_counter[event['event_type']] = Counter()
        for argument in event['arguments']:
            total_argument += 1
            entity = id2entity[argument['entity_id']]
            all_roles.add(argument['role'])
            if argument['role'] not in event_role_pair[event['event_type']]:
                event_role_pair[event['event_type']].append(argument['role'])
            event_role_counter[event['event_type']][argument['role']] += 1
            if argument['role'] not in role_enttype_pair.keys():
                role_enttype_pair[argument['role']] = []
                role_enttype_counter[argument['role']] = Counter()
            if entity.get('entity_type', "UNK") not in role_enttype_pair[argument['role']]:
                role_enttype_pair[argument['role']].append(entity.get('entity_type', "UNK"))
            role_enttype_counter[argument['role']][entity.get('entity_type', "UNK")] += 1
        # if event['event_type'] == 'Contact:Phone-Write':
        #     ipdb.set_trace()


    argument_per_instance[total_argument] += 1
    for t, cnt in event_per_eventtype_.items():
        if t not in event_per_eventtype.keys():
            event_per_eventtype[t] = Counter()
        event_per_eventtype[t][cnt] += 1


event_role_pair = OrderedDict(sorted(event_role_pair.items()))
role_enttype_pair = OrderedDict(sorted(role_enttype_pair.items()))
if args.valid_pattern_path is not None:
    event_role_pair_ = dict()
    for k, v in event_role_pair.items():
        event_role_pair_[k] = sorted(v)
    with open(os.path.join(args.valid_pattern_path,'event_role.json'), 'w') as f:
        json.dump(event_role_pair_, f, indent=2)
    
    role_enttype_pair_ = dict()
    for k, v in role_enttype_pair.items():
        role_enttype_pair_[k] = sorted(v)
    with open(os.path.join(args.valid_pattern_path,'role_entity.json'), 'w') as f:
        json.dump(role_enttype_pair_, f, indent=2)