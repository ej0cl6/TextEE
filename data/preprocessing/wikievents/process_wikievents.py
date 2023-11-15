import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

def read_docs(path):
    all_objs = []
    with open(os.path.join(path, "train.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    with open(os.path.join(path, "dev.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    with open(os.path.join(path, "test.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    return all_objs

def get_split(objs, split_folder):
    
    with open(os.path.join(split_folder, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_folder, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_folder, "test.txt")) as fp:
        lines = fp.readlines()
        test_doc_ids = set([l.strip() for l in lines])
    
    train_objs = []
    dev_objs = []
    test_objs = []
    
    for obj in objs:
        if obj["doc_id"] in train_doc_ids:
            train_objs.append(obj)
        if obj["doc_id"] in dev_doc_ids:
            dev_objs.append(obj)
        if obj["doc_id"] in test_doc_ids:
            test_objs.append(obj)
    
    train_objs.sort(key=lambda x: x["doc_id"])
    dev_objs.sort(key=lambda x: x["doc_id"])
    test_objs.sort(key=lambda x: x["doc_id"])
    
    return train_objs, dev_objs, test_objs

def convert_format(objs, seg_map):
    
    data = []
    n_arg_drop = 0
    for obj in tqdm(objs, ncols=100):
        
        n_sent = len(obj['sentences'])
        sent_tokens = [[t[0] for t in s[0]] for s in obj['sentences']]
        sent_lens = [len(s[0]) for s in obj['sentences']]
        assert sum(sent_lens) == len(obj['tokens'])
        
        sent_token_start = [0] + [sum(sent_lens[:i+1]) for i in range(n_sent)]
        
        sents = seg_map[obj["doc_id"]]
            
        n_seg = len(sents)
        sent_starts = [sum(sent_lens[0:s[0]]) for s in sents]
        
        idx2seg_map = np.zeros((n_sent, ), dtype=int)
        for i in range(n_sent):
            idx2seg_map[i] = -1
        for i, (idx1, idx2) in enumerate(sents):
            for j in range(idx1, idx2):
                idx2seg_map[j] = i
        
        entity_mentions = defaultdict(list)
        entity_map = {}
        for entity_mention in obj['entity_mentions']:
            seg_loc = idx2seg_map[entity_mention['sent_idx']]
            if seg_loc == -1:
                continue
            entity = {
                "id": entity_mention["id"], 
                "text": entity_mention["text"], 
                "entity_type": entity_mention["entity_type"], 
                "start": entity_mention["start"] - sent_starts[seg_loc], 
                "end": entity_mention["end"] - sent_starts[seg_loc], 
            }
            entity_map[entity_mention["id"]] = (seg_loc, entity)
            entity_mentions[seg_loc].append(entity)
        
        event_mentions = defaultdict(list)
        for event_mention in obj['event_mentions']:
            seg_loc = idx2seg_map[event_mention['trigger']['sent_idx']]
            if seg_loc == -1:
                continue
            
            arguments = []
            for arg in event_mention["arguments"]:
                entity_loc, entity = entity_map[arg["entity_id"]]
                if seg_loc != entity_loc:
                    n_arg_drop += 1
                    continue
                
                argument = {
                    "entity_id": arg["entity_id"], 
                    "role": arg["role"], 
                    "text": arg["text"], 
                    "start": entity["start"], 
                    "end": entity["end"], 
                }
                arguments.append(argument)
            arguments.sort(key=lambda x: (x["start"], x["end"]))
            
            event = {
                "id": event_mention["id"], 
                "event_type": event_mention["event_type"],
                "trigger": {
                    "text": event_mention["trigger"]["text"], 
                    "start": event_mention["trigger"]["start"] - sent_starts[seg_loc], 
                    "end": event_mention["trigger"]["end"] - sent_starts[seg_loc], 
                },
                "arguments": arguments, 
            }
            event_mentions[seg_loc].append(event)
        
        for i in range(n_seg):
            entity_mentions_ = entity_mentions[i]
            event_mentions_ = event_mentions[i]
            
            entity_mentions_ = sorted(entity_mentions_, key=lambda x: (x["start"], x["end"]))
            event_mentions_ = sorted(event_mentions_, key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]))
            dt = {
                "doc_id": obj["doc_id"], 
                "wnd_id": f"{obj['doc_id']}_{i}", 
                "text": " ".join(obj["tokens"][sent_token_start[sents[i][0]]:sent_token_start[sents[i][1]]]), 
                "tokens": obj["tokens"][sent_token_start[sents[i][0]]:sent_token_start[sents[i][1]]], 
                "event_mentions": event_mentions_, 
                "entity_mentions": entity_mentions_, 
                "lang": "en", 
            }
            data.append(dt)
    
    print(f"Number of dropped arguments: {n_arg_drop}")
            
    return data
            
def get_statistics(data):
    event_type_count = defaultdict(int)
    role_type_count = defaultdict(int)
    doc_ids = set()
    max_len = 0
    for dt in data:
        max_len = max(max_len, len(dt["tokens"]))
        doc_ids.add(dt["doc_id"])
        for event in dt["event_mentions"]:
            event_type_count[event["event_type"]] += 1
            for argument in event["arguments"]:
                role_type_count[argument["role"]] += 1
    
    print(f"# of Instances: {len(data)}")
    print(f"# of Docs: {len(doc_ids)}")
    print(f"Max Length: {max_len}")
    print(f"# of Event Types: {len(event_type_count)}")
    print(f"# of Events: {sum(event_type_count.values())}")
    print(f"# of Role Types: {len(role_type_count)}")
    print(f"# of Arguments: {sum(role_type_count.values())}")
    # pprint(event_type_count)
    print()
    
def save_data(out_path, split, train_data, dev_data, test_data):
    data_path = os.path.join(out_path, f"{split}")
    os.makedirs(data_path)
    
    with open(os.path.join(data_path, "train.json"), "w") as fp:
        for data in train_data:
            fp.write(json.dumps(data)+"\n")
    
    with open(os.path.join(data_path, "dev.json"), "w") as fp:
        for data in dev_data:
            fp.write(json.dumps(data)+"\n")
    
    with open(os.path.join(data_path, "test.json"), "w") as fp:
        for data in test_data:
            fp.write(json.dumps(data)+"\n")
    
def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--in_folder", help="Path to the input folder")
    parser.add_argument("-o", "--out_folder", help="Path to the output folder")
    parser.add_argument('--split', help="Path to split folder")
    parser.add_argument('--seg_map', default='seg_map.json')
    args = parser.parse_args()
    
    with open(args.seg_map) as fp:
        seg_map = json.load(fp)
        
    objs = read_docs(args.in_folder)
    
    train_objs, dev_objs, test_objs = get_split(objs, args.split)
    
    train_data = convert_format(train_objs, seg_map)
    dev_data = convert_format(dev_objs, seg_map)
    test_data = convert_format(test_objs, seg_map)
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
