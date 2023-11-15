import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

MAX_TOKEN_NUM = 300

def read_docs(path):
    with open("Few-Shot_ED.json") as fp:
        all_objs = json.load(fp)
    
    n = 0
    for etype in sorted(all_objs.keys()):
        for ex in all_objs[etype]:
            ex.append(f"Doc{n:05d}")
            n += 1
    
    return all_objs

def get_split(objs, split):
    
    with open(os.path.join(split, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split, "test.txt")) as fp:
        lines = fp.readlines()
        test_doc_ids = set([l.strip() for l in lines])

    train_objs = []
    dev_objs = []
    test_objs = []
    
    for etype in objs:
        for obj in objs[etype]:
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

def convert_format(objs, token_map):
    data = defaultdict(list)
    for etype in sorted(objs):
        print(etype)
        n_drop = 0
        for obj in objs[etype]:
            text, trigger, offsets, doc_id = obj[0], obj[1], obj[2], obj[-1]
            if doc_id not in token_map:
                n_drop += 1
                continue
                
            token_offsets, trigger_offset = token_map[doc_id]
            tokens = [text[token_offset[0]:token_offset[1]] for token_offset in token_offsets]
            if len(tokens) > MAX_TOKEN_NUM:
                n_drop += 1
                continue
            
            event = {
                "id": f"{doc_id}_Evt1", 
                "event_type": etype,
                "trigger": {
                    "text": trigger, 
                    "start": trigger_offset[0], 
                    "end": trigger_offset[1], 
                },
                "arguments": [], 
            }
            
            dt = {
                "doc_id": doc_id, 
                "wnd_id": f"{doc_id}_1", 
                "text": text, 
                "tokens": tokens, 
                "event_mentions": [event], 
                "entity_mentions": [], 
                "lang": "en", 
            }
            data[etype].append(dt)
            
        print(etype, n_drop, len(objs[etype]))
    
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
    parser.add_argument('--token_map', default="token_map.json")
    args = parser.parse_args()
    
    objs = read_docs(args.in_folder)
    
    with open(args.token_map) as fp:
        token_map = json.load(fp)
    objs = convert_format(objs, token_map)
    
    train_data, dev_data, test_data = get_split(objs, args.split)
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
