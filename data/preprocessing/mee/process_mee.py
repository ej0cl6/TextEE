import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

LANG_MAP = {
    "english": "en", 
}

def read_docs(path, lang):
    all_objs = []
    with open(os.path.join(path, lang, "train.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    with open(os.path.join(path, lang, "dev.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    with open(os.path.join(path, lang, "test.jsonl")) as fp:
        lines = fp.readlines()
    objs = [json.loads(l) for l in lines]
    all_objs.extend(objs)
    
    for i, obj in enumerate(all_objs):
        obj["doc_id"] = f"Seg{i:05d}"
    
    return all_objs

def get_split(objs, split_path, split_folder):
    
    with open(os.path.join(split_path, split_folder, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_path, split_folder, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split_path, split_folder, "test.txt")) as fp:
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

def convert_format(objs, lang):
    data = []
    for obj in tqdm(objs, ncols=100):
        token_lens = [len(x) for x in obj["tokens"]]
        offset_map_start = [sum(token_lens[:i])+i for i in range(len(token_lens))]
        offset_map_end = [sum(token_lens[:i])+i+token_lens[i] for i in range(len(token_lens))]
        offset_map_start = {v: i for i, v in enumerate(offset_map_start)}
        offset_map_end = {v: i+1 for i, v in enumerate(offset_map_end)}
        
        entity_mentions = {}
        for k in obj["entities"]:
            entity = obj["entities"][k]
            try:
                start = offset_map_start[int(entity["start"])]
                end = offset_map_end[int(entity["end"])]
                assert entity["text"] == " ".join(obj["tokens"][start:end])
            except:
                continue
                
            entity_ = {
                "id": f"{obj['doc_id']}_{k}", 
                "text": entity["text"], 
                "entity_type": entity["type"], 
                "start": start, 
                "end": end, 
            }
            entity_mentions[k] = entity_
        
        
        event_mentions = {}
        for k in obj["triggers"]:
            trigger = obj["triggers"][k]
            
            # fix bug in data, there should not be this type
            if trigger["type"] == "Life_Attack":
                continue
                
            try:
                start = offset_map_start[int(trigger["start"])]
                end = offset_map_end[int(trigger["end"])]
                assert trigger["text"] == " ".join(obj["tokens"][start:end])
            except:
                continue
                
            
            event = {
                "id": f"{obj['doc_id']}_EV_{k}", 
                "event_type": trigger["type"],
                "trigger": {
                    "text": trigger["text"], 
                    "start": start, 
                    "end": end, 
                },
                "arguments": [], 
            }
            event_mentions[k] = event
            
        for argument in obj["arguments"]:
            try:
                entity = entity_mentions[argument["argument"]]
            except:
                continue
            argument_ = {
                "entity_id": entity_mentions[argument["argument"]]["id"], 
                "role": argument["role"], 
                "text": entity["text"],
                "start": entity["start"],
                "end": entity["end"],
            }
            try:
                event_mentions[argument["trigger"]]["arguments"].append(argument_)
            except:
                continue
            
        entity_mentions = sorted(entity_mentions.values(), key=lambda x: (x["start"], x["end"]))
        event_mentions = sorted(event_mentions.values(), key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]))
        for event in event_mentions:
            event["arguments"].sort(key=lambda x: (x["start"], x["end"]))
            
        dt = {
            "doc_id": obj["doc_id"], 
            "wnd_id": f"{obj['doc_id']}_1", 
            "text": " ".join(obj["tokens"]), 
            "tokens": obj["tokens"], 
            "event_mentions": event_mentions, 
            "entity_mentions": entity_mentions, 
            "lang": LANG_MAP[lang], 
        }
        data.append(dt)
            
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
    parser.add_argument('--lang', help="Language")
    parser.add_argument('--split_path', help="Path to split folder (language level)")
    parser.add_argument('--split', help="Path to split folder")
    args = parser.parse_args()
    
    objs = read_docs(args.in_folder, args.lang)
    
    train_objs, dev_objs, test_objs = get_split(objs, args.split_path, args.split)
    
    train_data = convert_format(train_objs, args.lang)
    dev_data = convert_format(dev_objs, args.lang)
    test_data = convert_format(test_objs, args.lang)
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
