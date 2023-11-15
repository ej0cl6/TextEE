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
    
    with open(os.path.join(path, "valid.jsonl")) as fp:
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
        if obj["id"] in train_doc_ids:
            train_objs.append(obj)
        if obj["id"] in dev_doc_ids:
            dev_objs.append(obj)
        if obj["id"] in test_doc_ids:
            test_objs.append(obj)
    
    train_objs.sort(key=lambda x: x["id"])
    dev_objs.sort(key=lambda x: x["id"])
    test_objs.sort(key=lambda x: x["id"])
    
    return train_objs, dev_objs, test_objs

def convert_format(objs):
    data = []
    for obj in tqdm(objs, ncols=100):
        for sent_id, sent in enumerate(obj["content"]):
            events = []
            for event in obj["events"]:
                event_type = event["type"]
                for mention in event["mention"]:
                    if mention["sent_id"] != sent_id:
                        continue
                    evt = {
                        "id": mention["id"], 
                        "event_type": event_type, 
                        "trigger": {
                            "start": mention["offset"][0], 
                            "end": mention["offset"][1], 
                            "text": mention["trigger_word"], 
                        }, 
                        "arguments": [], 
                    }
                    events.append(evt)
                    events.sort(key=lambda x: x["trigger"]["start"])
                    
            dt = {
                "doc_id": obj["id"], 
                "wnd_id": f"{obj['id']}_{sent_id}", 
                "text": sent["sentence"], 
                "tokens": sent["tokens"], 
                "event_mentions": events, 
                "entity_mentions": [], 
                "lang": "en", 
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
    parser.add_argument('--split', help="Path to split folder")
    args = parser.parse_args()
    
    objs = read_docs(args.in_folder)
    
    train_objs, dev_objs, test_objs = get_split(objs, args.split)
    
    train_data = convert_format(train_objs)
    dev_data = convert_format(dev_objs)
    test_data = convert_format(test_objs)
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
