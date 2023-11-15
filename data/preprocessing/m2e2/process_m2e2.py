import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

def read_docs(path):
    all_objs = []
    with open(os.path.join(path, "text_only_event.json")) as fp:
        objs = json.load(fp)
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
        if obj["sentence_id"] in train_doc_ids:
            train_objs.append(obj)
        if obj["sentence_id"] in dev_doc_ids:
            dev_objs.append(obj)
        if obj["sentence_id"] in test_doc_ids:
            test_objs.append(obj)
    
    train_objs.sort(key=lambda x: x["sentence_id"])
    dev_objs.sort(key=lambda x: x["sentence_id"])
    test_objs.sort(key=lambda x: x["sentence_id"])
    
    return train_objs, dev_objs, test_objs

def convert_format(objs):
    data = []
    for obj in tqdm(objs, ncols=100):
        
        entity_mentions = {}
        for k, entity in enumerate(obj["golden-entity-mentions"]):
            entity_ = {
                "id": f"{obj['sentence_id']}_Ent{k}", 
                "text": entity["text"], 
                "entity_type": entity["entity-type"], 
                "start": entity["start"], 
                "end": entity["end"], 
            }
            entity_mentions[(entity["start"], entity["end"])] = entity_
            
        event_mentions = []
        for k, event in enumerate(obj["golden-event-mentions"]):
            
            for argument in event["arguments"]:
                argument["entity_id"] = entity_mentions[(argument["start"], argument["end"])]["id"]
            
            event = {
                "id": f"{obj['sentence_id']}_Evt{k}", 
                "event_type": event["event_type"],
                "trigger": event["trigger"],
                "arguments": event["arguments"], 
            }
            event_mentions.append(event)
            
        entity_mentions = sorted(entity_mentions.values(), key=lambda x: (x["start"], x["end"]))
        event_mentions = sorted(event_mentions, key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]))
        for event in event_mentions:
            event["arguments"].sort(key=lambda x: (x["start"], x["end"]))
            
        dt = {
            "doc_id": obj["sentence_id"], 
            "wnd_id": f"{obj['sentence_id']}_1", 
            "text": obj["sentence"], 
            "tokens": obj["words"], 
            "event_mentions": event_mentions, 
            "entity_mentions": entity_mentions, 
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
    parser.add_argument('--seed', type=int, default=0, help="Path to split folder")
    parser.add_argument('--split_path', help="Path to split folder (language level)")
    parser.add_argument('--split', help="Path to split folder")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
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
