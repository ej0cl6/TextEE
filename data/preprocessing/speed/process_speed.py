import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

def read_file(filename):
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    all_data = [json.loads(l) for l in lines]
    return all_data

def get_split(data, split):
    
    with open(os.path.join(split, "train.txt")) as fp:
        lines = fp.readlines()
        train_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split, "dev.txt")) as fp:
        lines = fp.readlines()
        dev_doc_ids = set([l.strip() for l in lines])
        
    with open(os.path.join(split, "test.txt")) as fp:
        lines = fp.readlines()
        test_doc_ids = set([l.strip() for l in lines])

    train_data = []
    dev_data = []
    test_data = []
    
    for dt in data:
        if dt["doc_id"] in train_doc_ids:
            train_data.append(dt)
        if dt["doc_id"] in dev_doc_ids:
            dev_data.append(dt)
        if dt["doc_id"] in test_doc_ids:
            test_data.append(dt)
    
    return train_data, dev_data, test_data

def convert_format(data):
    new_data = []
    for dt in tqdm(data, ncols=100):     
        new_dt = {
            "doc_id": dt["doc_id"], 
            "wnd_id": dt["wnd_id"], 
            "text": dt["sentence"], 
            "tokens": dt["tokens"], 
            "event_mentions": dt["event_mentions"], 
            "entity_mentions": [], 
            "lang": "en", 
        }
        new_data.append(new_dt)
            
    return new_data
            
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
    
    print(f"# of Data: {len(data)}")
    print(f"# of Docs: {len(doc_ids)}")
    print(f"Max Length: {max_len}")
    print(f"# of Event Types: {len(event_type_count)}")
    print(f"# of Events: {sum(event_type_count.values())}")
    print(f"# of Role Types: {len(role_type_count)}")
    print(f"# of Arguments: {sum(role_type_count.values())}")
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
    parser.add_argument("-i", "--in_file", help="Path to the input file")
    parser.add_argument("-o", "--out_folder", help="Path to the output folder")
    parser.add_argument('--split', help="Path to split folder")
    args = parser.parse_args()
    
    all_data = read_file(args.in_file)
    
    train_data, dev_data, test_data = get_split(all_data, args.split)
    
    train_data = convert_format(train_data)
    dev_data = convert_format(dev_data)
    test_data = convert_format(test_data)
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
