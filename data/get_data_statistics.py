import os, json
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from pprint import pprint
import ipdb
            
def get_statistics(fn):
    
    with open(fn) as fp:
        lines = fp.readlines()
    data = [json.loads(l) for l in lines]
    
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
    
    print(f"# of Docs: {len(doc_ids)}")
    print(f"# of Instances: {len(data)}")
    print(f"Max Length: {max_len}")
    print(f"# of Event Types: {len(event_type_count)}")
    print(f"# of Events: {sum(event_type_count.values())}")
    print(f"# of Role Types: {len(role_type_count)}")
    print(f"# of Arguments: {sum(role_type_count.values())}")
    
def main():
    parser = ArgumentParser()
    parser.add_argument("--data")
    args = parser.parse_args()
    
    get_statistics(args.data)
    
        
if __name__ == "__main__":
    main()
