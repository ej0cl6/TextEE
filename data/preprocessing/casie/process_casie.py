import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

def read_docs(path):
    all_objs = []
    
    for file in os.listdir(path):
        if not file.endswith("json"):
            continue
        
        try:
            with open(os.path.join(path, file)) as fp:
                obj = json.load(fp)
        except:
            continue
        all_objs.append(obj)
    
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
    
    for obj in objs:
        if obj["sourcefile"] in train_doc_ids:
            train_objs.append(obj)
        if obj["sourcefile"] in dev_doc_ids:
            dev_objs.append(obj)
        if obj["sourcefile"] in test_doc_ids:
            test_objs.append(obj)
    
    train_objs.sort(key=lambda x: x["sourcefile"])
    dev_objs.sort(key=lambda x: x["sourcefile"])
    test_objs.sort(key=lambda x: x["sourcefile"])
    
    return train_objs, dev_objs, test_objs

def convert_format(objs, token_map, seg_map):
    
    data = []
    n_drop_event = 0
    n_drop_arg = 0
    for obj in tqdm(objs, ncols=100):
        
        text_tokens = [[obj["content"][si:ei] for si, ei in sent] for sent in token_map[obj["sourcefile"]]]
        n_sent = len(text_tokens)
        sent_lens = [len(s) for s in text_tokens]
        
        sents = seg_map[obj["sourcefile"]]
        n_seg = len(sents)
        
        entity_mentions = defaultdict(list)
        event_mentions = defaultdict(list)
        sent_offsets = token_map[obj["sourcefile"]]
        seg_offsets = []
        for si, ei in sents:
            segs = []
            for i in range(si, ei):
                segs.extend(sent_offsets[i])
            seg_offsets.append(segs)
                
        for hopper in obj['cyberevent']['hopper']:
            for event in hopper["events"]:
                trigger = event['nugget']
                tri_seg_id = -1
                for i, sent_offsets in enumerate(seg_offsets):
                    if sent_offsets[0][0] <= trigger['startOffset'] and trigger['endOffset'] <= sent_offsets[-1][1]:
                        tri_seg_id = i
                        break
                if tri_seg_id == -1:
                    n_drop_event += 1
                    continue
                    
                tri_start_id = -1
                tri_end_id = -1
                for i, token_offsets in enumerate(seg_offsets[tri_seg_id]):
                    if token_offsets[0] == trigger["startOffset"]:
                        tri_start_id = i
                        for j in range(i, len(seg_offsets[tri_seg_id])):
                            if seg_offsets[tri_seg_id][j][1] == trigger["endOffset"]:
                                tri_end_id = j + 1
                        break
                
                if tri_start_id == -1 or tri_end_id == -1:
                    if obj['content'][trigger["startOffset"]:trigger["endOffset"]] != trigger["text"] and obj['content'][trigger["startOffset"]-1:trigger["endOffset"]-1] == trigger["text"]:
                        tri_start_id = -1
                        tri_end_id = -1
                        for i, token_offsets in enumerate(seg_offsets[tri_seg_id]):
                            if token_offsets[0] == trigger["startOffset"]-1:
                                tri_start_id = i
                                for j in range(i, len(seg_offsets[tri_seg_id])):
                                    if seg_offsets[tri_seg_id][j][1] == trigger["endOffset"]-1:
                                        tri_end_id = j + 1
                                break
                        
                        if tri_start_id == -1 or tri_end_id == -1:
                            n_drop_event += 1
                            continue
                    else:
                        n_drop_event += 1
                        continue
                    
                arguments = []
                if 'argument' in event:
                    for arg in event['argument']:
                        arg_seg_id = -1
                        for i, sent_offsets in enumerate(seg_offsets):
                            if sent_offsets[0][0] <= arg['startOffset'] and arg['endOffset'] <= sent_offsets[-1][1]:
                                arg_seg_id = i
                                break
                        if arg_seg_id == -1 or tri_seg_id != arg_seg_id:
                            n_drop_arg += 1
                            continue
                            
                        start_id = -1
                        end_id = -1
                        for i, token_offsets in enumerate(seg_offsets[arg_seg_id]):
                            if token_offsets[0] == arg["startOffset"]:
                                start_id = i
                                for j in range(i, len(seg_offsets[arg_seg_id])):
                                    if seg_offsets[arg_seg_id][j][1] == arg["endOffset"]:
                                        end_id = j + 1
                                break

                        if start_id == -1 or end_id == -1:
                            if obj['content'][arg["startOffset"]:arg["endOffset"]] != arg["text"] and obj['content'][arg["startOffset"]-1:arg["endOffset"]-1] == arg["text"]:
                                start_id = -1
                                end_id = -1
                                
                                for i, token_offsets in enumerate(seg_offsets[arg_seg_id]):
                                    if token_offsets[0] == arg["startOffset"]-1:
                                        start_id = i
                                        for j in range(i, len(seg_offsets[arg_seg_id])):
                                            if seg_offsets[arg_seg_id][j][1] == arg["endOffset"]-1:
                                                end_id = j + 1
                                        break
                                
                                if start_id == -1 or end_id == -1:
                                    n_drop_arg += 1
                                    continue
                            else:
                                n_drop_arg += 1
                                continue
                        
                        entity_id = f"{obj['sourcefile'][:-4]}_Seg{tri_seg_id}_Ent{len(entity_mentions[tri_seg_id])}"
                        entity = {
                            "id": entity_id, 
                            "text": arg["text"], 
                            "entity_type": "Entity", 
                            "start": start_id, 
                            "end": end_id, 
                        }
                        entity_mentions[tri_seg_id].append(entity)
                        
                        argument = {
                            "entity_id": entity_id, 
                            "role": arg["role"]["type"], 
                            "text": arg["text"], 
                            "start": start_id, 
                            "end": end_id, 
                        }
                        arguments.append(argument)
                    arguments.sort(key=lambda x: (x["start"], x["end"]))
                
                event_ = {
                    "id": f"{obj['sourcefile'][:-4]}_Seg{tri_seg_id}_Evt{len(event_mentions[tri_seg_id])}",
                    "event_type": f"{event['type']}:{event['subtype']}",
                    "trigger": {
                        "text": trigger["text"], 
                        "start": tri_start_id, 
                        "end": tri_end_id, 
                    },
                    "arguments": arguments, 
                }
                event_mentions[tri_seg_id].append(event_)
                
        for i in range(n_seg):
            entity_mentions_ = entity_mentions[i]
            event_mentions_ = event_mentions[i]
            
            entity_mentions_ = sorted(entity_mentions_, key=lambda x: (x["start"], x["end"]))
            event_mentions_ = sorted(event_mentions_, key=lambda x: (x["trigger"]["start"], x["trigger"]["end"]))
            tokens = [t for j in range(sents[i][0], sents[i][1]) for t in text_tokens[j]]
            
            dt = {
                "doc_id": obj['sourcefile'][:-4], 
                "wnd_id": f"{obj['sourcefile'][:-4]}_{i}", 
                "text": obj["content"][seg_offsets[i][0][0]:seg_offsets[i][-1][1]], 
                "tokens": tokens, 
                "event_mentions": event_mentions_, 
                "entity_mentions": entity_mentions_, 
                "lang": "en", 
            }
            data.append(dt)
        
    print(f"Number of dropped events: {n_drop_event}")
    print(f"Number of dropped arguments: {n_drop_arg}")
            
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
    parser.add_argument('--seg_map', default="seg_map.json")
    args = parser.parse_args()
    
    objs = read_docs(args.in_folder)
    
    with open(args.token_map) as fp:
        token_map = json.load(fp)
    
    with open(args.seg_map) as fp:
        seg_map = json.load(fp)
        
    train_objs, dev_objs, test_objs = get_split(objs, args.split)

    train_data = convert_format(train_objs, token_map, seg_map)
    dev_data = convert_format(dev_objs, token_map, seg_map)
    test_data = convert_format(test_objs, token_map, seg_map)
    
    
    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
