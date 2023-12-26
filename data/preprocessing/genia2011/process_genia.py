import os, json
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from collections import defaultdict
from pprint import pprint
import ipdb

TRAIN_DIR = "BioNLP-ST_2011_genia_train_data_rev1"
DEV_DIR = "BioNLP-ST_2011_genia_devel_data_rev1"

def read_docs(path):
    all_objs = []
    
    file_list1 = [(f, TRAIN_DIR) for f in sorted(os.listdir(os.path.join(path, TRAIN_DIR))) if f.endswith("txt")]
    file_list2 = [(f, DEV_DIR) for f in sorted(os.listdir(os.path.join(path, DEV_DIR))) if f.endswith("txt")]
    for file, file_dir in tqdm(file_list1+file_list2, ncols=100):
    
        with open(os.path.join(path, file_dir, file)) as fp:
            text = fp.read().strip()
            
        wnd_id = file[:-4]
        doc_id = "-".join(wnd_id.split("-")[:2])
        
        with open(os.path.join(path, file_dir, f"{wnd_id}.a1")) as fp:
            lines1 = fp.readlines()
        with open(os.path.join(path, file_dir, f"{wnd_id}.a2")) as fp:
            lines2 = fp.readlines()
        
        t_lines = []
        r_lines = []
        e_lines = []
        q_lines = []
        for line in lines1+lines2:
            line = line.strip()
            if line.startswith("T"): # entity annotation
                t_lines.append(line)
            elif line.startswith("E"): # event annotation
                e_lines.append(line)
            elif line.startswith("R"): # relation annotation
                r_lines.append(line)
            elif line.startswith("*"): # equivalence annotation
                q_lines.append(line)
            elif line.startswith("A"): # attribute annotation
                continue
            elif line.startswith("M"): # modification annotation
                continue
            else:
                ipdb.set_trace()
        
        eq_map = defaultdict(set)
        for line in q_lines:
            line = line.strip()
            _, eq = line.split("\t")
            _, *eq_objs = eq.split()
            eq_objs = set(eq_objs)
            for eq_obj in sorted(eq_objs):
                eq_map[eq_obj] |= eq_objs - set([eq_obj])
        
        entity_mentions = {}
        for line in t_lines:
            entity_id, entity_info, entity_text = line.split("\t")

            entity_id = entity_id.strip()
            entity_info = entity_info.strip()
            entity_text = entity_text.strip()

            entity_type, start, end = entity_info.rsplit(" ", 2)
            start = int(start)
            end = int(end)

            entity = {
                "id": entity_id, 
                "text": entity_text, 
                "entity_type": entity_type, 
                "start": start, 
                "end": end, 
            }

            assert text[start:end] == entity_text
            entity_mentions[entity_id] = entity
            
        relation_mentions = []
        for line in r_lines:
            relation_id, relation_info = line.split("\t")
            
            relation_id = relation_id.strip()
            relation_info = relation_info.strip()
            
            role, arg1, arg2 = relation_info.split()
            
            arg1_role, arg1_id = arg1.split(":")
            arg2_role, arg2_id = arg2.split(":")
            
            relation = {
              "id": relation_id,
              "relation_type": role,
              "arguments": [
                {
                  "entity_id": arg1_id, 
                  "role": arg1_role,
                  "text": entity_mentions[arg1_id]["text"], 
                  "start": entity_mentions[arg1_id]["start"], 
                  "end": entity_mentions[arg1_id]["end"], 
                },
                {
                  "entity_id": arg2_id, 
                  "role": arg2_role,
                  "text": entity_mentions[arg2_id]["text"], 
                  "start": entity_mentions[arg2_id]["start"], 
                  "end": entity_mentions[arg2_id]["end"], 
                }
              ]
            }
            relation_mentions.append(relation)
                
        event_mentions = []
        for line in e_lines:
            event_id, event_info = line.split("\t")

            event_id = event_id.strip()
            event_info = event_info.strip()

            trigger_info, *args = event_info.split()

            arguments = []
            for arg in args:
                role, arg_id = arg.split(":")

                if arg_id.startswith("E"): # discard event cross-reference
                    continue

                argument = {
                    "entity_id": arg_id, 
                    "role": role, 
                    "text": entity_mentions[arg_id]["text"], 
                    "start": entity_mentions[arg_id]["start"], 
                    "end": entity_mentions[arg_id]["end"], 
                }
                arguments.append(argument)
                
                for eq_arg_id in sorted(eq_map[arg_id]):
                    argument = {
                        "entity_id": eq_arg_id, 
                        "role": role, 
                        "text": entity_mentions[eq_arg_id]["text"], 
                        "start": entity_mentions[eq_arg_id]["start"], 
                        "end": entity_mentions[eq_arg_id]["end"], 
                    }
                    arguments.append(argument)
                

            event_type, trigger_id = trigger_info.split(":")

            event = {
                "id": event_id,
                "event_type": event_type,
                "trigger": {
                    "text": entity_mentions[trigger_id]["text"], 
                    "start": entity_mentions[trigger_id]["start"], 
                    "end": entity_mentions[trigger_id]["end"], 
                },
                "arguments": arguments, 
            }
            event_mentions.append(event)
            
        obj = {
            "doc_id": doc_id, 
            "wnd_id": wnd_id, 
            "text": text, 
            "relation_mentions": relation_mentions, 
            "event_mentions": event_mentions, 
            "entity_mentions": list(entity_mentions.values()), 
            "lang": "en", 
        }
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


def convert_format(objs, token_map, seg_map):
    
    data = []
    n_drop_entity = 0
    n_drop_relation = 0
    n_drop_event = 0
    n_drop_arg = 0
    for obj in tqdm(objs, ncols=100):
        
        text_tokens = [[obj["text"][si:ei] for si, ei in sent] for sent in token_map[obj["wnd_id"]]]
        n_sent = len(text_tokens)
        sent_lens = [len(s) for s in text_tokens]
        
        sents = seg_map[obj["wnd_id"]]
        n_seg = len(sents)
        
        entity_mentions = defaultdict(list)
        relation_mentions = defaultdict(list)
        event_mentions = defaultdict(list)
        sent_offsets = token_map[obj["wnd_id"]]
        seg_offsets = []
        for si, ei in sents:
            segs = []
            for i in range(si, ei):
                segs.extend(sent_offsets[i])
            seg_offsets.append(segs)
            
        for entity in obj["entity_mentions"]:
            ent_seg_id = -1
            for i, sent_offsets in enumerate(seg_offsets):
                if sent_offsets[0][0] <= entity['start'] and entity['end'] <= sent_offsets[-1][1]:
                    ent_seg_id = i
                    break
            if ent_seg_id == -1:
                n_drop_entity += 1
                continue
                
            ent_start_id = -1
            ent_end_id = -1
            for i, token_offsets in enumerate(seg_offsets[ent_seg_id]):
                if token_offsets[0] == entity["start"]:
                    ent_start_id = i
                    for j in range(i, len(seg_offsets[ent_seg_id])):
                        if seg_offsets[ent_seg_id][j][1] == entity["end"]:
                            ent_end_id = j + 1
                    break

            if ent_start_id == -1 or ent_end_id == -1:
                n_drop_entity += 1
                continue
                
            entity_ = {
                "id": entity["id"],
                "text": entity["text"],
                "entity_type": entity["entity_type"],
                "start": ent_start_id, 
                "end": ent_end_id, 
            }
                
            entity_mentions[ent_seg_id].append(entity_)
            
        for relation in obj["relation_mentions"]:
            arg1 = relation["arguments"][0]
            arg1_seg_id = -1
            for i, sent_offsets in enumerate(seg_offsets):
                if sent_offsets[0][0] <= arg1['start'] and arg1['end'] <= sent_offsets[-1][1]:
                    arg1_seg_id = i
                    break
            if arg1_seg_id == -1:
                n_drop_relation += 1
                continue
                
            arg1_start_id = -1
            arg1_end_id = -1
            for i, token_offsets in enumerate(seg_offsets[arg1_seg_id]):
                if token_offsets[0] == arg1["start"]:
                    arg1_start_id = i
                    for j in range(i, len(seg_offsets[arg1_seg_id])):
                        if seg_offsets[arg1_seg_id][j][1] == arg1["end"]:
                            arg1_end_id = j + 1
                    break

            if arg1_start_id == -1 or arg1_end_id == -1:
                n_drop_relation += 1
                continue
                
            arg2 = relation["arguments"][0]
            arg2_seg_id = -1
            for i, sent_offsets in enumerate(seg_offsets):
                if sent_offsets[0][0] <= arg2['start'] and arg2['end'] <= sent_offsets[-1][1]:
                    arg2_seg_id = i
                    break
            if arg2_seg_id == -1:
                n_drop_relation += 1
                continue
                
            arg2_start_id = -1
            arg2_end_id = -1
            for i, token_offsets in enumerate(seg_offsets[arg2_seg_id]):
                if token_offsets[0] == arg2["start"]:
                    arg2_start_id = i
                    for j in range(i, len(seg_offsets[arg2_seg_id])):
                        if seg_offsets[arg2_seg_id][j][1] == arg2["end"]:
                            arg2_end_id = j + 1
                    break

            if arg2_start_id == -1 or arg2_end_id == -1:
                n_drop_relation += 1
                continue
                
            if arg1_seg_id != arg2_seg_id:
                n_drop_relation += 1
                continue
                
            
            assert arg1["entity_id"] in set(e["id"] for e in entity_mentions[arg1_seg_id])
            assert arg2["entity_id"] in set(e["id"] for e in entity_mentions[arg2_seg_id])
                
            relation_ = {
              "id": relation["id"],
              "relation_type": relation["relation_type"],
              "arguments": [
                {
                  "entity_id": arg1["entity_id"],
                  "role": arg1["role"],
                  "text": arg1["text"],
                  "start": arg1_start_id, 
                  "end": arg1_end_id, 
                },
                {
                  "entity_id": arg2["entity_id"],
                  "role": arg2["role"],
                  "text": arg2["text"],
                  "start": arg2_start_id, 
                  "end": arg2_end_id, 
                }
              ]
            }
            relation_mentions[arg1_seg_id].append(relation_)
                
        for event in obj["event_mentions"]:
            trigger = event["trigger"]
            tri_seg_id = -1
            for i, sent_offsets in enumerate(seg_offsets):
                if sent_offsets[0][0] <= trigger['start'] and trigger['end'] <= sent_offsets[-1][1]:
                    tri_seg_id = i
                    break
            if tri_seg_id == -1:
                n_drop_event += 1
                continue

            tri_start_id = -1
            tri_end_id = -1
            for i, token_offsets in enumerate(seg_offsets[tri_seg_id]):
                if token_offsets[0] == trigger["start"]:
                    tri_start_id = i
                    for j in range(i, len(seg_offsets[tri_seg_id])):
                        if seg_offsets[tri_seg_id][j][1] == trigger["end"]:
                            tri_end_id = j + 1
                    break

            if tri_start_id == -1 or tri_end_id == -1:
                n_drop_event += 1
                continue
        
            arguments = []
            for arg in event["arguments"]:
                arg_seg_id = -1
                for i, sent_offsets in enumerate(seg_offsets):
                    if sent_offsets[0][0] <= arg['start'] and arg['end'] <= sent_offsets[-1][1]:
                        arg_seg_id = i
                        break
                if arg_seg_id == -1 or tri_seg_id != arg_seg_id:
                    n_drop_arg += 1
                    continue

                start_id = -1
                end_id = -1
                for i, token_offsets in enumerate(seg_offsets[arg_seg_id]):
                    if token_offsets[0] == arg["start"]:
                        start_id = i
                        for j in range(i, len(seg_offsets[arg_seg_id])):
                            if seg_offsets[arg_seg_id][j][1] == arg["end"]:
                                end_id = j + 1
                        break

                if start_id == -1 or end_id == -1:
                    if obj['text'][arg["start"]:arg["end"]] != arg["text"] and obj['text'][arg["start"]-1:arg["end"]-1] == arg["text"]:
                        start_id = -1
                        end_id = -1

                        for i, token_offsets in enumerate(seg_offsets[arg_seg_id]):
                            if token_offsets[0] == arg["start"]-1:
                                start_id = i
                                for j in range(i, len(seg_offsets[arg_seg_id])):
                                    if seg_offsets[arg_seg_id][j][1] == arg["end"]-1:
                                        end_id = j + 1
                                break

                        if start_id == -1 or end_id == -1:
                            n_drop_arg += 1
                            continue
                    else:
                        n_drop_arg += 1
                        continue
                        
                assert arg["entity_id"] in set(e["id"] for e in entity_mentions[tri_seg_id])

                argument = {
                    "entity_id": arg["entity_id"], 
                    "role": arg["role"], 
                    "text": arg["text"], 
                    "start": start_id, 
                    "end": end_id, 
                }
                arguments.append(argument)
            arguments.sort(key=lambda x: (x["start"], x["end"]))

            event_ = {
                "id": event["id"],
                "event_type": event["event_type"],
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
                "doc_id": obj['doc_id'], 
                "wnd_id": f"{obj['wnd_id']}_{i}", 
                "text": obj["text"][seg_offsets[i][0][0]:seg_offsets[i][-1][1]], 
                "tokens": tokens, 
                "event_mentions": event_mentions_, 
                "entity_mentions": entity_mentions_, 
                "lang": "en", 
            }
            data.append(dt)
        
    print(f"Number of dropped entitys: {n_drop_entity}")
    print(f"Number of dropped relations: {n_drop_relation}")
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
        
    all_data = convert_format(objs, token_map, seg_map)
    
    train_data, dev_data, test_data = get_split(all_data, args.split)
    

    print("Train")
    get_statistics(train_data)
    print("Dev")
    get_statistics(dev_data)
    print("Test")
    get_statistics(test_data)
    
    save_data(args.out_folder, args.split, train_data, dev_data, test_data)
    
        
if __name__ == "__main__":
    main()
