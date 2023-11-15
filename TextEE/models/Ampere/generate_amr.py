import argparse, json, os
from collections import Counter, defaultdict
from tqdm import tqdm
import amrlib
import ipdb

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, required=True)
parser.add_argument("-o", "--output_path", type=str, required=True)
parser.add_argument("-b", "--batch_size", type=int, default=32)
parser.add_argument("-x", "--model_dir", type=str, default="model_parse_spring-v0_1_0/")
args = parser.parse_args()

stog = amrlib.load_stog_model(model_dir=args.model_dir, device='cuda:0', batch_size=args.batch_size, num_beams=4)

with open(args.input_path) as fp:
    inp = [json.loads(line) for line in fp]

with open(args.output_path, 'w') as fp:
    sents = []
    stored_doc = []
    counter = 0
    for doc in tqdm(inp, ncols=100):
        if len(doc["event_mentions"]) == 0:
            continue
        sents.append(doc['text'])
        stored_doc.append(doc)
        counter += 1
        if counter % args.batch_size == 0:
            graphs = stog.parse_sents(sents)
            for graph, d in zip(graphs, stored_doc):
                x = {
                    "doc_id": d["doc_id"], 
                    "wnd_id": d["wnd_id"],
                    "amrgraph": graph,
                }
                fp.write(json.dumps(x) + '\n')
            sents = []
            stored_doc = []
            counter = 0
    if len(sents) > 0:
        graphs = stog.parse_sents(sents)
        for graph, d in zip(graphs, stored_doc):
            x = {
                "doc_id": d["doc_id"], 
                "wnd_id": d["wnd_id"],
                "amrgraph": graph,
            }
            fp.write(json.dumps(x) + '\n')
        counter = 0
