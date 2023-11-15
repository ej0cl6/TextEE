"""
This script extracts IE annotations from ACE2005 (LDC2006T06).

Usage:
python process_ace.py \
    
"""
import os
import re
import json
import glob
import tqdm, random
import torch
from argparse import ArgumentParser
import ipdb


def generate_split(seed, all_doc_set, save_path):
    train_docs, dev_docs, test_docs = set(), set(), set()
    random.seed(seed)
    doc_keys = list(all_doc_set)
    random.shuffle(doc_keys)
    test_splits = sorted(doc_keys[:60])
    dev_splits = sorted(doc_keys[60:120])
    train_splits = sorted(doc_keys[120:])

    with open(os.path.join(save_path, 'train.doc.txt'), 'w') as fw:
        for f in train_splits:
            fw.write(f+'\n')

    with open(os.path.join(save_path, 'dev.doc.txt'), 'w') as fw:
        for f in dev_splits:
            fw.write(f+'\n')

    with open(os.path.join(save_path, 'test.doc.txt'), 'w') as fw:
        for f in test_splits:
            fw.write(f+'\n')

def split_data(input_file: str,
               output_dir: str,
               split_path: str):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))
    
    # split the dataset
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(os.path.join(output_dir, 'train.json'), 'w') as w_train, \
        open(os.path.join(output_dir, 'dev.json'), 'w') as w_dev, \
        open(os.path.join(output_dir, 'test.json'), 'w') as w_test:
        for line in r:
            inst = json.loads(line)
            doc_id = inst['doc_id']
            if doc_id in train_docs:
                w_train.write(line)
            elif doc_id in dev_docs:
                w_dev.write(line)
            elif doc_id in test_docs:
                w_test.write(line)
            else:
                print('missing!! {}'.format(doc_id))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', help='Path to formatted full file')
    parser.add_argument('-o', '--output', help='Path to the output folder')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    input_path = args.input
    data = [json.loads(line) for line in open(input_path, 'r', encoding='utf-8')]
    all_doc = set()
    for d in data:
        all_doc.add(d['doc_id'])

    # randomly generate split 
    split_path = os.path.join(args.output, 'split_docs_seed{}'.format(args.seed))
    os.makedirs(split_path, exist_ok=True)
    generate_split(args.seed, all_doc, split_path)
    output_path = os.path.join(args.output, 'data_seed{}'.format(args.seed))
    os.makedirs(output_path, exist_ok=True)
    split_data(input_path, output_path, split_path)