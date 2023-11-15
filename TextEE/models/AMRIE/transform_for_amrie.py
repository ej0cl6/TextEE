import math
import json
from argparse import ArgumentParser


def tokens_to_offset_list(tokens_list):
    tokens_num = len(tokens_list)
    curr_len = 0
    offsets_list = []

    for i in range(tokens_num):
        start = curr_len
        end = curr_len + len(tokens_list[i])
        curr_len = end + 1

        offsets_list.append((start, end))
    
    return offsets_list

def span_to_offset(offset_list, start, end):
    # offset_list: list of tuples, span: [s1, s2]
    start_offset = offset_list[start][0]
    end_offset = offset_list[end - 1][1]
    return start_offset, end_offset

def transform_for_amrie(data_dict, tokenizer):
    new_data_dict = data_dict.copy()

    tokens = data_dict["tokens"]
    recomb_sent = " ".join(tokens)
    # generate the string offsets for each token
    offset_list = tokens_to_offset_list(tokens)
    # modify the dictionarys for entities, event triggers
    tmp_entities = []
    for i, entity in enumerate(data_dict["entity_mentions"]):
        new_start, new_end = span_to_offset(offset_list, entity["start"], entity["end"])
        tmp_entity = entity.copy()
        tmp_entity["start"] = new_start
        tmp_entity["end"] = new_end
        tmp_entities.append(tmp_entity)
    
    tmp_events = []
    for i, event in enumerate(data_dict["event_mentions"]):
        new_start, new_end = span_to_offset(offset_list, event["trigger"]["start"], event["trigger"]["end"])
        tmp_event = event.copy()
        tmp_event["start"] = new_start
        tmp_event["end"] = new_end
        tmp_events.append(tmp_event)
    
    # do roberta tokenization
    pieces = tokenizer.tokenize(recomb_sent)
    offset_mappings = tokenizer(recomb_sent, return_offsets_mapping=True)["offset_mapping"][1:-1].copy()
    offset_mappings.insert(0, (-math.inf, -math.inf))
    offset_mappings.append((math.inf, math.inf))

    # map the spans back to the offset mappings
    for i, entity in enumerate(tmp_entities):
        ent_start = entity["start"]
        ent_end = entity["end"]

        # first find out the minimum start
        for j in range(1, len(offset_mappings)):
            if offset_mappings[j][0] <= ent_start and offset_mappings[j+1][0] > ent_start:
                break
        span_start = j - 1

        # then find out the minimum end
        for j in range(0, len(offset_mappings)-1):
            if offset_mappings[j][1] < ent_end and offset_mappings[j+1][1] >= ent_end:
                break
        span_end = j + 1

        new_data_dict["entity_mentions"][i]["start"] = span_start
        new_data_dict["entity_mentions"][i]["end"] = span_end
    
    # map the spans back to offset mappings
    for i, event in enumerate(tmp_events):
        ent_start = event["start"]
        ent_end = event["end"]

        # first find out the minimum start
        for j in range(1, len(offset_mappings)):
            if offset_mappings[j][0] <= ent_start and offset_mappings[j+1][0] > ent_start:
                break
        span_start = j - 1

        # then find out the minimum end
        for j in range(0, len(offset_mappings)-1):
            if offset_mappings[j][1] < ent_end and offset_mappings[j+1][1] >= ent_end:
                break

        span_end = j + 1

        new_data_dict["event_mentions"][i]["trigger"]["start"] = span_start
        new_data_dict["event_mentions"][i]["trigger"]["end"] = span_end
    
    tokenslist = []

    for i in range(len(pieces)):
        if pieces[i].startswith('\u0120'):
            tokenslist.append(pieces[i][1:])
        else:
            tokenslist.append(pieces[i])
    
    assert (len(tokenslist) == len(pieces))

    new_data_dict["tokens"] = tokenslist
    new_data_dict["pieces"] = pieces
    new_data_dict["token_lens"] = [1 for _ in range(len(pieces))]
    new_data_dict["sentence"] = recomb_sent

    return new_data_dict


def transform_dataset(data_dir, output_dir, tokenizer):
    with open(data_dir, "r", encoding="utf-8") as f:
        with open(output_dir, "w", encoding="utf-8") as f1:
            done = 0
            while not done:
                line = f.readline()
                if line != "":
                    data_dict = json.loads(line)
                    output_dict = transform_for_amrie(data_dict, tokenizer)
                    output_line = json.dumps(output_dict) + '\n'
                    f1.write(output_line)
                else:
                    done = 1


if __name__ == "__main__":
    from transformers import RobertaTokenizerFast
    t = RobertaTokenizerFast.from_pretrained("roberta-large")
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', default='./data/ere_amrie/test.oneie.json')
    parser.add_argument('-i', '--input', default='./data/ere_oneie/test.oneie.json')
    args = parser.parse_args()
    transform_dataset(args.input, args.output, t)

    





