import torch
import dgl
import json
from argparse import ArgumentParser

from transition_amr_parser.stack_transformer_amr_parser import AMRParser

def get_amr_edge_idx(edge_type_str):
    if edge_type_str in ['location', 'destination', 'path']:
        return 0
    elif edge_type_str in ['year', 'time', 'duration', 'decade', 'weekday']:
        return 1
    elif edge_type_str in ['instrument', 'manner', 'poss', 'topic', 'medium', 'duration']:
        return 2
    elif edge_type_str in ['mod']:
        return 3
    elif edge_type_str.startswith('prep-'):
        return 4
    elif edge_type_str.startswith('op') and edge_type_str[-1].isdigit():
        return 5
    elif edge_type_str == 'ARG0':
        return 6
    elif edge_type_str == 'ARG1':
        return 7
    elif edge_type_str == 'ARG2':
        return 8
    elif edge_type_str == 'ARG3':
        return 9
    elif edge_type_str == 'ARG4':
        return 10
    else:
        return 11

def amr_parse(tokens_list, output_dir, amrparser_path):
    parser = AMRParser.from_checkpoint(amrparser_path)
    amr_list = parser.parse_sentences(tokens_list)
    torch.save(amr_list, output_dir)

def processing_amr(amr_dir, tokens_list):
    amr_list = torch.load(amr_dir)

    node_idx_list, edge_type_list, node_idx_offset_list, node_idx_offset_whole = [], [], [], []
    list_of_align_dict = []
    list_of_exist_dict = []

    total_edge_num = 0
    covered_edge_num = 0
    order_list = []
    for i, amr in enumerate(amr_list):
        amr_split_list = amr.split('\n')
        # print(amr_split_list)
        node_to_idx, node_to_offset, node_to_offset_whole = {}, {}, {}
        node_num = 0
        # first to fill in the node list
        for line in amr_split_list:
            if line.startswith('# ::node'):
                node_split = line.split('\t')
                # print(node_split)
                if len(node_split) != 4:
                    # check if the alignment text spans exist
                    continue
                else:
                    align_span = node_split[3].split('-')
                    if not align_span[0].isdigit():
                        continue
                    head_word_idx = int(align_span[1]) - 1
                    try:
                        start = int(align_span[0])
                    except:
                        # print(amr_list[i])
                        raise ValueError
                    end = int(align_span[1])
                    if (start, end) not in list(node_to_offset_whole.values()):
                        node_to_offset.update({node_split[1]: head_word_idx})
                        node_to_offset_whole.update({node_split[1]: (start, end)})
                        node_to_idx.update({node_split[1]: node_num})
                        node_num += 1
            else:
                continue

        node_idx_list.append(node_to_idx)
        # change str2offset to idx2offset
        node_idx_to_offset = {}
        for key in node_to_idx.keys():
            node_idx_to_offset.update({node_to_idx[key]: node_to_offset[key]})

        node_idx_to_offset_whole = {}
        for key in node_to_idx.keys():
            node_idx_to_offset_whole.update({node_to_idx[key]: node_to_offset_whole[key]})

        node_idx_offset_list.append(node_idx_to_offset)
        node_idx_offset_whole.append(node_idx_to_offset_whole)
        edge_type_dict = {}

        for line in amr_split_list:
            if line.startswith('# ::root'):
                root_split = line.split('\t')
                root = root_split[1]
        prior_dict = {root:[]}

        start_list = []
        end_list = []

        for line in amr_split_list:
            if line.startswith('# ::edge'):
                edge_split = line.split('\t')
                amr_edge_type = edge_split[2]
                edge_start = edge_split[4]
                edge_end = edge_split[5]
                # check if the start and end nodes exist
                if (edge_start in node_to_idx) and (edge_end in node_to_idx):
                    # check if the edge type is "ARGx-of", if so, reverse the direction of the edge
                    if amr_edge_type.startswith("ARG") and amr_edge_type.endswith("-of"):
                        edge_start, edge_end = edge_end, edge_start
                        amr_edge_type = amr_edge_type[0:4]
                    # deal with this edge here
                    edge_idx = get_amr_edge_idx(amr_edge_type)
                    total_edge_num += 1
                    if edge_idx == 11:
                        covered_edge_num += 1
                    start_idx = node_to_idx[edge_start]
                    end_idx = node_to_idx[edge_end]
                    edge_type_dict.update({(start_idx, end_idx): edge_idx})
                
                else:
                    continue
                # print(edge_start, edge_end)
                if edge_end != root and (not ((edge_start in end_list) and (edge_end in start_list))):
                    start_list.append(edge_start)
                    end_list.append(edge_end)
                if edge_start not in prior_dict:
                    prior_dict.update({edge_start:[edge_end]})
                else:
                    prior_dict[edge_start].append(edge_end)
            else:
                continue
        edge_type_list.append(edge_type_dict)
        # generating priority list for decoding
        final_order_list = []
        # output orders
        candidate_nodes = node_to_idx.copy()
        while len(candidate_nodes) != 0:
            current_level_nodes = []
            for key in candidate_nodes:
                if key not in end_list:
                    final_order_list.append(candidate_nodes[key])
                    current_level_nodes.append(key)
            # Remove current level nodes from the dictionary
            for node in current_level_nodes:
                candidate_nodes.pop(node)
            
            # deleting from start lists the current level nodes
            for node in current_level_nodes:
                indices_list = [i for i, x in enumerate(start_list) if x == node]
                start_list = [x for x in start_list if x != node]
                new_end_list = []
                for i in range(len(end_list)):
                    if i not in indices_list:
                        new_end_list.append(end_list[i])
                end_list = new_end_list

        order_list.append(final_order_list.copy())
    # feed into dgl graphs
    graphs_list = []

    for i in range(len(node_idx_list)):
        graph_i = dgl.DGLGraph()

        edge2type = edge_type_list[i]
        node2offset = node_idx_offset_list[i]
        node2offset_whole = node_idx_offset_whole[i]

        nodes_num = len(node2offset)

        graph_i.add_nodes(nodes_num)
        graph_i.ndata['token_pos'] = torch.zeros(nodes_num, 1, dtype=torch.long)
        graph_i.ndata['token_span'] = torch.zeros(nodes_num, 2, dtype=torch.long)

        # fill in token positions
        for key in node2offset:
            graph_i.ndata['token_pos'][key][0] = node2offset[key]
        for key in node2offset:
            graph_i.ndata['token_span'][key][0] = node2offset_whole[key][0]
            graph_i.ndata['token_span'][key][1] = node2offset_whole[key][1]
        # add nodes priorities
        node_prior_tensor = torch.zeros(nodes_num, 1, dtype=torch.long)
        for j in range(nodes_num):
            node_prior_tensor[j][0] = order_list[i].index(j)
        graph_i.ndata['priority'] = node_prior_tensor
        # add edges
        edge_num = len(edge2type)
    
        edge_iter = 0
        
        ''' bi-directional edges '''
        edge_type_tensor = torch.zeros(2 * edge_num, 1, dtype=torch.long)
        for key in edge2type:
            graph_i.add_edges(key[0], key[1])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1

        for key in edge2type:
            graph_i.add_edges(key[1], key[0])
            edge_type_tensor[edge_iter][0] = edge2type[key]
            edge_iter += 1
        
        graph_i.edata['type'] = edge_type_tensor
        graphs_list.append(graph_i)

        align_dict = {}
        exist_dict = {}

        span_list = graph_i.ndata["token_span"].tolist()

        for p in range(len(tokens_list[i])):
            min_dis = 2 * len(tokens_list[i])
            min_dis_idx = -1

            if_found = 0

            for q in range(len(span_list)):
                if p >= span_list[q][0] and p < span_list[q][1]:
                    if_found = 1
                    align_dict.update({p: q})
                    exist_dict.update({p: 1})
                    break
                else:
                    new_dis_1 = abs(p - span_list[q][0])
                    new_dis_2 = abs(p - (span_list[q][1] - 1))
                    new_dis = min(new_dis_1, new_dis_2)
                    if new_dis < min_dis:
                        min_dis = new_dis
                        min_dis_idx = q
            
            if not if_found:
                align_dict.update({p: min_dis_idx})
                exist_dict.update({p: 0})

        list_of_align_dict.append(align_dict)
        list_of_exist_dict.append(exist_dict)

    return graphs_list, list_of_align_dict, list_of_exist_dict


def get_amr_data(json_path, graph_pkl_path, amr_path, amrparser_path):
    with open(json_path, "r", encoding='utf-8') as f:
        sents = []
        done = 0
        sents = []
        
        while not done:
            line = f.readline()
            if line != '':
                data_dict = json.loads(line)
                sents.append(data_dict['tokens'])
            else:
                done = 1
    amr_parse(sents, amr_path, amrparser_path)
    graphs, align, exist = processing_amr(amr_path, sents)
    torch.save((graphs, align, exist), graph_pkl_path)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--parser', default='./amr_general/checkpoint_best.pt', help="The Path to AMRParser.")
    args = parser.parse_args()
    get_amr_data(f"{args.prefix}.json", f"{args.prefix}_graphs.pkl", f"{args.prefix}_amrs.pkl", args.parser)
    
    # Examples for precessing ace05 datasets
    # The model only need 'train.oneie.json' and 'train_graphs.pkl'. The 'train_amrs.pkl' is used to store the AMR graphs as strings.
    # get_amr_data("./data/ace05/train.oneie.json", "./data/ace05/train_graphs.pkl", "./data/ace05/train_amrs.pkl")
    # get_amr_data("./data/ace05/dev.oneie.json", "./data/ace05/dev_graphs.pkl", "./data/ace05/dev_amrs.pkl")
    # get_amr_data("./data/ace05/test.oneie.json", "./data/ace05/test_graphs.pkl", "./data/ace05/test_amrs.pkl")

