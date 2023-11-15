import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class EdgeGAT(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_out_dim, lamda):
        super(EdgeGAT, self).__init__()
        # Dimensions
        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim
        self.edge_in_dim = edge_in_dim
        self.mp_level = lamda
        # Linear Layers
        self.fc = nn.Linear(node_in_dim, node_out_dim, bias=False)
        self.edge_fc = nn.Linear(edge_in_dim, edge_in_dim)
        self.attn_fc = nn.Linear(2 * node_out_dim + edge_in_dim, 1, bias=False)
        # Activation Function
        self.actv = nn.ReLU()
        
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['embed']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z' : edges.src['z'], 'e' : edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def forward(self, g, node_h, edge_h):
        if g.num_edges() == 0:
            return node_h
        else:
            # edge_h is the edge_embeddings matrix according to 
            z = self.fc(node_h)
            edge_z = self.edge_fc(edge_h)
            g.ndata['z'] = z
            g.edata['embed'] = edge_z
            g.apply_edges(self.edge_attention)
            g.update_all(self.message_func, self.reduce_func)
            out = g.ndata.pop('h')
            final_out = (1 - self.mp_level) * node_h + self.mp_level * out
            return final_out


class MultiGAT(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, node_out_dim, layers, lamda):
        super(MultiGAT, self).__init__()
        self.node_in_dim = node_in_dim
        self.node_out_dim = node_out_dim
        self.layers = layers
        self.edge_in_dim = edge_in_dim
        self.lamda = lamda
        self.gats = nn.ModuleList([EdgeGAT(node_in_dim, edge_in_dim, node_out_dim, lamda) for _ in range(layers)])
    
    def forward(self, g, node_h, edge_h):
        output_h = node_h
        for i in range(self.layers):
            output_h = self.gats[i](g, output_h, edge_h)
            if i != self.layers - 1:
                output_h = nn.ReLU()(output_h)
        return output_h


class FinalGNN(nn.Module):
    
    def __init__(self, bert_dim, edge_dim, edge_type_num, layers, lamda, device):
        super(FinalGNN, self).__init__()
        self.bert_dim = bert_dim
        self.edge_dim = edge_dim
        self.nlayers = layers
        self.edge_type_num = edge_type_num
        self.device = device
        self.gnn = MultiGAT(bert_dim, edge_dim, bert_dim, layers, lamda)
        # for i in range(len(self.gnn.gats)):
        #     self.gnn.gats[i].attn_fc.to(device)
        #     self.gnn.gats[i].edge_fc.to(device)
        #     self.gnn.gats[i].fc.to(device)
        self.edge_embeds = nn.Embedding(edge_type_num, edge_dim)

    def forward(self, g, amr_emb):
        # amr_emb: (max_seq_len, bert_dim)

        edge_idx = g.edata['type'].squeeze(1)
        edge_embed = self.edge_embeds(edge_idx)
        nodes_out = self.gnn(g, amr_emb, edge_embed)
        return nodes_out
    
    
gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


if __name__ == "__main__":
    gat = EdgeGAT(5, 5, 5, 0.001).to(0)
    
    a = dgl.DGLGraph()
    a.add_nodes(4)
    a.add_edge(0, 3)
    a.add_edge(3, 0)
    a.add_edge(1, 2)
    a.add_edge(2, 1)
    
    a.ndata["token_pos"] = torch.LongTensor([[0], [1], [2], [3]])
    a.ndata["token_span"] = torch.LongTensor([[0, 0], [1, 2], [2, 3], [3, 4]])
    a.ndata["priority"] = torch.LongTensor([[0], [1], [2], [3]])
    a.edata["type"] = torch.LongTensor([[0], [1], [2], [3]])
    
    a = a.to(0)
    
    print(a)
    node_h = torch.rand(4, 5).to(0)
    edge_h = torch.rand(4, 5).to(0)
    out = gat(a, node_h, edge_h)