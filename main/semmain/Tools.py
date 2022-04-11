import torch
from torch import nn
from torch_geometric.nn import RGCNConv
import random
from sklearn.metrics import f1_score as f1

def load_emb(idx, emb):
    output = []
    if len(idx) > 0:
        for i in idx:
            output.append(emb[i])
    return output

def connect_link(link_a, link_b, bias):
    link_b = torch.stack([link_b[0], link_b[1]+bias])
    output = torch.cat([link_a,link_b],dim=1)
    return output


def rand_del_link(link_input, rate):
    new_link = [[],[]]
    link_input = link_input.tolist()
    for i in range(0,len(link_input[0])):
        if random.randint(0,100) < rate:
            continue
        for j in range(0,2):
            new_link[j].append(link_input[j][i])
    return torch.tensor(new_link, dtype=torch.long)

class GatedRGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GatedRGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.RGCN1 = RGCNConv(
            in_channels=out_channels, out_channels=out_channels, num_relations=num_relations)
        self.attention_layer = nn.Linear(2 * out_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        nn.init.xavier_uniform_(
            self.attention_layer.weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, node_features, edge_index, edge_type):

        u_0 = self.RGCN1(node_features, edge_index, edge_type)
        a_1 = self.sigmoid(self.attention_layer(
            torch.cat((u_0, node_features), dim=1)))
        h_1 = self.tanh(u_0) * a_1 + node_features * (1 - a_1)

        return h_1

class attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        torch.nn.init.kaiming_uniform(
            self.linear.weight, nonlinearity='leaky_relu')
        self.softmax = nn.Softmax()

    def forward(self, att_val, input):
        att_val = self.linear(att_val).squeeze(0)
        att_a = []
        input_num = len(input)
        for i in range(0, input_num):
            att_a.append(torch.dot(att_val, input[i]))
        att_a = self.softmax(torch.stack(att_a))
        output = 0
        for i in range(0, input_num):
            output = output + att_a[i] * input[i]
        return output

def get_metrics(probs, labels):
    probs = torch.argmax(probs, dim=1)
    correct = 0
    for i in range(len(probs)):
        if probs[i] == labels[i]:
            correct += 1
    return correct / len(probs)


def pad_collate(x):
    return x


def f1_score(probs, labels):
    y_true = labels.tolist()
    y_pred = (torch.argmax(probs, dim=1)).tolist()
    F1_result = f1(y_true,y_pred, average='binary')
    return F1_result
