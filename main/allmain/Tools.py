import torch
from torch import nn
from torch_geometric.nn import RGCNConv

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
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)
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
    # # hit 123
    # hit1 = 0
    # hit2 = 0
    # hit3 = 0
    # for i in range(len(labels)):
    #     temp = probs[i].clone()
    #     if torch.argmax(temp) == labels[i]:
    #         hit1 += 1
    #         hit2 += 1
    #         hit3 += 1
    #         continue
    #     temp[torch.argmax(temp)] = 0
    #     if torch.argmax(temp) == labels[i]:
    #         hit2 += 1
    #         hit3 += 1
    #         continue
    #     temp[torch.argmax(temp)] = 0
    #     if torch.argmax(temp) == labels[i]:
    #         hit3 += 1
    #         continue
    # hit1 = hit1 / len(labels)
    # hit2 = hit2 / len(labels)
    # hit3 = hit3 / len(labels)
    # F1 and accs
    TP = [0,0,0]
    TN = [0,0,0]
    FP = [0,0,0]
    FN = [0,0,0]
    for i in range(len(labels)):
        temp = probs[i]
        if torch.argmax(temp) == labels[i]:
            TP[labels[i]] += 1
            for j in range(3):
                if not j == labels[i]:
                    TN[j] += 1
        else:
            FP[torch.argmax(temp)] += 1
            FN[labels[i]] += 1
            for j in range(3):
                if not j == torch.argmax(temp) and not j == labels[i]:
                    TN[j] += 1
    
    precision = [TP[i] / max(TP[i] + FP[i], 1) for i in range(3)]
    recall = [TP[i] / max(TP[i] + FN[i], 1) for i in range(3)]
    F1 = [2 * precision[i] * recall[i] / max(precision[i] + recall[i], 1) for i in range(3)]
    #macro_precision = sum(precision) / 5
    #macro_recall = sum(recall) / 5
    macro_F1 = sum(F1) / 3
    # micro_precision = sum(TP) / (sum(TP) + sum(FP))
    # micro_recall = sum(TP) / (sum(TP) + sum(FN))
    # assert (micro_precision == micro_recall)
    # assert (micro_precision == hit1)
    # micro_F1 = micro_precision
    return macro_F1
    # return {'hit1':hit1, 'hit2':hit2, 'hit3':hit3, 'micro_F1':micro_F1, 'macro_F1':macro_F1}