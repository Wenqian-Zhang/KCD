# GateRGCN
# Add learning rate scheduler
import torch
from torch.utils.data import Dataset
from Tools import connect_link, load_emb, rand_del_link
import pickle


class KSDDataset(Dataset):
    def __init__(self, name, foldid, path, del_type=-1, del_rate=0):
        ####################################
        count = 0
        self.name = name
        self.graph = []
        # input article id of each fold, fold[i][0] is train; fold[i][1] is dev
        foldidx = torch.load(path + 'foldindex.pt')  # list
        ###
        sen2sen = torch.load(path + 'sen2sen.pt')  # list
        # list [dict : ent-para link; entity index]
        sen2ent = torch.load(path + 'sen2ent.pt')
        # list [dict : ent-para link; entity index]
        sen2top = torch.load(path + 'sen2top.pt')
        # list [dict : ent-para link; entity index]
        sen2sent = torch.load(path + 'sen2sent.pt')
        # list [dict : ent-para link; entity index]
        sen2quo = torch.load(path + 'sen2quo.pt')
        # list [dict : ent-para link; entity index]
        sen2ten = torch.load(path + 'sen2ten.pt')
        ###
        # input embedding, dict, key is id
        art_emb = torch.load(path + 'ArticleEmbedding.pt')  # dict key=0~645
        ent_emb = torch.load(path + 'EntityEmbedding.pt')  # dict key= entityid
        top_emb = torch.load(path + 'TopicEmbedding.pt')
        kgp_emb = torch.load(path + 'KGPEmbedding.pt')
        # input label
        label = torch.load(path + 'Label.pt')

        if self.name == "train":
            train_idx = foldidx[foldid][0]
            for i in range(0, len(train_idx)):
                gid = int(train_idx[i])
                graph = {}
                graph['art_emb'] = art_emb[gid]
                art_num = len(art_emb[gid])
                graph_sen2sen = sen2sen[gid]
                graph_sen2ent = sen2ent[gid]['sen2ent']
                graph_ent2id = sen2ent[gid]['ent2id']
                graph['ent_emb'] = load_emb(graph_ent2id.tolist(), ent_emb)
                graph['top_emb'] = load_emb(sen2top[gid]['top2id'].tolist(), top_emb)
                graph['kgp_emb'] = load_emb(graph_ent2id.tolist(), kgp_emb)
                graph['sen2ent'] = graph_sen2ent
                # ent
                if del_type == 1:
                    graph_sen2ent = rand_del_link(graph_sen2ent, del_rate)
                graph_index = connect_link(
                    graph_sen2sen, graph_sen2ent, art_num)
                # top
                if del_type == 2:
                    sen2top[gid]['sen2top'] = rand_del_link(sen2top[gid]['sen2top'], del_rate)
                graph_index = connect_link(
                    graph_index, sen2top[gid]['sen2top'], art_num + len(graph_ent2id))
                # sent
                if del_type == 3:
                    sen2sent[gid] = rand_del_link(sen2sent[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2sent[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']))
                # quo
                if del_type == 4:
                    sen2quo[gid] = rand_del_link(sen2quo[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2quo[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']) + 2)
                # tense
                if del_type == 5:
                    sen2ten[gid] = rand_del_link(sen2ten[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2ten[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']) + 4)

                graph_type = torch.tensor([0]*len(graph_sen2sen[0]) + [1]*len(graph_sen2ent[0]) + [2]*len(
                    sen2top[gid]['sen2top'][0]) + [3]*len(sen2sent[gid][0]) + [4]*len(sen2quo[gid][0]) + [5]*len(sen2ten[gid][0]))

                graph['edge_index'] = torch.stack([torch.cat(
                    [graph_index[0], graph_index[1]]), torch.cat([graph_index[1], graph_index[0]])])
                graph['edge_type'] = torch.cat([graph_type, graph_type])
                graph['label'] = label[gid]

                self.graph.append(graph)

        if self.name == "dev":
            val_idx = foldidx[foldid][1]
            for i in range(0, len(val_idx)):
                gid = val_idx[i]
                graph = {}
                graph['art_emb'] = art_emb[gid]
                art_num = len(art_emb[gid])
                graph_sen2sen = sen2sen[gid]
                graph_sen2ent = sen2ent[gid]['sen2ent']
                graph_ent2id = sen2ent[gid]['ent2id']
                graph['ent_emb'] = load_emb(graph_ent2id.tolist(), ent_emb)
                graph['top_emb'] = load_emb(sen2top[gid]['top2id'].tolist(), top_emb)
                graph['kgp_emb'] = load_emb(graph_ent2id.tolist(), kgp_emb)
                graph['sen2ent'] = graph_sen2ent
                # ent
                if del_type == 1:
                    graph_sen2ent = rand_del_link(graph_sen2ent, del_rate)
                graph_index = connect_link(
                    graph_sen2sen, graph_sen2ent, art_num)
                # top
                if del_type == 2:
                    sen2top[gid]['sen2top'] = rand_del_link(sen2top[gid]['sen2top'], del_rate)
                graph_index = connect_link(
                    graph_index, sen2top[gid]['sen2top'], art_num + len(graph_ent2id))
                # sent
                if del_type == 3:
                    sen2sent[gid] = rand_del_link(sen2sent[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2sent[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']))
                # quo
                if del_type == 4:
                    sen2quo[gid] = rand_del_link(sen2quo[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2quo[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']) + 2)
                # tense
                if del_type == 5:
                    sen2ten[gid] = rand_del_link(sen2ten[gid], del_rate)
                graph_index = connect_link(
                    graph_index, sen2ten[gid], art_num + len(graph_ent2id) + len(sen2top[gid]['top2id']) + 4)

                graph_type = torch.tensor([0]*len(graph_sen2sen[0]) + [1]*len(graph_sen2ent[0]) + [2]*len(
                    sen2top[gid]['sen2top'][0]) + [3]*len(sen2sent[gid][0]) + [4]*len(sen2quo[gid][0]) + [5]*len(sen2ten[gid][0]))

                graph['edge_index'] = torch.stack([torch.cat(
                    [graph_index[0], graph_index[1]]), torch.cat([graph_index[1], graph_index[0]])])
                graph['edge_type'] = torch.cat([graph_type, graph_type])
                graph['label'] = label[gid]

                self.graph.append(graph)

        self.length = len(self.graph)
        print(self.name, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        return self.graph[index]
