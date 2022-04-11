# server209
# GateRGCN
# Add learning rate scheduler
import torch
import numpy as ny
from torch.utils.data import Dataset
import pickle



class KSDDataset(Dataset):
    def __init__(self, name, foldid, path):


####################################
        count = 0
        self.name = name
        self.graph = []
        f = open(path + 'Info/newsbias_random_fold_' + str(foldid) + '.pickle','rb')
        info = pickle.load(f)
        ent_emb = torch.load(path + 'Info/EntityEmbedding.pt')
        top_emb = torch.load(path + 'Info/TopicEmbedding.pt')
        kgp_emb = ny.load(path + 'Info/KGPEmbedding.npy', allow_pickle = True).item() #dict, index = '0' ~ '1070', content = list

        if self.name == "train":
            train_idx = info['train_idx'].tolist()
            for i in range(0, len(train_idx)):
                try:
                    graph = ny.load(path + 'Train/' + str(train_idx[i]) + '.npy', allow_pickle = True).item()
                    ent_link = graph['ent_link']
                    
                    sen_kgp_emb = []
                    sen_ent_emb = []
                    sen_top_emb = []

                    if(len(graph['ent_inx']) > 0):
                        for e_in in graph['ent_inx']:
                            sen_ent_emb.append(ent_emb[int(e_in)])
                    for t_in in graph['top_inx']:
                        sen_top_emb.append(top_emb[t_in])

                    for s_l in ent_link:
                        ####句子循环
                        sen_kgp = [] #每一句的entities的kpgs
                        if(len(s_l) > 0):
                            for e_i in s_l:
                                sen_kgp = sen_kgp + kgp_emb[e_i]
                            sen_kgp = list(set(sen_kgp)) 
                        sen_kgp_emb.append(sen_kgp)
                    
                    graph['ent_emb'] = sen_ent_emb
                    graph['top_emb'] = sen_top_emb
                    graph['kgp'] = sen_kgp_emb
                    
                    self.graph.append(graph)
                except:
                    count = 1


        if self.name == "dev":
            val_idx = info['test_idx'].tolist()
            for i in range(0, len(val_idx)):
                try:
                    
                    graph = ny.load(path + 'Train/' + str(val_idx[i]) + '.npy', allow_pickle = True).item()
                    ent_link = graph['ent_link']
                    
                    sen_kgp_emb = []
                    sen_ent_emb = []
                    sen_top_emb = []

                    if(len(graph['ent_inx']) > 0):
                        for e_in in graph['ent_inx']:
                            sen_ent_emb.append(ent_emb[int(e_in)])
                        for s_l in ent_link:

                            sen_kgp = [] 
                            if(len(s_l) > 0):
                                for e_i in s_l:
                                    sen_kgp = sen_kgp + kgp_emb[e_i]
                                sen_kgp = list(set(sen_kgp))
                            sen_kgp_emb.append(sen_kgp)
                        
                    for t_in in graph['top_inx']:
                        sen_top_emb.append(top_emb[t_in])


                    
                    graph['ent_emb'] = sen_ent_emb
                    graph['top_emb'] = sen_top_emb
                    graph['kgp'] = sen_kgp_emb

                    self.graph.append(graph)
                except:
                    count = 1


        self.length = len(self.graph)
        print(self.name, self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        return self.graph[index]
