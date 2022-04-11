# server
# GateRGCN
# Allsides dataset
#
import torch
from torch import nn
import pytorch_lightning as pl
import pytorch_lightning as pl
from Tools import f1_score, GatedRGCN, get_metrics, attention

class KSD(pl.LightningModule):
    def __init__(self, in_channels, out_channels, dropout, num_heads, Weight_decay, lr, lr_s, ent_dim, top_dim, outtype):
        super().__init__()
        ###
        self.globalACC = 0
        self.globalF1 = 0
        ###
        self.outtype = outtype
        ###
        self.lr = lr
        self.lr_s = lr_s #lr scheduler
        self.weight_decay = Weight_decay
        ###
        self.in_channel = in_channels
        self.out_channels = out_channels
        ###graph convelotion
        self.RGCN1 = GatedRGCN(out_channels, out_channels, 6)
        self.RGCN2 = GatedRGCN(out_channels, out_channels, 6)
        ###graph convelotion
        ### kgp
        self.SenAttention = attention(in_channels, in_channels)
        self.multihead_attn = nn.MultiheadAttention(in_channels, num_heads)
        ###sentence, entity, topic mlp
        self.senLinear = nn.Linear(in_channels, out_channels)
        torch.nn.init.kaiming_uniform(
            self.senLinear.weight, nonlinearity='leaky_relu')
        self.entLinear = nn.Linear(ent_dim, out_channels)
        torch.nn.init.kaiming_uniform(
            self.entLinear.weight, nonlinearity='leaky_relu')
        self.topLinear = nn.Linear(top_dim, out_channels)
        torch.nn.init.kaiming_uniform(
            self.topLinear.weight, nonlinearity='leaky_relu')
        ###sentiment, quotation, tense parameters
        self.emoWeight = nn.Parameter(torch.randn(2, out_channels))
        self.quoWeight = nn.Parameter(torch.randn(2, out_channels))
        self.tenWeight = nn.Parameter(torch.randn(17, out_channels))
        ###
        self.dropout = nn.Dropout(dropout)
        self.CELoss = nn.CrossEntropyLoss()
        self.activation = nn.LeakyReLU()
        self.output = nn.Linear(out_channels, 2)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), self.lr, weight_decay = self.weight_decay)
        if self.lr_s > 0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                mode='min',
                                                                factor=0.1,
                                                                patience=self.lr_s,
                                                                min_lr=1e-6,
                                                                verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            return optimizer

    def training_step(self, train_batch, batch_index):
        # Initialize matric
        loss = 0
        totalpred = []
        totallabel = []
        # Learning loop
        for i in range(0, len(train_batch)):
            # edge
            edge_index = train_batch[i]['edge_index'] #torch.long
            edge_type = train_batch[i]['edge_type'] #torch.long
            # embedding
            art_emb = torch.stack(train_batch[i]['art_emb'])
            sen_num = len(art_emb)
            ent_emb = train_batch[i]['ent_emb']
            ent_num = len(ent_emb)
            if(ent_num > 0):
                # if entity exist
                ent_emb = torch.stack(ent_emb)
                # learnable
                ent_emb = self.dropout(self.activation(
                    self.entLinear(ent_emb))) 
            top_emb = torch.stack(train_batch[i]['top_emb'])
            top_emb = self.dropout(self.activation(
                self.topLinear(top_emb))) 
            # Initialize sentiment, quotation and tense
            emo_Emb = self.emoWeight
            quo_Emb = self.quoWeight
            ten_Emb = self.tenWeight

            # knowledge path
            if ent_num > 0:
                kgp_emb = train_batch[i]['kgp_emb']
                sen2ent = train_batch[i]['sen2ent']
                kgp_att_emb = [] # attention kgp
                for s_i in range(sen_num):
                    sen_kgp = []
                    for j in range(len(sen2ent[0])):
                        if sen2ent[0][j] == s_i:
                            sen_kgp = sen_kgp + kgp_emb[sen2ent[1][j]]
                    if len(sen_kgp) > 0:
                        kgp_att_emb.append(self.SenAttention(art_emb[s_i], sen_kgp))
                    if len(kgp_att_emb) > 0:
                        mutiheadinput = torch.cat(
                            (art_emb, torch.stack(kgp_att_emb)), dim=0).unsqueeze(1)   
                    else:
                        mutiheadinput = art_emb.unsqueeze(1)
            else:
                mutiheadinput = art_emb.unsqueeze(1)
            newsentence = self.dropout(self.activation((self.multihead_attn(
                mutiheadinput, mutiheadinput, mutiheadinput)[0])))  # fsb: ACTIVATION, DROPOUT ADDED
            
            art_emb = newsentence[0: sen_num].squeeze(1)
            art_emb = self.dropout(self.activation(
                self.senLinear(art_emb)))  # fsb: DROPOUT ADDED

            if(ent_num > 0):
                newnodeembedding = torch.cat(
                    (art_emb, ent_emb, top_emb, emo_Emb, quo_Emb, ten_Emb), dim=0)
            else:
                newnodeembedding = torch.cat(
                    (art_emb, top_emb, emo_Emb, quo_Emb, ten_Emb), dim=0)

        # gragh
            n_1 = self.RGCN1(newnodeembedding, edge_index, edge_type)
            a_1 = self.dropout(self.activation(n_1))
            n_2 = self.RGCN2(a_1, edge_index, edge_type)
            a_2 = self.dropout(self.activation(n_2))
            if self.outtype == 0:
                output_vec = a_2[0: sen_num]
            elif self.outtype == 1:
                output_vec = a_2[sen_num: ]
            elif self.outtype == 2:
                output_vec = a_2
        # matric
            pred = self.output(torch.mean(output_vec, dim=0))
            label = int(train_batch[i]['label'])
            label = torch.tensor([label]).long().cuda()
        # Calculate matric
            totallabel.append(label)
            totalpred.append(pred)
            loss = loss + self.CELoss(pred.unsqueeze(0), label)

        totalpred = torch.stack(totalpred)
        totallabel = torch.LongTensor(totallabel)
        loss = loss / len(train_batch)
        accuracy = get_metrics(totalpred, totallabel)
        F1 = f1_score(totalpred, totallabel)
        self.log('train_loss', loss)
        self.log('train_ACC', accuracy)
        self.log('train_F1', F1)
        return loss

    def validation_step(self, val_batch, batch_index):
        # Initialize matric
        loss = 0
        totalpred = []
        totallabel = []
        # Learning loop
        for i in range(0, len(val_batch)):

            edge_index = val_batch[i]['edge_index']
            edge_type = val_batch[i]['edge_type']

            art_emb = torch.stack(val_batch[i]['art_emb'])
            sen_num = len(art_emb)
            ent_emb = val_batch[i]['ent_emb']
            ent_num = len(ent_emb)
            if(ent_num > 0):

                ent_emb = torch.stack(ent_emb)
                # learnable
                ent_emb = self.dropout(self.activation(
                    self.entLinear(ent_emb))) 
            top_emb = torch.stack(val_batch[i]['top_emb'])
            top_emb = self.dropout(self.activation(
                self.topLinear(top_emb))) 
            # Initialize sentiment, quotation and tense
            emo_Emb = self.emoWeight
            quo_Emb = self.quoWeight
            ten_Emb = self.tenWeight


            if ent_num > 0:
                kgp_emb = val_batch[i]['kgp_emb']
                sen2ent = val_batch[i]['sen2ent']
                kgp_att_emb = [] 
                for s_i in range(sen_num):
                    sen_kgp = []
                    for j in range(len(sen2ent[0])):
                        if sen2ent[0][j] == s_i:
                            sen_kgp = sen_kgp + kgp_emb[sen2ent[1][j]]
                    if len(sen_kgp) > 0:
                        kgp_att_emb.append(self.SenAttention(art_emb[s_i], sen_kgp))
                    if len(kgp_att_emb) > 0:
                        mutiheadinput = torch.cat(
                            (art_emb, torch.stack(kgp_att_emb)), dim=0).unsqueeze(1)   
                    else:
                        mutiheadinput = art_emb.unsqueeze(1)
            else:
                mutiheadinput = art_emb.unsqueeze(1)
            newsentence = self.dropout(self.activation((self.multihead_attn(
                mutiheadinput, mutiheadinput, mutiheadinput)[0])))  # fsb: ACTIVATION, DROPOUT ADDED
            

            art_emb = newsentence[0: sen_num].squeeze(1)
            art_emb = self.dropout(self.activation(
                self.senLinear(art_emb)))  # fsb: DROPOUT ADDED

            if(ent_num > 0):
                newnodeembedding = torch.cat(
                    (art_emb, ent_emb, top_emb, emo_Emb, quo_Emb, ten_Emb), dim=0)
            else:
                newnodeembedding = torch.cat(
                    (art_emb, top_emb, emo_Emb, quo_Emb, ten_Emb), dim=0)

            n_1 = self.RGCN1(newnodeembedding, edge_index, edge_type)
            a_1 = self.dropout(self.activation(n_1))
            n_2 = self.RGCN2(a_1, edge_index, edge_type)
            a_2 = self.dropout(self.activation(n_2))
            if self.outtype == 0:
                output_vec = a_2[0: sen_num]
            elif self.outtype == 1:
                output_vec = a_2[sen_num: ]
            elif self.outtype == 2:
                output_vec = a_2
        # matric
            pred = self.output(torch.mean(output_vec, dim=0))
            label = int(val_batch[i]['label'])
            label = torch.tensor([label]).long().cuda()
        # Calculate matric
            totallabel.append(label)
            totalpred.append(pred)
            loss = loss + self.CELoss(pred.unsqueeze(0), label)

        totalpred = torch.stack(totalpred)
        totallabel = torch.LongTensor(totallabel)
        loss = loss / len(val_batch)
        accuracy = get_metrics(totalpred, totallabel)
        F1 = f1_score(totalpred, totallabel)
        if(accuracy > self.globalACC):
            self.globalACC = accuracy
            self.globalF1 = F1

        self.log('val_loss', loss)
        self.log('val_ACC', accuracy)
        self.log('val_F1', F1)
