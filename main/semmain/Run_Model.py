# GateRGCN
# server
# Allsides dataset
from typing_extensions import runtime
from numpy import where
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from KSD_GatedRGCN import KSD
from KSD_Dataset import KSDDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from Tools import pad_collate
from datetime import datetime
import argparse
# Defalt
in_channels = 768  # aticle and topic embedding(roberata)
EPOCH = 150
OUT_CHANNELS = 512
BATCH_SIZE = 16
DROPOUT = 0.6
WEIGHT_DECAY = 1e-4
NUM_HEADS = 8  # KGP HEAD ##########
LEARNING_RATE = 1e-3
PATH = 206
EARLYSTOP = 40
LR_SCHEDUALER = 20
ENT_DIM = 768
TOP_DIM = 768

OUT_TYPE = 0
PATH = '../../sem/Train/'
runtimes = 5

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=0)
parser.add_argument("--lid", type=int, default=0)

args = parser.parse_args()

for run in range(0, runtimes):
    #####################################################################################################################
    OUT_TYPE = args.type
    #
    valname = 'OutType' + str(OUT_TYPE) + "_log_id_" + str(args.lid)
    logpath = 'log/' + valname + '/'
    txtpath = 'txt/' + valname
    #####################################################################################################################

    resulttxt = open(txtpath + '.txt', 'a')
    resulttxt.write('***********' + valname + '***********\n')
    resulttxt.write('Out Type='+str(OUT_TYPE) + '\n')

    resulttxt.write('\nTraining Begin\n')
    resulttxt.close()
    accuracy = 0
    f1 = 0
    for FOLDID in range(0, 10):
        print(FOLDID)
        model = KSD(in_channels=in_channels, out_channels=OUT_CHANNELS, dropout=DROPOUT,
                    num_heads=NUM_HEADS, Weight_decay=WEIGHT_DECAY, lr=LEARNING_RATE, lr_s=LR_SCHEDUALER, ent_dim=ENT_DIM, top_dim=TOP_DIM, outtype=OUT_TYPE)
        train_dataset = KSDDataset(
            name='train', foldid=FOLDID, path=PATH)
        dev_dataset = KSDDataset(
            name='dev', foldid=FOLDID, path=PATH)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, collate_fn=pad_collate)
        dev_loader = DataLoader(dev_dataset, batch_size=len(
            dev_dataset), collate_fn=pad_collate)
        ####################################
        early_stop_callback = EarlyStopping(
            monitor="val_ACC", min_delta=0.00, patience=EARLYSTOP, verbose=False, mode="max")
        comet_logger = pl_loggers.TensorBoardLogger(
            save_dir=logpath)
        if EARLYSTOP > 0:
            trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, max_epochs=EPOCH, callbacks=[
                                 early_stop_callback], logger=comet_logger)
        elif EARLYSTOP == 0:
            trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16,
                                 max_epochs=EPOCH, logger=comet_logger)
        ########################################
        trainer.fit(model, train_loader, dev_loader)
        accuracy = accuracy + model.globalACC
        f1 = f1 + model.globalF1
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        resulttxt = open(txtpath + '.txt', 'a')
        resulttxt.write('Time:' + current_time + '\nFoldId= '+str(FOLDID)+' ')
        resulttxt.write('ACC='+format(model.globalACC, '.4f')+' ')
        resulttxt.write('F1= '+format(model.globalF1, '.4f')+'\n')
        resulttxt.close()
    resulttxt = open(txtpath + '.txt', 'a')
    resulttxt.write('ACC='+format(accuracy/10, '.4f')+' ')
    resulttxt.write('F1= '+format(f1/10, '.4f')+'\n')
    resulttxt.close()
