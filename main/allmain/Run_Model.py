# GateRGCN
# server
# Allsides dataset
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
in_channels = 768

EPOCH = 150
OUT_CHANNELS = 512
BATCH_SIZE = 16
DROPOUT = 0.6
WEIGHT_DECAY = 1e-4
NUM_HEADS = 32  # KGP HEAD ##########
LEARNING_RATE = 1e-3
PATH = '../../all/'
EARLYSTOP = 40
LR_SCHEDUALER = 20


parser = argparse.ArgumentParser()
parser.add_argument("--type", type=int, default=0)
parser.add_argument("--lid", type=int, default=0)

args = parser.parse_args()


#####################################################################################################################
OUTTYPE = args.type# 0: PA; 1: CA; 2: GA;
runtimes = 5
#####################################################################################################################

for run in range(0, runtimes):
    valname = 'main' + str(OUTTYPE) + '_log_id_' + str(args.lid)
    logpath = 'log/' + valname + '/'
    txtpath = 'txt/' + valname
    #####################################################################################################################

    resulttxt = open(txtpath + '.txt', 'a')
    resulttxt.write('***********' + valname + '***********\n')
    resulttxt.write('Out Type = '+str(OUTTYPE) + '\n')

    resulttxt.write('\nTraining Begin\n')
    resulttxt.close()
    accuracy = []
    f1 = []
    for FOLDID in range(1, 4):
        print(FOLDID)
        model = KSD(in_channels=in_channels,out_channels = OUT_CHANNELS, dropout = DROPOUT,
            num_heads = NUM_HEADS,Weight_decay= WEIGHT_DECAY,lr = LEARNING_RATE, lr_s=LR_SCHEDUALER, outtype = OUTTYPE)
        train_dataset = KSDDataset(
            name='train', foldid=FOLDID, path=PATH)
        dev_dataset = KSDDataset(name='dev', foldid=FOLDID,
                                path=PATH)

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
            trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16, max_epochs=EPOCH, callbacks=[early_stop_callback], logger=comet_logger)
        elif EARLYSTOP == 0:
            trainer = pl.Trainer(gpus=1, num_nodes=1, precision=16,
                                max_epochs=EPOCH, logger=comet_logger)
        ########################################
        trainer.fit(model, train_loader, dev_loader)
        accuracy.append(format(model.globalACC, '.4f'))
        f1.append(format(model.globalF1, '.4f'))
        resulttxt = open(txtpath + '.txt', 'a')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        resulttxt.write('Time:' + current_time + ' FoldId= '+str(FOLDID)+' ')
        resulttxt.write('ACC= '+str(model.globalACC)+' ')
        resulttxt.write('F1= '+str(model.globalF1)+'\n')
        resulttxt.close()

    sumacc = 0
    sumf1 = 0
    for i in range(0,3):
        sumacc += float(accuracy[i])
        sumf1 += float(f1[i])
    sumacc = format(sumacc/3, '.4f')
    sumf1 = format(sumf1/3,'.4f')

    resulttxt = open(txtpath + '.txt', 'a')
    resulttxt.write('{:<8s}{:^7s}{:^7s}{:^7s}{:^7s}'.format('Fold', str(1), str(2), str(3), 'sum'))
    resulttxt.write('\n{:<8s} {} {} {}\n'.format('ACC', accuracy[0], accuracy[1], accuracy[2]))
    resulttxt.write('\n{:<8s} {} {} {}\n{} {} {}'.format('F1', f1[0], f1[1], f1[2],'Sumacc, Sumf1 = ' ,sumacc, sumf1))
    resulttxt.close()
