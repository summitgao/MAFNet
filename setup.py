# coding=utf-8
# Version:python 3.7

import argparse
import os
import torch.optim as optim
import torch
from Net.model_mafnet import MAFNet

def option():
    parse = argparse.ArgumentParser()
    parse.add_argument('-p', '--path')
    parse.add_argument('-i', '--input', default=31, type=int)
    parse.add_argument('-c', '--cuda', default=1)
    parse.add_argument('-e', '--epochs', default=300, type=int)
    parse.add_argument('-l', '--lr', default=0.0002)
    parse.add_argument('-mp', '--mat', default='./data/Mat/')
    parse.add_argument('-d', '--dataset', default='./data/dataset_p/')
    parse.add_argument('-dn', '--dataset_name', default='icvl')
    parse.add_argument('-bs', '--batch_size', default=(10, 3))
    parse.add_argument('-m', '--model', default='MAFNet', type=str)
    parse.add_argument('-b', '--blind', default='gauss', type=str)
    parse.add_argument('-n', '--noise', default=70, type=int)
    parse.add_argument('-pm', '--pretrain_modle', default='./model/pamodelcdm.pkl', type=str)
    arg = parse.parse_args()
    return arg


def getmodel(channel, middle, model_name = 'MAFNet'):
    model = MAFNet(channel, channel, middle)

    return model


def load_model(model, path, mode='train'):

    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 50, 100, 150], gamma=0.5, last_epoch=-1)

    if os.path.exists(path):
        dic = torch.load(path)
        netp, optp, opts = dic['net'], dic['optimizer'], dic['scheduler']
        model.load_state_dict(netp)
        print('model load successfully!')
        if mode == 'train':
            optimizer.load_state_dict(optp)
            scheduler.load_state_dict(opts)
            return model, optimizer, scheduler
    else:
        print('model does not exist at {}'.format(path))
        return model, optimizer, scheduler
    return model
