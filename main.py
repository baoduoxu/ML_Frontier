import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

import Database.database as database

from utils import *
from Database.utils import split_train_valid_test
from configs.config import (
    AEConfig,
    ADNI_config
)
from models.AE import Encoder, Decoder
if __name__ == "__main__":
    db_cfg = ADNI_config  # 在主函数里换数据集只改这一个参数
    db = database(db_cfg.name, db_cfg.name)
    for randomstate in db_cfg.random_state:
        setup_seed(randomstate)
        data, label, labelmap = db.rawdatatonumpy()
        train_data, train_label, valid_data, valid_label, test_data, test_label = split_train_valid_test(
            data, label, randomstate=randomstate)
        train_data = torch.from_numpy(train_data).float()
        train_label = torch.from_numpy(train_label).long()
        valid_data = torch.from_numpy(valid_data).float()
        valid_label = torch.from_numpy(valid_label).long()
        test_data = torch.from_numpy(test_data).float()
        test_label = torch.from_numpy(test_label).long()

        trainDataset = torch.utils.data.TensorDataset(train_data, train_label)
        validDataset = torch.utils.data.TensorDataset(valid_data, valid_label)
        testDataset = torch.utils.data.TensorDataset(test_data, test_label)

        # trainLoader = torch.utils.data.DataLoader(
        #     trainDataset, batch_size=db_cfg.batch_size, shuffle=True)
        # validLoader = torch.utils.data.DataLoader(
        #     validDataset, batch_size=db_cfg.batch_size, shuffle=False)
        # testLoader = torch.utils.data.DataLoader(
        #     testDataset, batch_size=db_cfg.batch_size, shuffle=False)

        encoder = Encoder(train_data.shape[1], db_cfg.encoder_layers,
                          db_cfg.num_layers, db_cfg.latent_dim)
        decoder = Decoder(db_cfg.latent_dim, db_cfg.decoder_layers,
                          db_cfg.num_layers, train_data.shape[1])

        encoder = encoder.to(db_cfg.device)
        decoder = decoder.to(db_cfg.device)
        trainDataset = trainDataset.to(db_cfg.device)
        testDataset = testDataset.to(db_cfg.device)
        
        optimizer = torch.optim.AdamW(list(encoder.parameters(
        )) + list(decoder.parameters()), db_cfg.lr, weight_decay=0.9)
        
        train(encoder, decoder, trainDataset, testDataset,
              db_cfg.loss_function, optimizer, db_cfg.num_epochs)
