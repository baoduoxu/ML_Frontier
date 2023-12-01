import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

from configs.config import AEConfig

# Encoder
class Encoder(nn.Module):
    """ Vanilla encoder with 3 fc layers 

    Args:
            dIn (int): Dim of input
        dHidden (int): Dim of intermediate FCs.
             dZ (int): Dim of z.
             
    Inputs:
        x (torch.FloatTensor): [N, d_origin]. Input images.
        
    Outputs:
        torch.FloatTensor: [N, dZ]. Latent variable z.
    """
    def __init__(self, dIn: int, dH: [int], dnum: int, dZ: int):
        super().__init__()
        # [fc -> relu -> bn] * dnum -> fc 
        layers = []
        
        layers=[
            nn.Linear(dIn, dH[0]),
            nn.ReLU(),
            nn.BatchNorm1d(dH[0]),

            nn.Linear(dH[0], dZ),
        ]
        # # 添加第一个线性层、LeakyReLU和批量归一化
        # layers.append(nn.Linear(dIn, dH[0]))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(dH[0]))

        # # 添加中间的线性层、LeakyReLU和批量归一化
        # for i in range(0, dnum-1):
        #     layers.append(nn.Linear(dH[i], dH[i+1]))
        #     layers.append(nn.LeakyReLU())
        #     # layers.append(nn.Dropout(0.5))
        #     layers.append(nn.BatchNorm1d(dH[i+1]))

        # # 添加最后一个线性层
        # layers.append(nn.Linear(dH[dnum-1], dZ))

        # 创建Sequential模型
        self._net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self._net(x)

# Decoder
class Decoder(nn.Module):
    """ Vanilla decoder with 3 fc layers 
    Args:
             dZ (int): Dim of z.
        dHidden (int): Dim of intermediate FCs.
             
    Inputs:
        z (torch.FloatTensor): [N, dZ]. Input latent variable z.
        
    Outputs:
        torch.FloatTensor: [N, channel, hw, hw]. xHat, restored x.
    """
    def __init__(self, dZ: int, dH: [int], dnum: int, dOut: int): # 传入的 dH 是 Encoder 的 dH 的 reverse
        super().__init__()
        # [fc -> relu -> bn] * 2 -> fc 
        layers = [
            nn.Linear(dZ, dH[0]),
            nn.ReLU(),
            nn.BatchNorm1d(dH[0]),

            nn.Linear(dH[0], dOut),
            nn.Sigmoid()
        ]
        # # 添加第一个线性层、LeakyReLU和批量归一化
        # layers.append(nn.Linear(dZ, dH[0]))
        # layers.append(nn.LeakyReLU())
        # layers.append(nn.BatchNorm1d(dH[0]))

        # # 添加中间的线性层、LeakyReLU和批量归一化
        # for i in range(0, dnum-1):
        #     layers.append(nn.Linear(dH[i], dH[i+1]))
        #     layers.append(nn.LeakyReLU())
        #     # layers.append(nn.Dropout(0.5))
        #     layers.append(nn.BatchNorm1d(dH[i+1]))

        # # 添加最后一个线性层
        # layers.append(nn.Linear(dH[dnum-1], dOut))
        # layers.append(nn.Sigmoid())
        # 创建Sequential模型
        self._net = nn.Sequential(*layers)
        
    def forward(self, z):
        return self._net(z)