from datasets import load_metric
from typing import List, Tuple
from configs.config import *
import numpy as np
import torch
import evaluate
import peft
import transformers
import datasets
import matplotlib.pyplot as plt
import random
import torch.nn as nn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    PromptEncoderConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from Database.utils import split_train_valid_test
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def setup_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(encoder: nn.Module, decoder: nn.Module, trainDataset, testDataset, lossFn, optimizer, step = 1000):
    ############### FOR VISUALIZATION, DO NOT CHANGE ###############
    def visualize(encoder: nn.Module, decoder: nn.Module, testInput: torch.Tensor,losses: [float]):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(losses)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Over Iterations")
        plt.grid(True)
        plt.show()
        # display.clear_output(wait=True)
        # xHat = decoder(encoder(testImages))
        # results = torch.cat([testImages, xHat], 0)
        # results = torchvision.utils.make_grid(results, nrow=4)
        # fig, ax2 = plt.subplots(1, 2, figsize=(15,5))
        # # results = results.permute(1, 2, 0)
        # # ax1.imshow(results)
        # # ax1.axis("off")
        # ax2.plot(losses)
        # ax2.grid(True)
        # plt.show()
        # # return results
    # # create Dataset loader
    # def visualize(encoder: nn.Module, decoder: nn.Module, testInput: torch.Tensor, losses: [float]):
    #     # display.clear_output(wait=True)
    #     print(testInput.shape)
    #     testInput.unsqueeze_(0)
    #     xHat = decoder(encoder(testInput))
    #     print(xHat.shape)
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
    
    #     # 绘制xHat和testInput
    #     # ax1.plot(xHat, label='xHat')
    #     # ax1.plot(testInput, label='testInput')
    #     # # ax1.legend()
    #     # ax1.grid(True)
        
    #     # 绘制losses
    #     ax2.plot(losses, label='Loss')
    #     # ax2.legend()
    #     ax2.grid(True)
        
    #     plt.show()
    loader = torch.utils.data.DataLoader(trainDataset, batch_size=2, shuffle=True, num_workers=2)
    # create Optimizer
    
    
    # test images for visualization
    # testImgs = list()
    random_index = random.randint(0, len(testDataset) - 1)
    sample_data, _ = testDataset[random_index]
    # pick = random.sample(range(len(testDataset)), 4)
    # for i in pick:
    #     testImgs.append(testDataset[i][0])
    # testImgs = torch.stack(testImgs, 0)
    
    # loss logging
    losses = list()
    iterator = iter(loader)
    
    # all results
    # allImages = list()
    
    for i in range(step):
        try:
            x, _ = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, _ = next(iterator)
        z = encoder(x)
        # decoding
        xHat = decoder(z)
        # compute reconstruction loss
        loss = lossFn(xHat, x)
        # log losses
        losses.append(loss.item())

        # gradient backward and step
        optimizer.zero_grad()
        # print(f'before loss.requires_grad:{loss.requires_grad}')
        # if loss.requires_grad==False:
        #     loss.requires_grad=True
        # print(f'after loss.requires_grad:{loss.requires_grad}')
        loss.backward()
        optimizer.step()

        # visualize
        if i % 100 == 0:
            torch.set_grad_enabled(False)
            encoder.eval()
            decoder.eval()
            print(f"Step {i} loss: {loss.item()}")
            # allImages.append()
            # visualize(encoder, decoder,sample_data, losses)
            encoder.train()
            decoder.train()
            torch.set_grad_enabled(True)
    # final
    print(f"Train on {step} steps finished.")
    print(f"Final loss: {loss.item()}")
    # return allImages