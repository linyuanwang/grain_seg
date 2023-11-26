#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:15:36 2023

@author: linyuanwang
"""


import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


class Data(Dataset):

    def __init__(self,list_ids,folder):
        self.list_ids = list_ids
        self.folder = folder
    def __getitem__(self, index):
        
        id = self.list_ids[index]
        pth = os.path.join(self.folder,str(id))
        arr = np.load(pth).astype(np.float32)
        x = torch.from_numpy(arr)


        return x
    
    def __len__(self):

        return(len(self.list_ids))
    

class DiceBCELoss(nn.Module):
    
    def __inti__(self,weight = None,size_average = True):
        super(DiceBCELoss,self).__init__()
        
    def forward(self,inputs,targets,wc=torch.tensor([1.0,1.0])):
        
        m = nn.LogSoftmax(dim=1)
        inputs_2 = m(inputs)
        
        BCE_loss = nn.NLLLoss(weight = wc).to('cuda:3')
        
        DICE_BCE = BCE_loss(inputs_2,targets)  
        
        return DICE_BCE
    
    
    
from scipy.spatial.distance import directed_hausdorff

def dice_coefficient(y_true, y_pred, epsilon=1e-6):

    y_true_f = y_true.reshape(y_true.shape[0], -1)
    y_pred_f = y_pred.reshape(y_pred.shape[0], -1)
    intersection = (y_true_f * y_pred_f).sum(axis=1)
    return ((2. * intersection + epsilon) / (y_true_f.sum(axis=1) + y_pred_f.sum(axis=1) + epsilon)).mean()

def hausdorff_distance(y_true, y_pred):
    distances = []
    for i in range(y_true.shape[0]):
        distances.append(max(directed_hausdorff(y_true[i], y_pred[i])[0], directed_hausdorff(y_pred[i], y_true[i])[0]))
    return np.mean(distances)

def hausdorff_distance(y_true, y_pred):
    distances = []
    for i in range(y_true.shape[0]):
        true_2d = y_true[i, 0]  
        pred_2d = y_pred[i, 0]
        distances.append(max(directed_hausdorff(true_2d, pred_2d)[0], directed_hausdorff(pred_2d, true_2d)[0]))
    return np.mean(distances)
