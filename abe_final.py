#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 17:11:44 2023

@author: linyuanwang
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from sklearn.model_selection import train_test_split
from model import Unet
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from abe_data import Data,DiceBCELoss

    

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:3" if use_cuda else "cpu")    

curr_folder = os.getcwd()
train_folder = os.path.join(curr_folder,'train_with_cu_lw')
files_ = os.listdir(train_folder)
num_epochs = 20               
model = Unet(1,2)
model.to(device)
loss_fn = DiceBCELoss() # binary class
loss_fn.to(device)


train_params = {
    'batch_size': 8,
    'shuffle': True  
}
valid_params = {
    'batch_size': 8,
    'shuffle': False  
}

optim_params = {
    'lr': 1e-4, 
    'alpha': 0.9,
    'eps': 1e-5,
    'momentum': 0.9
}


optimizer = torch.optim.RMSprop(model.parameters(), **optim_params)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    factor=0.8,  
    min_lr=1e-6  
)

train_ids, val_ids = train_test_split(files_, test_size=0.25, random_state=42)

train_set = Data(train_ids, train_folder)
valid_set = Data(val_ids, train_folder)
train_generator = DataLoader(train_set, **train_params)
valid_generator = DataLoader(valid_set, **valid_params)


best_v_loss = 100_0

pth = os.getcwd()
model_path = os.path.join(pth,'model_new_1')

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    
    print('Epoch {0}'.format(epoch))
    epoch_loss = 0.0
    run_loss = 0.0
    model.train(True)

    for i,x in enumerate(train_generator):

        x = x[0].to(device)
        x = torch.unsqueeze(x,1)           
# augmentation 
        v_img = transform.RandomVerticalFlip(p=1)(x)
        h_img = transform.RandomHorizontalFlip(p=1)(x)           
        r_img = transform.RandomRotation(degrees=(-180,180))(x)
        
        ip = torch.cat((x[0],v_img[0],h_img[0],r_img[0]),0)
        ip = torch.unsqueeze(ip,1)

        op = torch.cat((x[1],v_img[1],h_img[1],r_img[1]),0)
        op = op.long()

        optimizer.zero_grad()
        pred = model.forward(ip)
      
        gg = op.view(-1)
        class_1 = torch.sum(op).item() 
        class_0 = gg.size(dim = 0) - class_1 
        wc = torch.tensor([class_1/class_0,class_0/class_1])
        
        loss = loss_fn(pred,op,wc)
        loss.backward()

        optimizer.step()

        run_loss+=loss.item()
        epoch_loss+=loss.item()

        if (i % 10 == 9):
            
            print("\tBatch Loss for curent for {0} is {1:.5f}".format(i,run_loss/10))
            run_loss = 0.0
        
    avg_e_loss = epoch_loss/(i+1)
    print('The average loss for the epoch is {0}'.format(avg_e_loss))
    train_losses.append(avg_e_loss)
    model.train(False)

    val_loss = 0.0

    for k,val in enumerate(valid_generator):
        
        val = val[0].to(device)
        ip = torch.unsqueeze(torch.unsqueeze(val[0],0),0)
        op = torch.unsqueeze(val[1],0)
        op = op.long()
        pred = model.forward(ip)

        loss = loss_fn(pred,op)

        val_loss+=loss.item()

    avg_val_loss = val_loss/(k+1)
    
    print('The average validation loss for the epoch is {0:.5f}'.format(avg_val_loss))
    val_losses.append(avg_val_loss)
    
    if avg_val_loss < best_v_loss:

        best_v_loss = avg_val_loss
        best_epoch = epoch

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Losses')
plt.show()            


