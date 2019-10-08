# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:35:38 2019

@author: xiaoke
"""
from datetime import datetime
from skimage import io
from PIL import Image
import torchvision.transforms as tfs
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

from fcn8 import fcn

lo=[]
accury=[]
meaniu=[]
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],[0,128,128]]
cm = np.array(colormap).astype('uint8')
class BuildingsDataset(Dataset):
    '''
    voc dataset
    '''
    def __init__(self, crop_size, transforms):
        self.crop_size = crop_size
        self.transforms = transforms
        self.root=''
        
        
    def __getitem__(self, idx):
        img  = np.array(Image.open("E:/济南综合/data/128乘128/data/data"+str(idx+1)+".png").convert('L'), 'f')
        label=np.array(Image.open("E:/济南综合/data/128乘128/label/label"+str(idx+1)+".png").convert('L'), 'f')
        a=self.transforms(img,self.crop_size).float()
        b=self.transforms(label,self.crop_size).float()
        return a,b
    
    def __len__(self):
        return 606
    
    
def img_transforms(img,  crop_size):
    im=(img==(img).max()).astype(float)
    im = torch.from_numpy(im)
    im=im.unsqueeze(0)
    #im=tfs.Resize(crop_size)(im)
    #im=tfs.ToTensor()(im)
    
    #a[a>0]=1
    return im

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

#imgg=img_transforms(imgg,(128,128)).cuda()
input_shape = (128, 128)
print('1')
voc_train = BuildingsDataset(input_shape, img_transforms)
print('2')
train_data = Data.DataLoader(voc_train, 20, shuffle=True)
print('3')
net2=fcn().cuda()
print('4')
criterion = nn.NLLLoss()
log_softmax=nn.LogSoftmax(dim=1)

def train(net,train_data,epoch=150):
    optimizer=torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for e in range(epoch):
        
        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0
    
        prev_time = datetime.now()
        net = net.train()
        k=0
        for data, label in train_data:
            
            k=k+1
            im = (data.cuda())
            label = (label.squeeze(1).long().cuda())
            # forward
            out = net(im)
            out = log_softmax(out) # (b, n, h, w)
            loss = criterion(out, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(loss)
            label_pred = out.max(dim=1)[1].data.cpu().numpy()
            label_true = label.data.cpu().numpy()
            for lbt, lbp in zip(label_true, label_pred):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, 6)
                train_acc += acc
                train_acc_cls += acc_cls
                train_mean_iu += mean_iu
                train_fwavacc += fwavacc
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        epoch_str = ('Epoch: {}, Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f}, '.format(
                    e, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train)))
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(time_str)
        print(epoch_str + time_str + ' lr: {}'.format(0.001))
        accury.append(train_acc / len(voc_train))
        meaniu.append(train_mean_iu / len(voc_train))
        lo.append(train_loss / len(train_data))
        '''
        imgg1=np.load('E:/地图综合/4城市/数据/data501.npy')
        
        imgg=img_transforms(imgg1,(128,128))[0].cuda()
        
        plt.imshow(cm[net2(imgg.unsqueeze(0)).cpu().detach()[0].max(0)[1]])
        plt.pause(1)

        imgg=img_transforms(imgg1,(128,128))[1].cuda()
        plt.imshow(cm[imgg.cpu().squeeze(0).long()])
        plt.pause(1)
        #plt.imshow(net(imgg.unsqueeze(0)).cpu()[0][0])
        '''


train(net2,train_data)