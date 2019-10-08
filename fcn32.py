# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 16:05:09 2019

@author: xiaoke
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
# 数据库模块
import matplotlib.pyplot as plt
#from tensorboardX import SummaryWriter
import torch.nn.functional as F

class fcn(nn.Module):
    def __init__(self):
        super(fcn, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1) #s1
        self.downconv1 = nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(24, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) #s2
        self.downconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        
        self.upconv1=nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.upconv2=nn.ConvTranspose2d(64, 24, kernel_size=4, stride=2, padding=1)
        self.conv13 = nn.Conv2d(24, 12, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(12, 2, kernel_size=3, stride=1, padding=1)
        
        
        self.upconv1.weight.data = bilinear_kernel(128,64, 4) # 使用双线性 kernel      
        
        self.upconv2.weight.data = bilinear_kernel(64, 24, 4) # 使用双线性 kernel
        

        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x1 = F.relu(x)
        x = self.downconv1(x1)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x2 = F.relu(x)
        x = self.downconv2(x2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.upconv1(x)
        #x = torch.cat((x,x2),1)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.upconv2(x)
        #x = torch.cat((x,x1),1)
        x = self.conv13(x)
        x = F.relu(x)
        x = self.conv14(x)
        s = F.relu(x)
        
        return s
    
def bilinear_kernel(in_channels, out_channels, kernel_size):
    '''
    return a bilinear filter tensor
    '''
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return torch.from_numpy(weight)