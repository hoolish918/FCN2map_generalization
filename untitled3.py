# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 13:52:35 2019

@author: xiaoke
"""

from skimage import io
import matplotlib
from PIL import Image
import numpy as np
import scipy.misc

j=1
for i in range(1,251):
    imgg1=io.imread(r'E:\地图综合\4城市\四城市\北京\栅格\北京'+str(i)+'.tif')
    imgg1=imgg1+1
    imgg1[imgg1==256]=0
    a1=imgg1
    if max(imgg1.shape)<128:    
        a=np.zeros((128,128))
        a[:imgg1.shape[0],:imgg1.shape[1]]=imgg1
        np.save("E:/地图综合/4城市/数据/data"+str(j)+".npy",a)
        j=j+1
for i in range(1,251):
    imgg1=io.imread(r'E:\地图综合\4城市\四城市\武汉1\栅格\武汉'+str(i)+'.tif')
    imgg1=imgg1+1
    imgg1[imgg1==256]=0
    a1=imgg1
    if max(imgg1.shape)<128:    
        a=np.zeros((128,128))
        a[:imgg1.shape[0],:imgg1.shape[1]]=imgg1
        np.save("E:/地图综合/4城市/数据/data"+str(j)+".npy",a)
        j=j+1
for i in range(1,251):
    imgg1=io.imread(r'E:\地图综合\4城市\四城市\长沙\栅格\长沙'+str(i)+'.tif')
    imgg1=imgg1+1
    imgg1[imgg1==256]=0
    a1=imgg1
    if max(imgg1.shape)<128:    
        a=np.zeros((128,128))
        a[:imgg1.shape[0],:imgg1.shape[1]]=imgg1
        np.save("E:/地图综合/4城市/数据/data"+str(j)+".npy",a)
        j=j+1
    
    
    
imgg1=io.imread(r'E:\济南综合\data\data\data1.png')
l=[]
for i in range(1,652):
    imgg1=io.imread('E:\济南综合\data\data\data'+str(i)+'.png')
    l.append((imgg1.shape)[0])
    l.append((imgg1.shape)[1])
    
j=1
for i in range(1,650):
    imgg2=np.array(Image.open(r'E:\济南综合\栅格\6m\data\data\data'+str(i)+'.png').convert('L'))
    imgg1=np.array(Image.open(r'E:\济南综合\栅格\6m\data\label\label'+str(i)+'.png').convert('L'))
    imgg1=(imgg1-imgg1.min())/(imgg1.max()-imgg1.min())
    imgg2=(imgg2-imgg2.min())/(imgg2.max()-imgg2.min())
    if max(imgg1.shape)<128:    
        a=np.zeros((128,128))
        b=np.zeros((128,128))
        a[:imgg1.shape[0],:imgg1.shape[1]]=imgg1
        b[:imgg1.shape[0],:imgg1.shape[1]]=imgg2
        scipy.misc.imsave('E:/济南综合/栅格/6m/128乘128/label/label'+str(j)+'.png', a)
        scipy.misc.imsave('E:/济南综合/栅格/6m/128乘128/data/data'+str(j)+'.png', b)
        j=j+1
    