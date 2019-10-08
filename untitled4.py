# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:46:09 2019

@author: xiaoke
"""
idx=idx-2
img=np.array(Image.open("E:/济南综合/data/128乘128/data/data"+str(idx+1)+".png").convert('L'), 'f')
label=np.array(Image.open("E:/济南综合/data/128乘128/label/label"+str(idx+1)+".png").convert('L'), 'f')
plt.imshow(img)
fig=plt.figure()
plt.imshow(label)
fig1=plt.figure()
plt.imshow(net2(img_transforms(img,(128,128)).unsqueeze(0).cuda().float()).cpu().detach()[0].max(0)[1])

