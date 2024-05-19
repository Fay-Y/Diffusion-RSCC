import cv2
import numpy as np
import torch
import torchvision
#from improved_diffusion import  dist_util

import torch.nn as nn
import torch.nn.functional as F
import math
from .dist_util import *


class ResNet_Encoder(nn.Module):
  
    def __init__(self, NetType = 'resnet',encoded_image_size=14):
        super(ResNet_Encoder, self).__init__()
        # self.NetType = NetType
        self.enc_image_size = encoded_image_size
        self.mean = torch.tensor(np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)).to(dev())
        self.std = torch.tensor(np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)).to(dev())
        cnn = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        layers = [
                cnn.conv1,
                cnn.bn1,
                cnn.relu,
                cnn.maxpool,
            ]

            # 使用前model_stage个卷积块（最大为4）提取特征
        model_stage = 3
        for i in range(model_stage):
            name = 'layer%d' % (i + 1)
            layers.append(getattr(cnn, name))
        self.net = nn.Sequential(*layers)
        
        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        # self.linear = nn.Linear(1024,768)
        
        self.fine_tune()

    def forward(self, img):
        # print('img.size()',img.size())
        img = (img / 255.0 - self.mean) / self.std
        
        out = self.net(img.float())  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # [batch_size, 2048/512, 8, 8] -> [batch_size, 2048/512, 14, 14]
        # print('img shape after resnet:',out.size())
        bs,C,H,W = out.shape

        out = out.permute(0, 2, 3, 1).view(bs, H*W, C)
        # out = self.linear(out)
        # print('img shape after  reshape:',out.size())
        return out
    

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.net.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.net.children())[5:]:  #
            for p in c.parameters():
                p.requires_grad = fine_tune

 


