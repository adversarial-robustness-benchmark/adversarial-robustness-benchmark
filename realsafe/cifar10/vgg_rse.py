from __future__ import absolute_import

import os
import tensorflow as tf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Function
import numpy as np

from realsafe.pytorch_model import pytorch_classifier_differentiable
from six.moves import urllib
from realsafe.utils import get_model_path, show_progress

@pytorch_classifier_differentiable(x_min=0.0, x_max=1.0, x_shape=(32,32,3), x_dtype=tf.float32, y_dtype=tf.int32,
                                   n_class=10)
class VGG_RSE(torch.nn.Module):
    def __init__(self, use_cuda= True):
        torch.nn.Module.__init__(self)
        self.model = VGG16().cuda()
        self.num_ensemble = 50
            
    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        x_ex = x.unsqueeze(1).repeat(1, self.num_ensemble, 1, 1, 1)\
                .view(-1, x.shape[1], x.shape[2], x.shape[3])
        labels = self.model(x_ex.cuda())
        labels = labels.view(x.shape[0], self.num_ensemble, -1).mean(dim = 1)
        return labels.cpu()
    
    def load(self, **kwargs):
        model_path = get_model_path('cifar10_vgg_rse.pth')
        
        try:
            checkpoint = torch.load(model_path)
            from collections import OrderedDict
            state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  
                state_dict[name] = v
            self.model.load_state_dict(state_dict)
            self.eval()
        except FileNotFoundError:
            raise IOError
    
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.noise_init = 0.2
        self.noise_inner = 0.1
        self.img_width = 32
        self.layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(self.layers)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                if i == 0:
                    noise_layer = Noise(self.noise_init)
                else:
                    noise_layer = Noise(self.noise_inner)
                layers += [noise_layer,
                           nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)
        
class Noise(nn.Module):
    def __init__(self, std):
        super(Noise, self).__init__()
        self.std = std
        self.buffer = None

    def forward(self, x):
        if self.std > 0:
            if self.buffer is None:
                self.buffer = torch.Tensor(x.size()).normal_(0, self.std).cuda()
            else:
                self.buffer.data.resize_(x.size()).normal_(0, self.std)
            return x + self.buffer
        return x
