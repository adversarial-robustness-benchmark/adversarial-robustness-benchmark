from __future__ import absolute_import

import os
import tensorflow as tf
import torch.nn as nn
import torch.autograd
import numpy as np
import scipy.io as sio
from realsafe.pytorch_model import pytorch_classifier_differentiable
from six.moves import urllib
from realsafe.utils import get_model_path, show_progress
import torch.nn.functional as F

@pytorch_classifier_differentiable(x_min=0.0, x_max=255.0, x_shape=(32,32,3), x_dtype=tf.float32, y_dtype=tf.int32,
                                   n_class=10)
class ConvNet_DeepDefense(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = ConvNet().cuda()
        self.loadMat()
    
    def forward(self, x):
        x = x - self.mean
        x = x.transpose(1, 3).reshape((x.shape[0], -1))
        x = x - x.mean(dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = x * 53.0992088 / std.clamp(min=40).reshape((-1, 1))
        input_var =torch.mm(x, self.trans)
        input_var = input_var.reshape((input_var.shape[0], 3, 32, 32))
        
        labels = self.model(input_var.cuda())
        return labels.cpu()
    
    def loadMat(self):
        stats_path = get_model_path('stats.mat')
        
        self.stats = sio.loadmat(stats_path)
        self.mean = torch.from_numpy(self.stats['dataMean'][np.newaxis])
        self.trans = torch.from_numpy(self.stats['Trans'].T)
        
    def load(self, **kwargs):
        model_path = get_model_path('cifar10-convnet-15742544.mat')

        mcn = sio.loadmat(model_path)
        mcn_weights = dict()
        mcn_weights['conv1.weights'] = mcn['net'][0][0][0][0][0][0][0][1][0][0].transpose()
        mcn_weights['conv1.bias'] = mcn['net'][0][0][0][0][0][0][0][1][0][1].flatten()
        mcn_weights['conv2.weights'] = mcn['net'][0][0][0][0][3][0][0][1][0][0].transpose()
        mcn_weights['conv2.bias'] = mcn['net'][0][0][0][0][3][0][0][1][0][1].flatten()
        mcn_weights['conv3.weights'] = mcn['net'][0][0][0][0][6][0][0][1][0][0].transpose()
        mcn_weights['conv3.bias'] = mcn['net'][0][0][0][0][6][0][0][1][0][1].flatten()
        mcn_weights['conv4.weights'] = mcn['net'][0][0][0][0][9][0][0][1][0][0].transpose()
        mcn_weights['conv4.bias'] = mcn['net'][0][0][0][0][9][0][0][1][0][1].flatten()
        mcn_weights['conv5.weights'] = mcn['net'][0][0][0][0][11][0][0][1][0][0].transpose()
        mcn_weights['conv5.bias'] = mcn['net'][0][0][0][0][11][0][0][1][0][1].flatten()
        
        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            t = self.model.__getattr__(k)
            assert t.weight.data.size() == mcn_weights['%s.weights' % k].shape
            t.weight.data[:] = torch.from_numpy(mcn_weights['%s.weights' % k])
            assert t.bias.data.size() == mcn_weights['%s.bias' % k].shape
            t.bias.data[:] = torch.from_numpy(mcn_weights['%s.bias' % k])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4)
        self.conv5 = nn.Conv2d(64, 10, 1)
        for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            w = self.__getattr__(k)
            torch.nn.init.kaiming_normal_(w.weight.data)
            w.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x, pool1_ind = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return x

    

        