from __future__ import absolute_import

import os
import tensorflow as tf
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
import numpy as np

from realsafe.pytorch_model import pytorch_classifier_differentiable
from six.moves import urllib
from realsafe.utils import get_model_path, show_progress

@pytorch_classifier_differentiable(x_min=0.0, x_max=1.0, x_shape=(32,32,3), x_dtype=tf.float32, y_dtype=tf.int32,
                                   n_class=10)
class ResNet_Convex(torch.nn.Module):
    def __init__(self, use_cuda= True):
        torch.nn.Module.__init__(self)
        self.model = cifar_model_resnet().cuda()
            
        self.mean_torch = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape([1,3,1,1]).astype('float32'))
        self.std_torch = torch.from_numpy(np.array([0.225, 0.225, 0.225]).reshape([1,3,1,1]).astype('float32'))
        
    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3).contiguous()
        input_var = (x - self.mean_torch)/self.std_torch
        
        labels = self.model(input_var.cuda())
        return labels.cpu()
    
    def load(self, **kwargs):
        model_path = get_model_path('cifar_resnet_2px.pth')
        
        try:
            print(model_path)
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint['state_dict'][0])
            self.eval()
        except FileNotFoundError:
            raise IOError
            
def cifar_model_resnet(N = 1, factor=1): 
    def  block(in_filters, out_filters, k, downsample): 
        if not downsample: 
            k_first = 3
            skip_stride = 1
            k_skip = 1
        else: 
            k_first = 4
            skip_stride = 2
            k_skip = 2
        return [Dense(nn.Conv2d(in_filters, out_filters, k_first, stride=skip_stride, padding=1)), 
                nn.ReLU(), 
                Dense(nn.Conv2d(in_filters, out_filters, k_skip, stride=skip_stride, padding=0), 
                    None, nn.Conv2d(out_filters, out_filters, k, stride=1, padding=1)), 
                nn.ReLU()]
    conv1 = [nn.Conv2d(3,16,3,stride=1,padding=1), nn.ReLU()]
    conv2 = block(16,16*factor,3, False)
    for _ in range(N): 
        conv2.extend(block(16*factor,16*factor,3, False))
    conv3 = block(16*factor,32*factor,3, True)
    for _ in range(N-1): 
        conv3.extend(block(32*factor,32*factor,3, False))
    conv4 = block(32*factor,64*factor,3, True)
    for _ in range(N-1): 
        conv4.extend(block(64*factor,64*factor,3, False))
    layers = (conv1 + conv2 + conv3 + conv4 +
                [Flatten(), nn.Linear(64*factor*8*8,1000), nn.ReLU(), nn.Linear(1000, 10)])
    model = DenseSequential(*layers)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None: 
                m.bias.data.zero_()
    return model
    
class Dense(nn.Module): 
    def __init__(self, *Ws): 
        super(Dense, self).__init__()
        self.Ws = nn.ModuleList(list(Ws))
        if len(Ws) > 0 and hasattr(Ws[0], 'out_features'): 
            self.out_features = Ws[0].out_features

    def forward(self, *xs): 
        xs = xs[-len(self.Ws):]
        out = sum(W(x) for x,W in zip(xs, self.Ws) if W is not None)
        return out

class DenseSequential(nn.Sequential): 
    def forward(self, x):
        xs = [x]
        for module in self._modules.values():
            if 'Dense' in type(module).__name__:
                xs.append(module(*xs))
            else:
                xs.append(module(xs[-1]))
        return xs[-1]

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
        
