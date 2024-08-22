#Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import weight_init, DropPath
from timm.models.registry import register_model
from torch.nn.functional import interpolate, elu
import thop
import time


class OutnetCoordConv(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutnetCoordConv, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)



    def forward(self, x):

        outputs = {}
        p = elu(self.p_out(x))
        outputs['p'] = self.p_out1(p)

        xyzi =elu(self.xyzi_out(x))
        outputs['xyzi'] = self.xyzi_out1(xyzi)
        xyzis = elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
        return outputs


# Series informed activation function. Implemented by conv.
class activation(nn.ReLU):
    def __init__(self, dim, act_num=3, deploy=False):
        super(activation, self).__init__()
        self.act_num = act_num  # the number of stacking activation functions
        self.deploy = deploy
        self.dim = dim
        self.weight = torch.nn.Parameter(torch.randn(dim, 1, act_num*2 + 1, act_num*2 + 1))
        if deploy:
            self.bias = torch.nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None
            self.bn = nn.BatchNorm2d(dim, eps=1e-6)
        weight_init.trunc_normal_(self.weight, std=.02)

    def forward(self, x):
        if self.deploy:
            return torch.nn.functional.conv2d(
                super(activation, self).forward(x), 
                self.weight, self.bias, padding=self.act_num, groups=self.dim)
        else:
            return self.bn(torch.nn.functional.conv2d(
                super(activation, self).forward(x),
                self.weight, padding=self.act_num, groups=self.dim))

    def _fuse_bn_tensor(self, weight, bn):
        kernel = weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (0 - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.weight, self.bn)
        self.weight.data = kernel
        self.bias = torch.nn.Parameter(torch.zeros(self.dim))
        self.bias.data = bias
        self.__delattr__('bn')
        self.deploy = True


class Block(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=1, deploy=False, ada_pool=None):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num, deploy=self.deploy)
 
    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)
            
            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            
            x = self.conv2(x)

        # x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1,3), self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True


class Block_dilated(nn.Module):
    def __init__(self, dim, dim_out, act_num=3, stride=1, deploy=False, ada_pool=None, dilation=(1, 1), paddings=(1, 1)):
        super().__init__()
        self.act_learn = 1
        self.deploy = deploy
        if self.deploy:
            self.conv = nn.Conv2d(dim, dim_out, kernel_size=1, dilation=dilation, padding=paddings)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, dilation=dilation, padding=paddings),
                nn.BatchNorm2d(dim, eps=1e-6),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out, eps=1e-6)
            )

        if not ada_pool:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        else:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((ada_pool, ada_pool))

        self.act = activation(dim_out, act_num, deploy=self.deploy)

    def forward(self, x):
        if self.deploy:
            x = self.conv(x)
        else:
            x = self.conv1(x)

            # We use leakyrelu to implement the deep training technique.
            x = torch.nn.functional.leaky_relu(x, self.act_learn)

            x = self.conv2(x)

        # x = self.pool(x)
        x = self.act(x)
        return x

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std

    def switch_to_deploy(self):
        kernel, bias = self._fuse_bn_tensor(self.conv1[0], self.conv1[1])
        self.conv1[0].weight.data = kernel
        self.conv1[0].bias.data = bias
        # kernel, bias = self.conv2[0].weight.data, self.conv2[0].bias.data
        kernel, bias = self._fuse_bn_tensor(self.conv2[0], self.conv2[1])
        self.conv = self.conv2[0]
        self.conv.weight.data = torch.matmul(kernel.transpose(1, 3),
                                             self.conv1[0].weight.data.squeeze(3).squeeze(2)).transpose(1, 3)
        self.conv.bias.data = bias + (self.conv1[0].bias.data.view(1, -1, 1, 1) * kernel).sum(3).sum(2).sum(1)
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        self.act.switch_to_deploy()
        self.deploy = True
    

class VanillaNet(nn.Module):  #或许可以在最前面加一个norm
    def __init__(self, in_chans=1, num_classes=64, dims=[96, 192, 384, 768],
                 drop_rate=0, act_num=3, strides=None, deploy=False, ada_pool=None, paddings=None, **kwargs):
        super().__init__()
        self.deploy = deploy
        # stride, padding = (4, 0) if not ada_pool else (3, 1)
        stride, padding = (1, 1)
        if self.deploy:
            self.stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=stride, padding=padding),
                activation(dims[0], act_num, deploy=self.deploy)
            )
        else:
            self.stem1 = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(dims[0], eps=1e-6),
            )
            self.stem2 = nn.Sequential(
                nn.Conv2d(dims[0], dims[0], kernel_size=1, stride=1),
                nn.BatchNorm2d(dims[0], eps=1e-6),
                activation(dims[0], act_num)
            )

        self.act_learn = 1  # range from 0 to 1 during training

        self.stages = nn.ModuleList()
        for i in range(len(strides)):
            if not ada_pool:
                stage = Block_dilated(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=1, deploy=deploy,
                                      dilation=paddings[i], paddings=paddings[i])
                # stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=1, deploy=deploy)
            else:
                stage = Block(dim=dims[i], dim_out=dims[i+1], act_num=act_num, stride=strides[i], deploy=deploy,
                              ada_pool=ada_pool[i])
            self.stages.append(stage)
        self.depth = len(strides)

        if self.deploy:
            self.cls = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
            )
        else:
            self.cls1 = nn.Sequential(
                # nn.AdaptiveAvgPool2d((1,1)),
                nn.Dropout(drop_rate),
                nn.Conv2d(dims[-1], num_classes, 1),
                nn.BatchNorm2d(num_classes, eps=1e-6),
            )
            self.cls2 = nn.Sequential(
                nn.Conv2d(num_classes, num_classes, 1)
            )
        
        self.apply(self._init_weights)
        self.pred = OutnetCoordConv(64, 1, 3)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weight_init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def change_act(self, m):
        for i in range(self.depth):
            self.stages[i].act_learn = m
        self.act_learn = m

    def forward(self, x):
        if self.deploy:
            x = self.stem(x)
        else:
            x = self.stem1(x)
            x = torch.nn.functional.leaky_relu(x,self.act_learn)
            x = self.stem2(x)

        for i in range(self.depth):
            x = self.stages[i](x)
        # x = interpolate(x, scale_factor=2)

        if self.deploy:
            x = self.cls(x)
        else:
            x = self.cls1(x)
            x = torch.nn.functional.leaky_relu(x, self.act_learn)
            x = self.cls2(x)

        x = self.pred(x)

        probs = torch.sigmoid(torch.clamp(x['p'], -16., 16.))
        xyzi_est = x['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(x['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig

    def _fuse_bn_tensor(self, conv, bn):
        kernel = conv.weight
        bias = conv.bias
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta + (bias - running_mean) * gamma / std
    
    def switch_to_deploy(self):
        self.stem2[2].switch_to_deploy()
        kernel, bias = self._fuse_bn_tensor(self.stem1[0], self.stem1[1])
        self.stem1[0].weight.data = kernel
        self.stem1[0].bias.data = bias
        kernel, bias = self._fuse_bn_tensor(self.stem2[0], self.stem2[1])
        self.stem1[0].weight.data = torch.einsum('oi,icjk->ocjk', kernel.squeeze(3).squeeze(2), self.stem1[0].weight.data)
        self.stem1[0].bias.data = bias + (self.stem1[0].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.stem = torch.nn.Sequential(*[self.stem1[0], self.stem2[2]])
        self.__delattr__('stem1')
        self.__delattr__('stem2')

        for i in range(self.depth):
            self.stages[i].switch_to_deploy()

        kernel, bias = self._fuse_bn_tensor(self.cls1[1], self.cls1[2])
        self.cls1[1].weight.data = kernel
        self.cls1[1].bias.data = bias
        kernel, bias = self.cls2[0].weight.data, self.cls2[0].bias.data
        self.cls1[1].weight.data = torch.matmul(kernel.transpose(1,3), self.cls1[1].weight.data.squeeze(3).squeeze(2)).transpose(1,3)
        self.cls1[1].bias.data = bias + (self.cls1[1].bias.data.view(1,-1,1,1)*kernel).sum(3).sum(2).sum(1)
        self.cls = torch.nn.Sequential(*self.cls1[0:2])
        self.__delattr__('cls1')
        self.__delattr__('cls2')
        self.deploy = True

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 256, 256).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(200):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 200:.4f} s')



@register_model
def vanillanet_5_v1(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128, 64, 64, 64], strides=[2, 2, 2], **kwargs)
    return model

@register_model  # fix the number of channels 128 (no try)
def vanillanet_5_v2(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128, 128, 128, 128], strides=[2, 2, 2], **kwargs)
    return model

@register_model
def vanillanet_dilated(pretrained=False,in_22k=False, **kwargs):  # add dilated convolutions
    model = VanillaNet(dims=[128, 64, 64, 64, 64], strides=[1, 1, 1, 1], paddings=[(2, 2), (4, 4), (8, 8), (16, 16)], **kwargs)
    return model

@register_model
def vanillanet_dilated_6(pretrained=False,in_22k=False, **kwargs):  # add dilated convolutions
    model = VanillaNet(dims=[64, 64, 64, 64, 64, 64], strides=[1, 1, 1, 1, 1],
                       paddings=[(1, 1), (2, 2), (4, 4), (8, 8), (16, 16)], **kwargs)
    return model

@register_model
def vanillanet_7_v1(pretrained=False,in_22k=False, **kwargs):  #
    model = VanillaNet(dims=[64, 64, 64, 64, 64, 64], strides=[1, 1, 1, 1, 1], **kwargs)
    return model

@register_model
def vanillanet_5(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4], strides=[2,2,2], **kwargs)
    return model

@register_model
def vanillanet_6(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[2,2,2,1], **kwargs)
    return model

@register_model
def vanillanet_7(pretrained=False,in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,2,1], **kwargs)
    return model

@register_model
def vanillanet_8(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,2,1], **kwargs)
    return model

@register_model
def vanillanet_9(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 1024*4, 1024*4], strides=[1,2,2,1,1,2,1], **kwargs)
    return model

@register_model
def vanillanet_10(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,2,1],
        **kwargs)
    return model

@register_model
def vanillanet_11(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,2,1],
        **kwargs)
    return model

@register_model
def vanillanet_12(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,2,1],
        **kwargs)
    return model

@register_model
def vanillanet_13(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*4, 128*4, 256*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 512*4, 1024*4, 1024*4],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    return model

@register_model
def vanillanet_13_x1_5(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        **kwargs)
    return model

@register_model
def vanillanet_13_x1_5_ada_pool(pretrained=False, in_22k=False, **kwargs):
    model = VanillaNet(
        dims=[128*6, 128*6, 256*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 512*6, 1024*6, 1024*6],
        strides=[1,2,2,1,1,1,1,1,1,2,1],
        ada_pool=[0,38,19,0,0,0,0,0,0,10,0],
        **kwargs)
    return model
