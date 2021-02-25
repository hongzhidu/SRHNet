# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init as init
from models.seq_Cost import DisparityRegression, GetCostVolume

import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out



class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output      = self.layer1(output)
        output_raw  = self.layer2(output)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)


        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners='False')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners='False')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners='False')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear',align_corners='False')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature
    
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.gn = nn.GroupNorm(out_channels//8, out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.gn = nn.GroupNorm(out_channels//8, out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.gn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        
        if deconv and is_3d: 
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class ConvGRUCell(nn.Module):
    
    def __init__(self,input_size,hidden_size,kernel_size,cuda_flag):
        super(ConvGRUCell,self).__init__()
        self.input_size  = input_size
        self.cuda_flag   = cuda_flag
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.ConvGates   = nn.Conv2d(self.input_size + self.hidden_size,2 * self.hidden_size,3,padding=self.kernel_size//2)
        self.Conv_ct     = nn.Conv2d(self.input_size + self.hidden_size,self.hidden_size,3,padding=self.kernel_size//2) 
        self.gn = nn.GroupNorm(hidden_size//8, hidden_size)
    def forward(self,input,hidden):
        if hidden is None:
           size_h    = [input.data.size()[0],self.hidden_size] + list(input.data.size()[2:])
           if self.cuda_flag  == True:
              hidden    = Variable(torch.zeros(size_h)).cuda() 
           else:
              hidden    = Variable(torch.zeros(size_h))
        c1           = self.ConvGates(torch.cat((input,hidden),1))
        (rt,ut)      = c1.chunk(2, 1)
        rt           = self.gn(rt)
        ut           = self.gn(ut)
        reset_gate   = torch.sigmoid(rt)
        update_gate  = torch.sigmoid(ut)
        gated_hidden = torch.mul(reset_gate,hidden)
        p1           = self.Conv_ct(torch.cat((input,gated_hidden),1))
        p1           = self.gn(p1)
        ct           = F.tanh(p1)
        next_h       = torch.mul(update_gate,hidden) + (1-update_gate)*ct
        return next_h
    

class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp+1, x.size()[3]*4, x.size()[4]*4], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)

class GRU(nn.Module):
    def __init__(self, maxdisp=192):
        super(GRU, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(64, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=1, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.gru1a = ConvGRUCell(64, 32, kernel_size=3, cuda_flag=True)       
        self.gru2a = ConvGRUCell(32, 32, kernel_size=3, cuda_flag=True)
        
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.gru3a = ConvGRUCell(48, 48, kernel_size=3, cuda_flag=True)
        
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)         
        self.gru4a = ConvGRUCell(64, 64, kernel_size=3, cuda_flag=True)
        
        self.deconv2 = Conv2x(64, 48, deconv=True, concat=True)
        self.deconv1 = Conv2x(48, 32, deconv=True, concat=True)
        
        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        
        self.gru1b = ConvGRUCell(48, 48, kernel_size=3, cuda_flag=True)
        self.gru2b = ConvGRUCell(64, 64, kernel_size=3, cuda_flag=True)
        
        self.deconv4b = Conv2x(64, 48, deconv=True, concat=True)
        self.deconv3b = Conv2x(48, 32, deconv=True, concat=True)
        self.cov_prob = nn.Sequential(BasicConv(32, 8, kernel_size=3, padding=1),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1))
        self.cov_prob1 = nn.Sequential(BasicConv(32, 8, kernel_size=3, padding=1),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1))
        self.maxdisp=maxdisp
        self.disparity = Disp(maxdisp=self.maxdisp)
    def forward(self, x):
        h1a =x.new().resize_(x.size()[0], 32, x.size()[3], x.size()[4]).zero_().cuda()
        h2a =x.new().resize_(x.size()[0], 32, x.size()[3], x.size()[4]).zero_().cuda()
        h3a =x.new().resize_(x.size()[0], 48, x.size()[3]//2, x.size()[4]//2).zero_().cuda()
        h4a =x.new().resize_(x.size()[0], 64, x.size()[3]//4, x.size()[4]//4).zero_().cuda()
        h1b =x.new().resize_(x.size()[0], 48, x.size()[3]//2, x.size()[4]//2).zero_().cuda()
        h2b =x.new().resize_(x.size()[0], 64, x.size()[3]//4, x.size()[4]//4).zero_().cuda()
        prob_volume = x.new().resize_(x.size()[0], 1, x.size()[2], x.size()[3],x.size()[4]).zero_().cuda()
        if self.training:
            prob_volume1 = x.new().resize_(x.size()[0], 1, x.size()[2], x.size()[3],x.size()[4]).zero_().cuda()
        for e in range(x.size()[2]):
            cost = x[:, :, e]
            
            cost = self.gru1a(cost, h1a)
            h1a = cost
            cost = self.gru2a(cost, h2a)
            h2a = cost
            
            cost = self.conv1a(cost)
            cost = self.gru3a(cost, h3a)
            h3a = cost   #1/6*48
            cost = self.conv2a(cost)    #1/12*64
            cost =self.gru4a(cost, h4a)
            h4a = cost
            
            cost = self.deconv2(cost, h3a) #1/6 *48
            rem0 = cost
            cost = self.deconv1(cost, h2a) #1/3 *32 
            rem1  = cost
            cost1 = self.cov_prob1(cost)
            if self.training:
                prob_volume1[:,:,e] = cost1
            
            cost = self.conv1b(cost, rem0)
            cost = self.gru1b(cost, h1b)
            h1b = cost   #1/6*48
            cost = self.conv2b(cost, h4a)    #1/12*64
            cost =self.gru2b(cost, h2b)
            h2b = cost
            
            cost = self.deconv4b(cost, h1b) #1/6 *48
            cost = self.deconv3b(cost, rem1) #1/3 *32
            cost = self.cov_prob(cost)
            prob_volume[:,:,e] = cost
            
        prob_volume = prob_volume.contiguous()
        if self.training:
            prob_volume1 = prob_volume1.contiguous()
            return self.disparity(prob_volume1), self.disparity(prob_volume)
        else:
            return self.disparity(prob_volume)
        
      
        
class SHRNet(nn.Module):
    def __init__(self, maxdisp=192):
        super(SHRNet, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(
            BasicConv(3, 16, kernel_size=3, padding=1),
            BasicConv(16, 32, kernel_size=3, padding=1))
        self.feature =  feature_extraction()
        self.cv = GetCostVolume(int(self.maxdisp/4))
        self.gru_f = GRU(maxdisp=self.maxdisp)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, y):
        x = self.feature(x)
        y = self.feature(y)
        
        x = self.cv(x,y) #B, C, D=64, H, W
        
        return self.gru_f(x)

