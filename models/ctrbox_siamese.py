import torch.nn as nn
import numpy as np
import torch
from .model_parts_siamese import CombinationModule_FEM
from . import resnet
import matplotlib.pyplot as plt
import os

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return self.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):    #  kernel_size只能等于7或3， padding等于kernel_size的一半（3或1）
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAMModule(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16, kernel_size=7):
        super(CBAMModule, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class CbamEM(nn.Module):
    def __init__(self, in_channels):
        super(CbamEM, self).__init__()
        self.cbam = CBAMModule(in_channels)
        self.fl = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),   #这里输出的通道大小有待修改，可能不准
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU(inplace=True))

    def forward(self, x):
        x_ =self.cbam(x)
        x__ =self.fl(x_ + x)
        return x__
    
class FusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(FusionModule, self).__init__()
        # 门控机制模块，用于融合两个分支特征
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )
        # 通道注意力机制，用于突出有利于小目标的特征
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)

    def forward(self, x1, x2):
        combined = torch.cat((x1, x2), dim=1)
        gate_weights = self.gate_conv(combined)
        fused = x1 * gate_weights + x2 * (1 - gate_weights)
        enhanced_fused = self.channel_attention(fused) * fused
        return enhanced_fused
    
class CTRBOX_SIAMESE(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):  # head_conv = 256，down_ratio = 4
        super(CTRBOX_SIAMESE, self).__init__()
        channels = [3, 64, 256, 512, 1024, 2048]
        assert down_ratio in [2, 4, 8, 16]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet.resnet101(pretrained=pretrained) # 主干网
        self.dec_c2 = CombinationModule_FEM(512, 256, batch_norm=True)
        self.dec_c3 = CombinationModule_FEM(1024, 512, batch_norm=True)
        self.dec_c4 = CombinationModule_FEM(2048, 1024, batch_norm=True)
        self.cat_conv =  nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True))        
        self.cbam_c2 = CbamEM(256)
        self.fusion_module = FusionModule(in_channels=256)

        self.heads = heads

        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':                  #  channels[self.l1]=256 ，head_conv=256
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True), # nn.Conv2d(64,256, kernel_size=3, padding=1, bias=True)
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=3, padding=1, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channels[self.l1], head_conv, kernel_size=3, padding=1, bias=True),
                                #    nn.BatchNorm2d(head_conv),   # BN not used in the paper, but would help stable training
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
                
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def fill_fc_weights(self, m):
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x1, x2=None):

        # if self.training:
        x1 = self.base_network(x1)
        x2 = self.base_network(x2)
    
        c4_combine1 = self.dec_c4(x1[-1], x1[-2])
        c3_combine1 = self.dec_c3(c4_combine1, x1[-3])
        c2_combine1 = self.dec_c2(c3_combine1,  x1[-4])

        c4_combine2 = self.dec_c4(x2[-1], x2[-2])
        c3_combine2 = self.dec_c3(c4_combine2, x2[-3])
        c2_combine2 = self.dec_c2(c3_combine2,  x2[-4])

        # 使用融合模块融合两个分支的 c2 特征
        combine = self.fusion_module(c2_combine1, c2_combine2)

        combine = torch.cat([c2_combine1, c2_combine2], dim=1)
        combine = self.cbam_c2(self.cat_conv(combine))
        
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(combine)
            # dec_dict[head] = self.__getattr__(head)(combine)
            if 'hm' in head or 'cls' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])

        return dec_dict