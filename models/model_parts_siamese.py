import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d  # 引入 DeformConv2d


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

        
class DeformableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DeformableConvBlock, self).__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation)
    def forward(self, x):
        # 计算 offset（偏移量）
        offset = self.offsets(x)
        # 进行可变形卷积
        x = self.deform_conv(x, offset)
        return x


class DeformableConvBlock_A(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DeformableConvBlock_A, self).__init__()
        self.offsets = nn.Conv2d(in_channels * 2, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size,
                                        stride=stride, padding=padding, dilation=dilation)
    def forward(self, ci, pi):
        # 计算 offset（偏移量）
        offset = self.offsets(ci)
        # 进行可变形卷积
        pi = self.deform_conv(pi, offset)
        return pi
    

class CombinationModule_FEM(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule_FEM, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
            
            self.se_module = SEModule(c_low)
            self.cbam = CBAMModule(c_low)

            self.dcn_a = DeformableConvBlock_A(c_up, c_up, kernel_size=3, stride=1, padding=1)

    def forward(self, x_low, x_up):  
        x_low = self.se_module(x_low)
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False)) # 插值时边角对齐
        x_low = self.dcn_a(torch.cat([x_low,x_up], dim=1), x_low)
        combined = torch.cat((x_up, x_low), dim=1)       
        combined = self.cbam(combined)
        return self.cat_conv(combined)
    

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEModule, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y

class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=2, stride=1, dilation=2),  # 使用空洞卷积扩大感受野
                                    nn.BatchNorm2d(c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.BatchNorm2d(c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.ReLU(inplace=True))

        self.se_module = SEModule(c_low)
        self.cbam = CBAMModule(c_low)

    def forward(self, x_low, x_up):  # x_low：来自网络较高层的特征; x_up: ~浅层， low 指空间尺寸小，
        x_low = self.se_module(x_low)
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        combined = torch.cat((x_up, x_low), 1)
        combined = self.cbam(combined)
        return self.cat_conv(combined)
    