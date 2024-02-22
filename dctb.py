import torch.nn as nn

import torch
import torch.nn.functional as F
from . import SwinT     # SGBlock,FNet,Spartial_Attention
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ESRA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESRA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv_w = conv(f, f, kernel_size=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv_y = conv(f, f, kernel_size=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU(inplace=True)     ########wywywywywy
        self.GELU = nn.GELU()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.GELU(self.conv_max(v_max))    ########wywywywywy relu
        c3 = self.GELU(self.conv3(v_range))                ########wywywywywy relu
        cw = self.conv_w(c3)
        c3 = self.conv3_(cw)
        c3_w = cw + c3
        cy = self.conv_y(c3_w)
        c3_y = cy + c3
        c3 = F.interpolate(c3_y, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m

class ECCRA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ECCRA, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.conv_2 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.conv_3 = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        '''self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )'''
    
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y1 = self.conv_1(y)
        y2 = self.conv_2(y1)
        y3 = y1 + y2
        y4 = self.conv_3(y3)
        y5 = y + y4
        y = self.sigmoid(y5)
        #y = self.conv_du(y)
        return x * y
    


class DCTB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(DCTB, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.esa = ESRA(in_channels, nn.Conv2d)
        self.esa2 = ESRA(in_channels, nn.Conv2d)
        self.cca = ECCRA(in_channels)
        self.cca1 = ECCRA(in_channels)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        self.swinT1 = SwinT.SwinT()
        self.swinT2 = SwinT.SwinT()
        #self.swinT3 = SwinT.SwinT()
        #self.swinT4 = SwinT.SwinT()

    def forward(self, input):
        input5 = self.esa2(input)
        input5 = self.cca(input5)
        input1 = self.swinT1(input5)
        input2 = self.esa2(input1)
        input3 = self.cca(input2)
        input4 = self.swinT2(input3)
        out_fused1 = self.c1_r(input4)
        out_fused = input + out_fused1
        return out_fused

'''class HBCT(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(HBCT, self).__init__()
        self.rc = self.remaining_channels = in_channels
        self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.esa = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)
        self.cca = CCALayer(in_channels)
        self.cca1 = CCALayer(in_channels)
        # self.sparatt = Spartial_Attention.Spartial_Attention()
        self.swinT1 = SwinT.SwinT()
        self.swinT2 = SwinT.SwinT()
        #self.swinT3 = SwinT.SwinT()
        #self.swinT4 = SwinT.SwinT()

    def forward(self, input):
        #input5 = self.esa2(input)
        #input1 = self.cca(input5)
        input1 = self.swinT1(input)
        #input2 = self.esa2(input1)   #wywywyw
        #input3 = self.cca(input2)    #wywywyw
        input4 = self.swinT2(input1)
        out_fused1 = self.c1_r(input4)
        out_fused = input + out_fused1
        #input3 = self.swinT3(input2)
        #input4 = self.swinT4(input3)
        #out_fused = self.cca1(self.esa(self.c1_r(input2)))
        return out_fused'''


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
