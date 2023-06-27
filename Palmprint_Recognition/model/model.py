import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from thop import profile
import math
from torch.nn import Parameter
from config import device, num_classes, emb_size
import time


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class GDConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, padding, bias=False):
        super(GDConv, self).__init__()
        self.depthwise = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, groups=in_planes,
                                   bias=bias)
        self.bn = nn.BatchNorm2d(in_planes)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileFaceNet(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileFaceNet, self).__init__()
        block = InvertedResidual
        input_channel = 64
        last_channel = 512

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [2, 64, 5, 2],
                [4, 128, 1, 2],
                [1, 128, 6, 1],
                [2, 128, 1, 2],   # add a block to match dims
                [4, 128, 1, 2],
                [2, 128, 2, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        # input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv_first = nn.Conv2d(3, 3, kernel_size=3)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2)
        self.dw_conv = DepthwiseSeparableConv(in_planes=64, out_planes=64, kernel_size=3, padding=1)
        features = list()
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers

        self.conv2 = ConvBNReLU(input_channel, self.last_channel, kernel_size=1)
        self.gdconv = GDConv(in_planes=self.last_channel, out_planes=self.last_channel, kernel_size=7, padding=0)
        # self.conv3 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(512)
        # self.gdconv2 = GDConv(in_planes=self.last_channel, out_planes=self.last_channel, kernel_size=7, padding=0)

        # self.conv4 = nn.Conv2d(128, 64, kernel_size=1)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.Conv2d(64, 32, kernel_size=1)
        # self.bn5 = nn.BatchNorm2d(32)
        # self.conv6 = nn.Conv2d(32, 8, kernel_size=1)
        # self.bn6 = nn.BatchNorm2d(8)
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        # self.linear1 = nn.Conv2d(8, 512, 8)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        # x = F.interpolate(x, scale_factor=0.5)
        # print('downsample',x.shape)
        #改成先下采样到112，112
        x = self.conv1(x)
        # print('conv1',x.shape)
        x = self.dw_conv(x)
        # print('dw_conv',x.shape)
        x = self.features(x)
        # print('feature',x.shape)
        x = self.conv2(x)
        # print('conv2',x.shape)
        x = self.gdconv(x)
        # print('gdconv',x.shape)
        x = self.bn(x)
        # print('bn',x.shape)
        x = x.view(x.size(0), -1)
        # print('view',x.shape)
        
        # previous
        # x = self.conv4(x)
        # # print('shape',x.shape)
        # x = self.bn4(x)
        # x = self.conv5(x)
        # # print('shape',x.shape)
        # x = self.bn5(x)
        # x = self.conv6(x)
        # # print('shape',x.shape)
        # x = self.bn6(x)
        # print('shape',x.shape)
        # x = self.linear1(x)
        # print('shape1',x.shape)
        # x = x.view(x.size(0), -1)
        return x


class ArcMarginModel(nn.Module):
    ## loss function
    def __init__(self, emb_size=512, num_classes=320, margin_s=32.0, margin_m=0.50, easy_margin=False):
        super(ArcMarginModel, self).__init__()

        self.weight = Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.m = margin_m
        self.s = margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # from torchscope import scope
    # 输出的是embedding向量
    model = MobileFaceNet()
    # print(model)
    input = Variable(torch.FloatTensor(4, 3, 224, 224))  #B,C,H,W
    # input = torch.randn(4, 3, 224, 224)
    # x = model(input)
    # print(x.shape)  #([4, 512])
    # # scope(model, input_size=(3, 112, 112))
    T1 = time.time()

    out = model(input)
    print(out.shape)
    T2 = time.time()
    print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
    print(count_parameters(model))
    flops, params = profile(model, (input,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')