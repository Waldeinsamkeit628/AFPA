import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
from models.MPNCOV import MPNCOV
import torch.nn.functional as F
import models.resnet
import models.densenet
import models.senet
from models.operations import *
import torch.fft
import random

import re
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ['fpa']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def GlobalFilter(x, b, w):
    fft_pre1 = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    fre_p = torch.angle(fft_pre1)
    fre_p_random = fre_p[random.randint(0, b - 1)]
    fre_p = fre_p *(1-w) + fre_p_random * w
    fre_ = fre_m * torch.exp(1j * fre_p)

    fft1 = fre_.real
    fft2 = fre_.imag
    fft_pre = torch.log(1 + torch.sqrt(fft1 ** 2 + fft2 ** 2 + 1e-8))

    return fft_pre

class FreqAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(FreqAttention, self).__init__()
        self.channels = channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        freq_domain = torch.fft.fft2(x)
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        freq_domain = freq_domain * y
        out = torch.fft.ifft2(freq_domain).real

        return out

class Model(nn.Module):
    def __init__(self, pretrained=True, args=None):
        self.inplanes = 64
        num_classes = args.num_classes
        is_fix = args.is_fix
        sf_size = args.sf_size
        self.arch = args.backbone
        self.adj = args.adj
        self.sf = torch.from_numpy(args.sf).cuda()
        super(Model, self).__init__()

        ''' backbone net'''
        block = Bottleneck
        layers = [3, 4, 23, 3]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.b = args.batch_size
        self.w = args.phasew

        self.match_channels_x2 = nn.Conv2d(512, 2048, kernel_size=1)
        self.match_channels_x3 = nn.Conv2d(1024, 2048, kernel_size=1)

        nn.init.kaiming_normal_(self.match_channels_x2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.match_channels_x3.weight, mode='fan_out', nonlinearity='relu')


        self.cov_channel = 2048
        self.parts = 7
        self.map_threshold = 0.9
        self.cov = nn.Conv2d(self.cov_channel, self.parts, 1)
        self.pool = nn.MaxPool2d(28, 28)
        self.p_linear = nn.Linear(self.cov_channel * self.parts, 312, False)
        self.dropout2 = nn.Dropout(0.4)
        if (is_fix):
            for p in self.parameters():
                p.requires_grad = False

        if 'densenet' in self.arch:
            feat_dim = 1920
        else:
            feat_dim = 2048

        ''' Open-Domain Recognition Module '''
        self.odr_proj1 = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_proj2 = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.odr_spatial = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.odr_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, int(256 / 16), kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(256 / 16), 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.odr_classifier = nn.Linear(int(256 * (256 + 1) / 2), num_classes)

        ''' Zero-Shot Recognition Module '''
        self.zsr_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.zsr_sem = nn.Sequential(
            nn.Linear(sf_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, feat_dim),
            nn.LeakyReLU(),
        )
        self.zsr_aux = nn.Linear(feat_dim, num_classes)

        ''' FFT Module '''
        self.fft2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, args.att),
            nn.ReLU(),
        )
        self.fft_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        ''' params ini '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x2_upsampled = F.interpolate(x2, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x3_upsampled = F.interpolate(x3, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x2_matched = self.match_channels_x2(x2_upsampled)
        x3_matched = self.match_channels_x3(x3_upsampled)

        x = x2_matched + x3_matched + x4

        last_conv = x

        ''' ODR Module '''
        x1 = self.odr_proj1(last_conv)
        x2 = x1

        att1 = self.odr_spatial(x1)
        att2 = self.odr_channel(x2)

        x1 = att2 * x1 + x1
        x1 = x1.view(x1.size(0), x1.size(1), -1)

        x2 = att1 * x2 + x2
        x2 = x2.view(x2.size(0), x2.size(1), -1)

        x1 = x1 - torch.mean(x1, dim=2, keepdim=True)
        x2 = x2 - torch.mean(x2, dim=2, keepdim=True)
        A = 1. / x1.size(2) * x1.bmm(x2.transpose(1, 2))

        x = MPNCOV.SqrtmLayer(A, 5)
        x = MPNCOV.TriuvecLayer(x)

        odr_x = x.view(x.size(0), -1)
        odr_logit = self.odr_classifier(odr_x)

        weights = torch.softmax(self.cov(last_conv), dim=1)
        w = last_conv.size()
        batch, parts, width, height = weights.size()
        weights_layout = weights.view(batch, -1)
        threshold_value, _ = weights_layout.max(dim=1)
        local_max, _ = weights.view(batch, parts, -1).max(dim=2)
        threshold_value = self.map_threshold * threshold_value.view(batch, 1) \
            .expand(batch, parts)
        weights = weights * local_max.ge(threshold_value).view(batch, parts, 1, 1). \
            float().expand(batch, parts, width, height)
        last_conv = GlobalFilter(last_conv, self.b, self.w)

        for k in range(self.parts):
            Y = last_conv * weights[:, k, :, :]. \
                unsqueeze(dim=1). \
                expand(w[0], self.cov_channel, w[2], w[3])

        Y = Y + last_conv
        fft_last = self.fft_proj(Y).view(Y.size(0), -1)
        fft_att = self.fft2(fft_last)
        ''' ZSR Module '''
        x_all =  fft_last

        zsr_classifier = self.zsr_sem(self.sf)
        w_norm = F.normalize(zsr_classifier, p=2, dim=1)
        x_norm = F.normalize(x_all, p=2, dim=1)
        zsr_logit = x_norm.mm(w_norm.permute(1, 0))
        zsr_logit_aux = self.zsr_aux(x_all)
        fft_logit = fft_att.mm(self.sf.permute(1, 0))

        return (odr_logit, zsr_logit, zsr_logit_aux, fft_att,fft_logit), (odr_x, x_all,last_conv)


def WeightedL1(pred, gt):
    wt = (pred - gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0), wt.size(1))
    loss = wt * (pred - gt).abs()
    return loss.sum() / loss.size(0)


class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()

        self.cls_loss = nn.CrossEntropyLoss()
        self.sigma = args.sigma
        self.weight = args.lossw
        self.odr = args.odr

    def forward(self, label, logits, att):
        odr_logit = logits[0]
        zsr_logit = logits[1]
        zsr_logit_aux = logits[2]
        fft_att = logits[3]
        #p_output = logits[5]

        ''' ODR Loss '''
        prob = F.softmax(odr_logit, dim=1).detach()
        y = prob[torch.arange(prob.size(0)).long(), label]
        mw = torch.exp(-(y - 1.0) ** 2 / self.sigma)
        one_hot = torch.zeros_like(odr_logit)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        if self.odr == 1:
            odr_logit = odr_logit *(1-one_hot*mw.view(mw.size(0),1))
        else:
            odr_logit = odr_logit
        L_odr = self.cls_loss(odr_logit, label)

        ''' ZSL Loss '''
        idx = torch.arange(zsr_logit.size(0)).long()
        L_zsr = (1 - zsr_logit[idx, label]).mean()

        L_aux = self.cls_loss(zsr_logit_aux, label)
        L_fft = WeightedL1(fft_att, att)
        total_loss = L_odr + L_zsr + L_aux + L_fft * self.weight

        return total_loss, L_odr, L_zsr, L_aux, L_fft


def fpa(pretrained=False, loss_params=None, args=None):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Model(pretrained, args)
    loss_model = Loss(args)
    if pretrained:
        model_dict = model.state_dict()
        # pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        pretrained_dict = torch.load('/home/ywt/raw_data/model/resnet101-5d3b4d8f.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model, loss_model


