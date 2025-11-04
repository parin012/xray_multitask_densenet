#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models

def convert_first_conv_to_1ch(model):
    old = model.features.conv0
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)
    model.features.conv0 = new
    return model

class SimpleSegDecoder(nn.Module):
    def __init__(self, in_ch, mid_ch=64, out_ch=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, mid_ch, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, stride=2)
        self.up4 = nn.ConvTranspose2d(mid_ch, mid_ch, 2, stride=2)
        self.out_conv = nn.Conv2d(mid_ch, out_ch, 1)
    def forward(self, x, out_size):
        for up in [self.up1, self.up2, self.up3, self.up4]:
            x = F.relu(up(x))
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return self.out_conv(x)

class MultiTaskDenseNetXRay(nn.Module):
    def __init__(self, num_classes=5, seg_out_ch=1):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.backbone = convert_first_conv_to_1ch(base)
        ch = self.backbone.classifier.in_features
        self.dropout = nn.Dropout(p=0.2)
        self.cls_head = nn.Linear(ch, num_classes)
        self.seg_head = SimpleSegDecoder(ch, 64, seg_out_ch)
    def forward(self, x):
        H, W = x.shape[-2:]
        f = F.relu(self.backbone.features(x))
        pooled = F.adaptive_avg_pool2d(f, 1).flatten(1)
        pooled = self.dropout(pooled)
        cls = self.cls_head(pooled)
        seg = self.seg_head(f, (H, W))
        return cls, seg

