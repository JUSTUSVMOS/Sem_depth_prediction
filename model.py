#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

本檔案定義了基於 ResNet-18 編碼器的 U-Net 模型 (Resnet_UNet)，
可用於深度估計等任務。這份程式碼是根據先前的範例整理而成，
可以直接匯入使用或執行查看模型摘要。

使用方式：
    from model import Resnet_UNet
    model = Resnet_UNet(out_channels=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Resnet_UNet(nn.Module):
    def __init__(self, out_channels=1):
        super(Resnet_UNet, self).__init__()

        # 載入 ResNet-18 編碼器（此處預設不使用預訓練權重）
        self.encoder = models.resnet18(pretrained=False)

        # 提取 ResNet-18 的前置層
        self.conv1 = self.encoder.conv1
        self.bn1   = self.encoder.bn1
        self.relu  = self.encoder.relu
        self.maxpool = self.encoder.maxpool

        # ResNet 的四個主要層級
        self.block1 = self.encoder.layer1  # 輸出通道數 64
        self.block2 = self.encoder.layer2  # 輸出通道數 128
        self.block3 = self.encoder.layer3  # 輸出通道數 256
        self.block4 = self.encoder.layer4  # 輸出通道數 512

        # Decoder 部分：利用上採樣和跳躍連接融合 encoder 特徵
        self.up_conv6 = self.up_conv(512, 512)
        self.conv6    = self.double_conv(512 + 256, 256)

        self.up_conv7 = self.up_conv(256, 256)
        self.conv7    = self.double_conv(256 + 128, 128)

        self.up_conv8 = self.up_conv(128, 128)
        self.conv8    = self.double_conv(128 + 64, 64)

        self.up_conv9 = self.up_conv(64, 64)
        self.conv9_end = nn.Conv2d(64, 64, kernel_size=1)

        # 額外增加一層上採樣，學習如何從低解析度恢復至原始尺寸
        self.up_conv10 = self.up_conv(64, 64)
        self.conv10_end = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 保存輸入尺寸，用於最後裁剪
        input_size = x.size()[2:]  # (height, width)

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        block1 = self.block1(x)  # 保留較高解析度特徵
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # Decoder 部分（上採樣 + 跳躍連接）
        x = self.up_conv6(block4)
        x = self.center_crop(x, block3)
        x = torch.cat([x, block3], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = self.center_crop(x, block2)
        x = torch.cat([x, block2], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = self.center_crop(x, block1)
        x = torch.cat([x, block1], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = self.conv9_end(x)

        # 額外上採樣至接近原始解析度，再用中心裁剪精調
        x = self.up_conv10(x)
        x = self.conv10_end(x)
        x = self.center_crop_to_target(x, input_size)

        return x

    def up_conv(self, in_channels, out_channels):
        """
        上採樣層（反捲積）與 LeakyReLU 激活
        預設 kernel_size=2, stride=2, padding=0, output_padding=0
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2,
                padding=0, output_padding=0
            ),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def double_conv(self, in_channels, out_channels):
        """
        連續兩次捲積，每次捲積後附加 Batch Normalization 與 LeakyReLU 激活
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def center_crop(self, layer, target_layer):
        """
        將輸入的 layer 中心裁剪，使其與 target_layer 尺寸匹配
        """
        _, _, h, w = target_layer.size()
        _, _, h2, w2 = layer.size()
        diff_y = (h2 - h) // 2
        diff_x = (w2 - w) // 2
        return layer[:, :, diff_y:diff_y + h, diff_x:diff_x + w]

    def center_crop_to_target(self, layer, target_size):
        """
        將 layer 中心裁剪，使其尺寸符合 target_size (height, width)
        """
        _, _, h, w = layer.size()
        target_h, target_w = target_size
        diff_y = (h - target_h) // 2
        diff_x = (w - target_w) // 2
        return layer[:, :, diff_y:diff_y + target_h, diff_x:diff_x + target_w]


# 如果直接執行此檔案，則輸出模型結構摘要
if __name__ == "__main__":
    from torchinfo import summary

    model = Resnet_UNet(out_channels=1)
    summary(model, input_size=(1, 3, 1000, 750),
            row_settings=["var_names"],
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
            col_width=16)
