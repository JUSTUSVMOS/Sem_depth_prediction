#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

This file defines the U-Net model (Resnet_UNet) based on a ResNet-18 encoder,
which can be used for tasks such as depth estimation. This code is structured
from previous examples and can be directly imported or executed to view the model summary.

Usage:
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

        # Load ResNet-18 encoder (default: no pre-trained weights)
        self.encoder = models.resnet18(pretrained=False)

        # Extract the initial layers of ResNet-18
        self.conv1 = self.encoder.conv1
        self.bn1   = self.encoder.bn1
        self.relu  = self.encoder.relu
        self.maxpool = self.encoder.maxpool

        # The four main layers of ResNet
        self.block1 = self.encoder.layer1  # Output channels: 64
        self.block2 = self.encoder.layer2  # Output channels: 128
        self.block3 = self.encoder.layer3  # Output channels: 256
        self.block4 = self.encoder.layer4  # Output channels: 512

        # Decoder: Using upsampling and skip connections to fuse encoder features
        self.up_conv6 = self.up_conv(512, 512)
        self.conv6    = self.double_conv(512 + 256, 256)

        self.up_conv7 = self.up_conv(256, 256)
        self.conv7    = self.double_conv(256 + 128, 128)

        self.up_conv8 = self.up_conv(128, 128)
        self.conv8    = self.double_conv(128 + 64, 64)

        self.up_conv9 = self.up_conv(64, 64)
        self.conv9_end = nn.Conv2d(64, 64, kernel_size=1)

        # Additional upsampling layer to restore resolution closer to the original
        self.up_conv10 = self.up_conv(64, 64)
        self.conv10_end = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Store input size for final cropping
        input_size = x.size()[2:]  # (height, width)

        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        block1 = self.block1(x)  # Preserve high-resolution features
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # Decoder (Upsampling + Skip Connections)
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

        # Additional upsampling to approach original resolution, followed by center cropping
        x = self.up_conv10(x)
        x = self.conv10_end(x)
        x = self.center_crop_to_target(x, input_size)

        return x

    def up_conv(self, in_channels, out_channels):
        """
        Upsampling layer (transposed convolution) with LeakyReLU activation.
        Default parameters: kernel_size=2, stride=2, padding=0, output_padding=0
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
        Two consecutive convolutional layers, each followed by Batch Normalization and LeakyReLU activation.
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
        Crop the input layer centrally to match the target_layer dimensions.
        """
        _, _, h, w = target_layer.size()
        _, _, h2, w2 = layer.size()
        diff_y = (h2 - h) // 2
        diff_x = (w2 - w) // 2
        return layer[:, :, diff_y:diff_y + h, diff_x:diff_x + w]

    def center_crop_to_target(self, layer, target_size):
        """
        Crop the input layer centrally to match the target_size (height, width).
        """
        _, _, h, w = layer.size()
        target_h, target_w = target_size
        diff_y = (h - target_h) // 2
        diff_x = (w - target_w) // 2
        return layer[:, :, diff_y:diff_y + target_h, diff_x:diff_x + target_w]


# If this file is executed directly, output the model structure summary
if __name__ == "__main__":
    from torchinfo import summary

    model = Resnet_UNet(out_channels=1)
    summary(model, input_size=(1, 3, 1000, 750),
            row_settings=["var_names"],
            col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
            col_width=16)
