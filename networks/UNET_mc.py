#
# MC-Dropout U-Net
#
# Architecturally identical to networks/UNET.py (same depth / filter counts, so
# its scores are directly comparable to the baseline) with a SINGLE addition:
# an nn.Dropout2d layer inside the bottleneck ("center"). That dropout is kept
# ACTIVE at inference time (see utilities/mc_dropout.enable_dropout) so we can
# draw T stochastic forward passes and estimate predictive uncertainty
# (Monte-Carlo Dropout, Gal & Ghahramani 2016).
#
# Derived from the DKFZ basic_unet_example UNet. Apache License 2.0.
#

import torch
import torch.nn as nn


class MCDropoutUNet(nn.Module):

    def __init__(self, num_classes, in_channels=1, initial_filter_size=64,
                 kernel_size=3, dropout_p=0.2, do_instancenorm=True):
        super().__init__()
        self.dropout_p = dropout_p

        self.contr_1_1 = self.contract(in_channels, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.contr_1_2 = self.contract(initial_filter_size, initial_filter_size, kernel_size, instancenorm=do_instancenorm)
        self.pool = nn.MaxPool2d(2, stride=2)

        self.contr_2_1 = self.contract(initial_filter_size, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)
        self.contr_2_2 = self.contract(initial_filter_size*2, initial_filter_size*2, kernel_size, instancenorm=do_instancenorm)

        self.contr_3_1 = self.contract(initial_filter_size*2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)
        self.contr_3_2 = self.contract(initial_filter_size*2**2, initial_filter_size*2**2, kernel_size, instancenorm=do_instancenorm)

        self.contr_4_1 = self.contract(initial_filter_size*2**2, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)
        self.contr_4_2 = self.contract(initial_filter_size*2**3, initial_filter_size*2**3, kernel_size, instancenorm=do_instancenorm)

        # bottleneck with MC dropout -- the ONLY change vs the baseline UNet
        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, 2, stride=2),
            nn.ReLU(inplace=True),
        )

        self.expand_4_1 = self.expand(initial_filter_size*2**4, initial_filter_size*2**3)
        self.expand_4_2 = self.expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, kernel_size=2, stride=2)

        self.expand_3_1 = self.expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = self.expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, 2, stride=2)

        self.expand_2_1 = self.expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = self.expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, 2, stride=2)

        self.expand_1_1 = self.expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = self.expand(initial_filter_size, initial_filter_size)
        # segmentation head
        self.final = nn.Conv2d(initial_filter_size, num_classes, kernel_size=1)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, instancenorm=True):
        if instancenorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(inplace=True))
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.LeakyReLU(inplace=True))
        return layer

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True),
        )

    @staticmethod
    def center_crop(layer, target_width, target_height):
        _, _, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        return layer[:, :, xy1:(xy1 + target_width), xy2:(xy2 + target_height)]

    def forward(self, x):
        contr_1 = self.contr_1_2(self.contr_1_1(x))
        pool = self.pool(contr_1)

        contr_2 = self.contr_2_2(self.contr_2_1(pool))
        pool = self.pool(contr_2)

        contr_3 = self.contr_3_2(self.contr_3_1(pool))
        pool = self.pool(contr_3)

        contr_4 = self.contr_4_2(self.contr_4_1(pool))
        pool = self.pool(contr_4)

        center = self.center(pool)

        crop = self.center_crop(contr_4, center.size()[2], center.size()[3])
        concat = torch.cat([center, crop], 1)
        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        crop = self.center_crop(contr_3, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        crop = self.center_crop(contr_2, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        crop = self.center_crop(contr_1, upscale.size()[2], upscale.size()[3])
        concat = torch.cat([upscale, crop], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))

        return self.final(expand)
