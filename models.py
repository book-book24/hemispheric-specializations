import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.io import savemat


class Down3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Up_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2)
        self.conv = nn.Conv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up3d(nn.Module):
    def __init__(self, in_channels, out_channels, stri, pad, out_pad):
        super().__init__()
        self.dcov = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stri, padding=pad,
                               output_padding=out_pad),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.dcov(x)


class Up_res3d(nn.Module):
    def __init__(self, in_channels, out_channels, stri, pad, out_pad):
        super().__init__()
        self.dcov = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=stri, padding=pad,
                               output_padding=out_pad),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.dcov(x1)
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        x = x2 + x1
        return x


def save_parm(param1, save_name):
    param1 = param1.cpu()
    param1 = param1.numpy()
    # param2 = param2.cpu()
    # param2 = param2.numpy()
    # param3 = param3.cpu()
    # param3 = param3.numpy()
    savemat(save_name, {'FM': param1})



class Generator(nn.Module):
    # [55, 119, 87] -> [28, 60, 44] -> [14, 30, 22] -> [7, 15, 11] -> [4, 8, 6] DOWN
    # [4, 8, 6] -> [7, 15, 11] -> [14, 30, 22] -> [28, 60, 44] -> [55, 119, 87] UP
    def __init__(self):
        super(Generator, self).__init__()
        self.down1 = Down3d(1, 64)
        self.down2 = Down3d(64, 64)
        self.down3 = Down3d(64, 128)
        self.down4 = Down3d(128, 256)
        self.bottle = nn.Conv3d(256, 5000, 1)  # 3*4*3
        self.up1 = Up3d(5000, 256, 2, 1, 0)  # 6*8*6
        self.up2 = Up3d(256, 128, 2, 1, 1)  # 12*16*12
        self.up3 = Up3d(128, 64, 2, 1, 1)  # 24*32*24
        self.up4 = Up3d(64, 64, 2, 1, 1)  # 26*34*26
        self.out = nn.Sequential(
            nn.Conv3d(64, 1, 2, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, inp, train=True, save_name=''):
        if train:
            x = self.down1(inp)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.bottle(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.out(x)
        else:
            x = self.down1(inp)
            x = self.down2(x)
            x = self.down3(x)
            x = self.down4(x)
            x = self.bottle(x)
            x = self.up1(x)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.out(x)
            # save_parm(x.detach(), save_name)
        return x


class Discriminator(nn.Module):
    # [55, 119, 87] -> [27, 59, 43] -> [13, 29, 21] -> [6, 14, 10] -> [2, 6, 4]
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, bias=False),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.fc = nn.Linear(128 * 2 * 6 * 4, 1)
        # self.out = nn.Sequential(
        #     nn.Conv3d(128, 1, [2, 3, 2], 1, bias=False),
        #     nn.Sigmoid()
        # )
        # self.out = nn.Conv3d(128, 1, [2, 3, 2], 1, bias=False)

    def forward(self, img):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Pretrain_Net(nn.Module):
    def __init__(self, freeze):
        super().__init__()
        self.down1 = Down3d(1, 64)
        self.down2 = Down3d(64, 64)
        self.down3 = Down3d(64, 128)
        self.down4 = Down3d(128, 256)
        # self.bottle = nn.Conv3d(256, 5000, 1)
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

        # self.transcov = nn.Conv3d(5000, 256, 1)
        self.fc = nn.Sequential(
            nn.Linear(4 * 8 * 6 * 256, 2),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(0.25),
            # nn.Linear(256, 2),
            # nn.Dropout(0.25)
        )

    def forward(self, img):
        x = self.down1(img)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        # x = self.bottle(x)
        # x = self.transcov(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
