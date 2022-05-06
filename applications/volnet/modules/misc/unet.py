import torch
import torch.nn as nn
import torch.nn.functional as F


class PaddingBlock2d(nn.Module):
    def __init__(self, padding_size, padding_mode='circular'):
        super(PaddingBlock2d, self).__init__()
        self.padding_size = self._parse_padding_size(padding_size)
        self.padding_mode = self._parse_padding_mode(padding_mode)

    def forward(self, x):
        vpad = self.padding_size[:2] + [0, 0]
        hpad = [0, 0] + self.padding_size[2:]
        return F.pad(
            F.pad(x, hpad, mode=self.padding_mode[1]),
            vpad, mode=self.padding_mode[0]
        )

    @staticmethod
    def _parse_padding_size(padding_size):
        if isinstance(padding_size, int):
            out = [padding_size,] * 4
        elif isinstance(padding_size, (tuple, list)):
            if len(padding_size) == 2:
                out = ([padding_size[0]] * 2) + ([padding_size[1]] * 2)
            elif len(padding_size) == 4:
                out = list(padding_size)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        return out

    @staticmethod
    def _parse_padding_mode(padding_mode):
        if padding_mode is None:
            padding_mode = ('reflect', 'circular')
        elif isinstance(padding_mode, tuple):
            assert len(padding_mode) == 2
            for p in padding_mode:
                assert p in ['constant', 'reflect', 'replicate', 'circular']
            padding_mode = padding_mode
        elif isinstance(padding_mode, str):
            assert padding_mode in ['constant', 'reflect', 'replicate', 'circular']
            padding_mode = (padding_mode, padding_mode)
        else:
            raise NotImplementedError()
        return padding_mode


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, padding_mode=None):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels

        self.in_layer_encoder = nn.Sequential(       #64,41,180
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.in_channels, self.inner_channels, kernel_size=3, stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU()
        )
        self.hr_conv_layer_encoder = nn.Sequential(      #128,21,90
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, self.inner_channels* 2, kernel_size=3,  stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU()
        )
        self.conv_layer_encoder_1 = nn.Sequential(     #256,11,45
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels* 2, self.inner_channels* 4, kernel_size=3, stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU()
        )
        self.conv_layer_encoder_2 = nn.Sequential(     #512,6,11
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels* 4, self.inner_channels* 8, kernel_size=(3, 5), stride=(2, 4)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU()
        )
        self.lr_conv_layer_encoder = nn.Sequential(     #1024,1,6
            nn.Conv2d(self.inner_channels * 8, self.inner_channels* 16, kernel_size=(5, 3), stride=(2, 2), padding=(0, 1), padding_mode='circular'),
            nn.BatchNorm2d(self.inner_channels * 16),
            nn.LeakyReLU()
        )

        self.in_layer_decoder = nn.Sequential(     #512,6,11
            nn.ConvTranspose2d(self.inner_channels * 16, self.inner_channels* 8, kernel_size=(5, 3), stride=(2, 2), padding=(0, 1), output_padding=(1, 0)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 8, self.inner_channels* 8, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU()
        )

        self.lr_conv_layer_decoder = nn.Sequential(     #256,11,45
            nn.ConvTranspose2d(self.inner_channels * 16, self.inner_channels * 4, kernel_size=(3, 5), stride=(2, 4), padding=(1, 0), output_padding=(0, 0)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 4, self.inner_channels * 4, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU()
        )

        self.conv_layer_decoder_1 = nn.Sequential(     #128,21,90
            nn.ConvTranspose2d(self.inner_channels * 8, self.inner_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.inner_channels* 2),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels* 2, self.inner_channels* 2, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels* 2),
            nn.LeakyReLU()
        )

        self.conv_layer_decoder_2 = nn.Sequential(     #64,41,180
            nn.ConvTranspose2d(self.inner_channels * 4, self.inner_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU()
        )

        self.hr_conv_layer_decoder = nn.Sequential(     #1,81,360
            nn.ConvTranspose2d(self.inner_channels * 2, self.out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3)
        )

    def forward(self, x):
        #encoder
        features1 = self.in_layer_encoder(x)
        features2 = self.hr_conv_layer_encoder(features1)
        features3 = self.conv_layer_encoder_1(features2)
        features3_ = self.conv_layer_encoder_2(features3)
        features4 = self.lr_conv_layer_encoder(features3_)
        features5 = self.in_layer_decoder(features4)
        features6 = torch.cat([features3_, features5], dim=1)
        features7 = self.lr_conv_layer_decoder(features6)
        features8 = torch.cat([features3, features7], dim=1)
        features9 = self.conv_layer_decoder_1(features8)
        features10 = torch.cat([features2, features9], dim=1)
        features10_ = self.conv_layer_decoder_2(features10)
        features11 = torch.cat([features1, features10_], dim=1)
        features11_ = self.hr_conv_layer_decoder(features11)
        return features11_


class UNetAlternative(nn.Module):

    def __init__(self, in_channels, out_channels, inner_channels, padding_mode=None):
        super(UNetAlternative, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inner_channels = inner_channels

        self.in_layer = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.in_channels, self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU()
        )

        self.down1 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, 2 * self.inner_channels, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(2 * self.inner_channels, 2 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU(),
        )

        self.down2 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(2 * self.inner_channels, 4 * self.inner_channels, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(4 * self.inner_channels, 4 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
        )

        self.down3 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(4 * self.inner_channels, 8 * self.inner_channels, (3, 3), stride=(2, 2)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(8 * self.inner_channels, 8 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU(),
        )

        self.up1 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(8 * self.inner_channels, 8 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.inner_channels * 8, self.inner_channels * 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
        )

        self.up2 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(8 * self.inner_channels, 4 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 4),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.inner_channels * 4, self.inner_channels * 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU(),
        )

        self.up3 = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(4 * self.inner_channels, 2 * self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels * 2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(self.inner_channels * 2, self.inner_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 1)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU(),
        )

        self.out_layer = nn.Sequential(
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels * 2, self.inner_channels, (3, 3)),
            nn.BatchNorm2d(self.inner_channels),
            nn.LeakyReLU(),
            PaddingBlock2d(1, padding_mode),
            nn.Conv2d(self.inner_channels, self.out_channels, (3, 3)),
        )

    def forward(self, x):
        f1 = self.in_layer(x)
        f2 = self.down1(f1)
        f3 = self.down2(f2)
        f4 = self.down3(f3)
        f_up = self.up1(f4)
        f3 = torch.cat([f3, f_up], dim=1)
        f_up = self.up2(f3)
        f2 = torch.cat([f2, f_up], dim=1)
        f_up = self.up3(f2)
        f1 = torch.cat([f1, f_up], dim=1)
        out = self.out_layer(f1)
        return out


def _test():
    unet = UNetAlternative(5, 1, 64)
    print(unet)

    print('[INFO] Checking longitudes:')

    for lon in range(2, 361):
        a = torch.randn(1, 5, 81, lon)
        try:
            c = unet(a)
        except RuntimeError:
            continue
        else:
            if c.shape[-2:] == (81, lon):
                print(f'[INFO] {lon} is suitable longitude size!')
            else:
                raise Exception(f'[ERROR] Found shape {c.shape}. Something unexpected happened...')

    print('[INFO] Checking latitudes:')

    for lat in range(2, 82):
        a = torch.randn(1, 5, lat, 360)
        try:
            c = unet(a)
        except RuntimeError:
            continue
        else:
            if c.shape[-2:] == (lat, 360):
                print(f'[INFO] {lat} is suitable latitude size!')
            else:
                raise Exception(f'[ERROR] Found shape {c.shape}. Something unexpected happened...')


if __name__ == '__main__':
    _test()
