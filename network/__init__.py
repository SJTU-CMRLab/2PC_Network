import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # encoder 1
        self.conv1 = self.build_conv_block(1, 64)
        self.conv2 = self.build_conv_block(64, 128)
        self.conv3 = self.build_conv_block(128, 256)
        self.conv4 = self.build_conv_block(256, 512)

        # decoder
        self.conv5 = self.build_conv_block(512 * 2 + 256 * 2, 256)
        self.conv6 = self.build_conv_block(256 + 128 * 2, 128)
        self.conv7 = self.build_conv_block(128 + 64 * 2, 64)

        # prediction layer
        self.pred = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        # downsample layer
        self.downSample = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))

    def build_conv_block(self, in_c, out_c):
        conv_block = []
        conv_block += [
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
                      padding_mode='circular')]
        conv_block += [nn.GroupNorm(32, out_c)]
        conv_block += [nn.ReLU()]
        conv_block += [
            nn.Conv3d(in_channels=out_c, out_channels=out_c, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1),
                      padding_mode='circular')]
        conv_block += [nn.GroupNorm(32, out_c)]
        conv_block += [nn.ReLU()]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # encoder 1
        C11 = self.conv1(torch.unsqueeze(x[:, 0, :, :, :], 0))
        D11 = self.downSample(C11)
        C12 = self.conv2(D11)
        D12 = self.downSample(C12)
        C13 = self.conv3(D12)
        D13 = self.downSample(C13)
        C14 = self.conv4(D13)

        # encoder 2
        C21 = self.conv1(torch.unsqueeze(x[:, 1, :, :, :], 0))
        D21 = self.downSample(C21)
        C22 = self.conv2(D21)
        D22 = self.downSample(C22)
        C23 = self.conv3(D22)
        D23 = self.downSample(C23)
        C24 = self.conv4(D23)

        # Fusion
        C4 = torch.cat((C14, C24), dim=1)
        C3 = torch.cat((C13, C23), dim=1)
        C2 = torch.cat((C12, C22), dim=1)
        C1 = torch.cat((C11, C21), dim=1)

        U3 = nn.functional.interpolate(C4, scale_factor=[1, 2, 2], mode='trilinear', align_corners=False)
        concat3 = torch.cat((C3, U3), dim=1)
        C5 = self.conv5(concat3)

        U2 = nn.functional.interpolate(C5, scale_factor=[1, 2, 2], mode='trilinear', align_corners=False)
        concat2 = torch.cat((C2, U2), dim=1)
        C6 = self.conv6(concat2)

        U1 = nn.functional.interpolate(C6, scale_factor=[1, 2, 2], mode='trilinear', align_corners=False)
        concat1 = torch.cat((C1, U1), dim=1)
        C7 = self.conv7(concat1)

        # pred
        output = self.pred(C7)

        return output
