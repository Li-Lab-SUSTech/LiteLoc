import torch
import torch.nn as nn
from torch.nn.functional import interpolate, elu, max_pool2d
import torch.nn.functional as F


class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.lrelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)
        # self.fuse = FusedResNetBlock(self.conv, self.bn).cuda()

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)  # revert the order of bn and relu
        return out


class OutNet(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutNet, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad)

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad)
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0) # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0)  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad)
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0)

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)

    def forward(self, x):

        outputs = {}
        p = elu(self.p_out(x))
        outputs['p'] = self.p_out1(p)

        xyzi =elu(self.xyzi_out(x))
        outputs['xyzi'] = self.xyzi_out1(xyzi)
        xyzis = elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)
        return outputs

class LiteLoc_wo_local(nn.Module):
    def __init__(self):
        super(LiteLoc_wo_local, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
        self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
        self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
        self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))
        self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool = nn.AvgPool2d(2, stride=2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.pred = OutNet(64, 1, 3)

    def forward(self, im):

        img_h, img_w = im.shape[-2], im.shape[-1]
        im = im.reshape([-1, 1, img_h, img_w])
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im1 = self.layer0(im0)  # (10, 64, 128, 128)
        im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer30(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out4 = self.layer4(features) + out
        features = torch.cat((out4, im), 1)  # (10, 128, 64, 64)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, im), 1)  # (10, 128, 64, 64)

        out1 = self.deconv1(features)  # (10, 64, 64, 64)

        # Unet Stage
        out = self.pool(out1)  # (10, 64, 32, 32)
        out = self.layerU1(out)  # (10, 64, 32, 32)
        out = self.layerU2(out)  # (10, 128, 32, 32)
        out = self.layerU3(out)  # (10, 128, 32, 32)

        out = interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
        out = self.layerD3(out)  # (10, 64, 64, 64)
        out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
        out = self.layerD2(out)  # (10, 64, 64, 64)
        out = self.layerD1(out)  # (10, 64, 64, 64)
        out = interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
        out = self.layerD0(out)  # (10, 64, 128, 128)
        out = self.layerD00(out)  # (10, 64, 128, 128)

        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig


    def post_process(self, p, xyzi_est):

        xyzi_est = xyzi_est.to(torch.float32)

        p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]

        # localize maximum values within a 3x3 patch
        pool = max_pool2d(p_clip, 3, 1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        # Add probability values from the 4 adjacent pixels
        filt = torch.Tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]).float().to(p.device)#half().cuda()
        conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1, bias=None)
        p_ps1 = max_mask1 * conv

        # In order do be able to identify two fluorophores in adjacent pixels we look for probablity values > 0.6 that are not part of the first mask
        p_copy = p * (1 - max_mask1[:, 0])

        # p_clip = torch.where(p_copy > 0.6, p_copy, torch.zeros_like(p_copy))[:, None]  # fushuang
        max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]  # fushuang
        p_ps2 = max_mask2 * conv

        p = p_ps1 + p_ps2
        p = p[:, 0]

        xyzi_est[:, 0] += 0.5
        xyzi_est[:, 1] += 0.5

        p_index = torch.where(p > 0.3)
        frame_index = torch.unsqueeze(p_index[0], dim=1) + 1

        x = ((xyzi_est[:, 0])[p_index] + p_index[2]).unsqueeze(1)
        y = ((xyzi_est[:, 1])[p_index] + p_index[1]).unsqueeze(1)

        z = ((xyzi_est[:, 2])[p_index]).unsqueeze(1)
        ints = ((xyzi_est[:, 3])[p_index]).unsqueeze(1)
        p = (p[p_index]).unsqueeze(1)

        molecule_array = torch.cat([frame_index, x, y, z, ints, p], dim=1)

        return molecule_array

    def analyze(self, im):

        p, xyzi_est, xyzi_sig = self.forward(im)
        infer_dict = self.post_process(p, xyzi_est)

        return infer_dict