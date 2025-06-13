import torch
import torch.nn as nn
from torch.nn.functional import interpolate, elu, max_pool2d
import thop
import time


class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.lrelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)

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


class LiteLoc(nn.Module):
    def __init__(self):
        super(LiteLoc, self).__init__()

        # Initialization
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.initial_1 = Conv2DReLUBN(1, 64, 3, 1, 1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.initial_2 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.initial_3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.initial_4 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.initial_5 = Conv2DReLUBN(256, 64, 3, 1, 1)

        # Coarse feature extractor.
        self.cfe_1 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
        self.cfe_2 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
        self.cfe_3 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
        self.cfe_4 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))

        # Fine feature extractor.
        self.average_pool = nn.AvgPool2d(2, stride=2)
        self.unet_d1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.unet_d2 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.unet_d3 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.unet_d4 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.unet_u1 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.unet_u2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.unet_u3 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)

        # Output module.
        self.out_1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.out_2 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.out_3 = OutNet(64, 1, 3) # pred

    def forward(self, im, test=True):
        img_h, img_w = im.shape[-2], im.shape[-1]
        im = im.reshape([-1, 1, img_h, img_w])
        img_h, img_w = int(img_h/2), int(img_w/2)

        # Initialization.
        im0 = self.norm(im)  # (12, 1, 128, 128)
        im1 = self.initial_1.forward(im0)  # (12, 64, 128, 128)
        im = self.max_pool(im1)  # (12, 64, 64, 64)

        if test:
            img_curr = im[1:-1]  # (10, 64, 64, 64)
        else:
            img_curr = im[torch.arange(1, im.shape[0]-1, 3)]

        out = self.initial_2.forward(im)  # (12, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (12, 128, 64, 64)
        out = self.initial_3.forward(features) + out  # (12, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (12, 128, 64, 64)
        out = self.initial_4.forward(features) + out  # (12, 64, 64, 64)

        if test:
            zeros = torch.zeros_like(out[:1])
            h_t0 = out
            h_tm1 = torch.cat([zeros, out], 0)[:-1]
            h_tp1 = torch.cat([out, zeros], 0)[1:]
            out = torch.cat([h_tm1, h_t0, h_tp1], 1)[1:-1]
            out3_tmp = out.reshape(-1, 64*3, img_h, img_w)  # (10, 64*3, 64, 64)
            features = torch.cat((out3_tmp, img_curr), 1)  # (10, 64*3+64, 128, 128)
            out = self.initial_5.forward(features) + h_t0[1:-1]
        else:
            out3_tmp = out.reshape(-1, 64*3, img_h, img_w)  # (10, 64*3, 64, 64)
            features = torch.cat((out3_tmp, img_curr), 1)  # (10, 64*3+64, 128, 128)
            out = self.initial_5.forward(features) + out[torch.arange(1, im.shape[0]-1, 3)]

        # Coarse feature extractor.
        features = torch.cat((out, img_curr), 1)
        out4 = self.cfe_1.forward(features) + out
        features = torch.cat((out4, img_curr), 1)  # (10, 128, 64, 64)
        out = self.cfe_2.forward(features) + out4  # (10, 64, 64, 64)
        features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)
        out6 = self.cfe_3.forward(features) + out  # (10, 64, 64, 64)
        features = torch.cat((out6, img_curr), 1)  # (10, 128, 64, 64)
        out7 = self.cfe_4.forward(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, img_curr), 1)  # (10, 128, 64, 64)

        # Fine feature extractor.
        out1 = self.unet_d1.forward(features)  # (10, 64, 64, 64)
        out = self.average_pool(out1)  # (10, 64, 32, 32)
        out = self.unet_d2.forward(out)  # (10, 64, 32, 32)
        out = self.unet_d3.forward(out)  # (10, 128, 32, 32)
        out = self.unet_d4.forward(out)  # (10, 128, 32, 32)
        out = interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
        out = self.unet_u1.forward(out)  # (10, 64, 64, 64)
        out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
        out = self.unet_u2.forward(out)  # (10, 64, 64, 64)
        out = self.unet_u3.forward(out)  # (10, 64, 64, 64)

        # output module
        out = interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
        out = self.out_1.forward(out)  # (10, 64, 128, 128)
        out = self.out_2.forward(out)  # (10, 64, 128, 128)
        out = self.out_3.forward(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001  # uncertainty

        return probs[:, 0], xyzi_est, xyzi_sig

    def post_process(self, p, xyzi_est):

        xyzi_est = xyzi_est.to(torch.float32)

        p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]

        # localize maximum values within a 3x3 patch
        pool = max_pool2d(p_clip, 3, 1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        # Add probability values from the 4 adjacent pixels
        filt = torch.Tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]).half().to(p.device)#.cuda()  # maybe half tensor affect the precision of result
        conv = torch.nn.functional.conv2d(p[:, None], filt, padding=1, bias=None)
        p_ps1 = max_mask1 * conv

        # In order do be able to identify two fluorophores in adjacent pixels we look for probability values > 0.6 that are not part of the first mask
        p_copy = p * (1 - max_mask1[:, 0])

        max_mask2 = torch.where(p_copy > 0.6, torch.ones_like(p_copy), torch.zeros_like(p_copy))[:, None]
        p_ps2 = max_mask2 * conv

        p = p_ps1 + p_ps2
        p = p[:, 0]

        xyzi_est[:, 0] += 0.5
        xyzi_est[:, 1] += 0.5

        p_index = torch.where(p > 0.7)
        frame_index = torch.unsqueeze(p_index[0], dim=1) + 1

        x = ((xyzi_est[:, 0])[p_index] + p_index[2]).unsqueeze(1)
        y = ((xyzi_est[:, 1])[p_index] + p_index[1]).unsqueeze(1)

        z = ((xyzi_est[:, 2])[p_index]).unsqueeze(1)
        ints = ((xyzi_est[:, 3])[p_index]).unsqueeze(1)
        p = (p[p_index]).unsqueeze(1)

        molecule_array = torch.cat([frame_index, x, y, z, ints, p], dim=1)

        return molecule_array

    def get_parameter_number(self):
        print('-' * 200)
        print('Testing network parameters and multiply-accumulate operations (MACs)')
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(12, 128, 128).to(next(self.parameters()).device)#.cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(1000):
            self.forward(dummy_input, test=True)
        print(f'Average forward time: {(time.time() - t0) / 1000:.4f} s')


