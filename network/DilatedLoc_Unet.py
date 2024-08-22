import torch
import torch.nn as nn
from torch.nn.functional import interpolate,elu



# Define the basic Conv-LeakyReLU-BN
class Conv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation,  stride=1):
        super(Conv2DReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.bn(out)
        return out
class Conv2DReLULN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation,  stride=1):
        super(Conv2DReLULN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        #nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        #self.ln = nn.LayerNorm([out.shape[-3],out.shape[-2],out.shape[-1]],1e-5,False).cuda()
        #out = self.ln(out)
        return out


class OutnetCoordConv(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutnetCoordConv, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.zeros_(self.xyzi_out1.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

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




class LocalizationCNN(nn.Module):
    def __init__(self, dilation_flag, local_context,feature = 64):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DReLUBN(1, feature, 3, 1, 1)  # replace Conv2d
        self.layer2 = Conv2DReLUBN(feature+1, feature, 3, 1, 1)
        #
        # self.layer21 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)
        # if dilation_flag:
        #     self.layer3 = Conv2DReLUBN(featurecccc + 1, feature, 3, (2, 2), (2, 2))
        #     self.layer4 = Conv2DReLUBN(feature + 1, feature, 3, (4, 4), (4, 4))
        #     self.layer5 = Conv2DReLUBN(feature + 1, feature, 3, (8, 8), (8, 8))
        #     self.layer6 = Conv2DReLUBN(feature + 1, feature, 3, (16, 16), (16, 16))
        # else:
        #     self.layer3 = Conv2DReLUBN(feature + 1, feature, 3, (2, 2), (2, 2))
        #     # self.layer4 = Conv2DLeakyReLUBN(feature + 1, feature, 3, (4, 4), (4, 4))
        #     # self.layer5 = Conv2DLeakyReLUBN(featurec + 1, featurec, 3, (4, 4), (4, 4), 0.2)
        #     # self.layer6 = Conv2DLeakyReLUBN(featurec + 1, featurec, 3, 1, 1, 0.2)
        # self.deconv1 = Conv2DReLUBN(featurec + 1, featurecc, 3, 1, 1)
        # self.deconv2 = Conv2DReLUBN(featurecc, feature, 3, 1, 1)
        # self.pool = nn.MaxPool2d(2, stride=2)
        # self.layer7 = Conv2DReLUBN(feature, feature, 3, 1, 1)
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(feature, feature, kernel_size=1, dilation=1)
        # self.pred = OutnetCoordConv(feature, 1, 3)
        self.layer3 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)
        self.local = local_context
        if dilation_flag:
            if local_context:
                self.layer4 = Conv2DReLUBN(feature * 3 + 1, feature, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            else:
                self.layer4 = Conv2DReLUBN(feature + 1, feature, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k

            self.layer5 = Conv2DReLUBN(feature + 1, feature, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(feature + 1, feature, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(feature + 1, feature, 3, (16, 16), (16, 16))
            self.layer71 = Conv2DReLUBN(feature+ 1, feature, 3, (16, 16), (16, 16))

        else:
            self.layer4 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)
            self.layer5 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)
            self.layer6 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)
            self.layer7 = Conv2DReLUBN(feature + 1, feature, 3, 1, 1)

        self.deconv1 = Conv2DReLUBN(feature + 1, feature, kernel_size=3, padding=1, dilation=1)
        self.layerU1 = Conv2DReLUBN(feature, feature, kernel_size=3, stride=1,padding=1,dilation=1)
        self.layerU2 = Conv2DReLUBN(feature, feature*2, kernel_size=3, stride=1, padding=1,dilation=1)
        self.layerU3 = Conv2DReLUBN(feature*2, feature*2, kernel_size=3, stride=1,padding=1,dilation=1)
        self.layerD3 = Conv2DReLUBN(feature*2, feature, kernel_size=3, padding=1,dilation=1)
        self.layerD2 = Conv2DReLUBN(feature*2, feature, kernel_size=3, padding=1,dilation=1)
        self.layerD1 = Conv2DReLUBN(feature, feature, kernel_size=3, padding=1,dilation=1)
        self.norm1 = nn.BatchNorm2d(num_features=feature*2, affine=True)

        # self.deconv2 = Conv2DReLUBN(feature, feature, 3, 1, 1)
        self.pool = nn.AvgPool2d(2,stride=2)
        # self.layer8 = Conv2DReLUBN(feature , feature, 3, 1, 1)
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer9 = Conv2DReLUBN(feature, feature, 3, 1, 1)
        # self.layer10 = nn.Conv2d(feature, feature, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(feature, 1, 3)



    def forward(self, im, test=False):

        # extract multi-scale features
        img_h, img_w, batch_size = im.shape[-2], im.shape[-1], im.shape[0]
        if im.shape[1] > 1: # train, 取中间帧
            img_curr = torch.unsqueeze(im[:, 1], dim=1)
        else:  # test
            img_curr = im
        im = im.reshape([-1, 1, img_h, img_w])

        #im = interpolate(im, scale_factor=2)  # default: mode='nearest'
        out = self.norm(im)
        # out = self.layer1(im)
        # features = torch.cat((out, im), 1)
        # out2 = self.layer2(features) + out
        #
        # features = torch.cat((out2, im), 1)
        # out21 = self.layer21(features) + out2
        # features = torch.cat((out21, im), 1)
        # out3 = self.layer3(features) + out21
        # features = torch.cat((out3, im), 1)
        # out4 = self.layer4(features) + out3
        # features = torch.cat((out4, im), 1)
        # out5 = self.layer5(features) + out4
        #
        # features = torch.cat((out5, im), 1)
        # out = self.layer6(features) + out3 + out5
        #
        # # upsample by 4 in xy
        # features = torch.cat((out, im), 1)
        # out = interpolate(features, scale_factor=2)
        # out = self.deconv1(out)
        # out = interpolate(out, scale_factor=2)
        # out = self.deconv2(out)
        #
        # # refine z and exact xy
        # out = self.pool(out)
        # out = self.layer7(out)
        # out = self.pool1(out)
        #
        # out = self.layer8(out)
        #
        # out = self.pred(out)
        # probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))
        #
        # xyzi_est = out['xyzi']
        # xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        # xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        # xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
        # xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        # return probs[:, 0], xyzi_est, xyzi_sig


        out = self.layer1(out)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        if self.local:
            if test:
                zeros = torch.zeros_like(out[:1])
                h_t0 = out
                h_tm1 = torch.cat([zeros, out], 0)[:-1]
                h_tp1 = torch.cat([out, zeros], 0)[1:]
                out3 = torch.cat([h_tm1, h_t0, h_tp1], 1)

            out3_temp = out.reshape(-1, 64 * 3, img_h, img_w)  # (10, 192, 128, 128)
            features = torch.cat((out3_temp, img_curr), 1)
            if test:
                out4 = self.layer4(features) + h_t0[:, :, :, :]
            else:
                out4 = self.layer4(features) + out[batch_size:2*batch_size, :, :, :]

        else:
            features = torch.cat((out, im), 1)
            out4 = self.layer4(features) + out

        features = torch.cat((out4, im), 1)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)
        out = self.layer7(features) + out4 + out6
        features = torch.cat((out, im), 1)

        #out = self.layer71(features) + out
        #out = interpolate(features, scale_factor=2)  # default: mode='nearest'
        out1 = self.deconv1(features)
        # #out = interpolate(out, scale_factor=2)
        out=self.pool(out1)

        out = self.layerU1(out)
        out = self.layerU2(out)
        out = self.layerU3(out)
        out = interpolate(out, scale_factor=2)
        out = self.layerD3(out)
        out = torch.cat([out,out1],1)
        #out = self.norm1(out)
        out = self.layerD2(out)
        out = self.layerD1(out)
        # out = self.deconv2(out)  # 256
        #
        # # refine z and exact xy
        # #out = self.pool(out)
        # out = self.layer8(out)
        # out = self.pool1(out)

        #out = self.layer9(out)

        #upsample
        #features = torch.cat((out, im), 1)
        # features = interpolate(features, scale_factor=2)
        # out = self.deconv1(features)  # 256
        # #out = interpolate(out, scale_factor=2)
        #
        # out = self.layer8(out)
        #
        # # refine z and exact xy
        #
        #
        # out = self.pool(self.layer9(out))
        # out = self.deconv2(out)
        #
        #
        # out = self.layer10(out)

        # output module
        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig
