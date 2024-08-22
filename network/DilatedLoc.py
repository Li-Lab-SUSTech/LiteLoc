import torch
import torch.nn as nn
from torch.nn.functional import interpolate,elu
import thop
import time
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

class FusedResNetBlock(torch.nn.Module):
    def __init__(self, conv, bn):
        super(FusedResNetBlock, self).__init__()
        self.fused = torch.nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=True
        )
        # Fuse the weights and bias from conv and bn
        self._fuse(conv, bn)

    def _fuse(self, conv, bn):
        # Fuse the weights
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        # self.fused.weight.copy_(torch.mm(w_bn, w_conv).view(self.fused.weight.size()))
        self.fused.weight = torch.nn.Parameter(torch.mm(w_bn, w_conv).view(self.fused.weight.size()))

        # Fuse the bias
        b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0))
        b_conv = torch.mm(w_bn, b_conv.view(-1, 1)).view(-1)
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        self.fused.bias = torch.nn.Parameter(b_conv + b_bn)

    def forward(self, x):
        return self.fused(x)

# Define the basic Conv-LeakyReLU-BN
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


# Define the basic Conv-LeakyReLU-BN
class Conv2DReLUBN_infer(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DReLUBN_infer, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        self.bn = nn.BatchNorm2d(layer_width)
        self.lrelu = nn.ReLU()
        self.fuse = FusedResNetBlock(self.conv, self.bn).cuda()

    def forward(self, x):
        out = self.fuse(x)
        out = self.lrelu(out)
        return out

class DepthwiseConv2DReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(DepthwiseConv2DReLUBN, self).__init__()
        self.depth_conv = nn.Conv2d(input_channels, input_channels, kernel_size, stride, padding, dilation, groups=input_channels)
        self.point_conv = nn.Conv2d(input_channels, layer_width, kernel_size=1, stride=stride, padding=0, dilation=dilation)
        nn.init.kaiming_normal_(self.depth_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.point_conv.weight, mode='fan_in', nonlinearity='relu')
        self.bn = nn.BatchNorm2d(layer_width)
        self.lrelu = nn.ReLU()

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        out = self.bn(out)
        out = self.lrelu(out)
        return out

# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN_new(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN_new, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.conv_ = nn.Conv2d(input_channels, layer_width, 2*dilation[0]+1, 1, padding, (1, 1))
        self.lrelu = nn.ReLU()
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.conv_(x)
        out = out1 + out2
        out = self.lrelu(out)
        out = self.bn(out)
        return out

class Conv2DLeakyBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, stride=1):
        super(Conv2DLeakyBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out
    
    
class OutnetCoordConv_fd(nn.Module):
    """output module"""
    def __init__(self, n_filters, pred_sig=False, pred_bg=False, pad=1, ker_size=3, use_coordconv=True):
        super(OutnetCoordConv_fd, self).__init__()

        self.pred_bg = pred_bg
        self.pred_sig = pred_sig
        self.use_coordconv = use_coordconv

        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()
        self.p_out2 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu
        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.xyzi_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.constant_(self.p_out2.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.zeros_(self.xyzi_out2.bias)

        self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                    padding=pad).cuda()
        self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

        nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        nn.init.zeros_(self.xyzis_out2.bias)

    def forward(self, x):

        outputs = {}

        p = F.elu(self.p_out1(x))
        outputs['p'] = self.p_out2(p)

        xyzi = F.elu(self.xyzi_out1(x))
        outputs['xyzi'] = self.xyzi_out2(xyzi)


        xyzis = F.elu(self.xyzis_out1(x))
        outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

        return outputs
    
    
def add_coord(input, field_xy, aber_map_size):
    """ concatenate global coordinate channels to the input data

    Parameters
    ----------
    input:
         tensors with shape [batchsize, channel, height, width]
    field_xy:
        [xstart, xend, ystart, yend], the global position of the input sub-area image in the big aberration map.
        should satisfies this relationship: xstart - xend + 1 = input.width
    aber_map_size:
        [sizex sizey], the size of the aberration map, sizex corresponds to column, sizey corresponds to row.
    """
    x_start = field_xy[0].float()
    y_start = field_xy[2].float()
    x_end = field_xy[1].float()
    y_end = field_xy[3].float()

    batch_size = input.size()[0]
    x_dim = input.size()[3]
    y_dim = input.size()[2]

    x_step = 1 / (aber_map_size[0] - 1)
    y_step = 1 / (aber_map_size[1] - 1)

    xx_range = torch.arange(x_start / (aber_map_size[0] - 1), x_end / (aber_map_size[0] - 1) + 1e-6, step=x_step,
                            dtype=torch.float32).repeat([y_dim, 1]).reshape([1, y_dim, x_dim])

    xx_range = xx_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    yy_range = torch.arange(y_start / (aber_map_size[1] - 1), y_end / (aber_map_size[1] - 1) + 1e-6, step=y_step,
                            dtype=torch.float32).repeat([x_dim, 1]).transpose(1, 0).reshape([1, y_dim, x_dim])

    yy_range = yy_range.repeat_interleave(repeats=batch_size, dim=0).reshape([batch_size, 1, y_dim, x_dim])

    xx_range = xx_range.cuda()
    yy_range = yy_range.cuda()

    ret = torch.cat([input, xx_range, yy_range], dim=1)

    return ret

class CoordConv(nn.Module):
    """ CoordConv class, add coordinate channels to the data,
    apply extra 2D convolution on the coordinate channels and add the result"""
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CoordConv, self).__init__()
        self.conv2d_im = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   padding=padding)
        self.conv2d_coord = nn.Conv2d(in_channels=2, out_channels=out_channels, kernel_size=1, padding=0)

    def forward(self, input, field_xy, aber_map_size):
        y = add_coord(input, field_xy, aber_map_size)
        ret_1 = self.conv2d_im(y[:, 0:-2])
        ret_2 = self.conv2d_coord(y[:, -2:])
        return ret_1 + ret_2

class UnetCoordConv(nn.Module):
    """used for frame analysis module and temporal context module"""
    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3, use_coordconv=True):
        super(UnetCoordConv, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = nn.ModuleList()
        self.use_coordconv = use_coordconv

        if self.use_coordconv:
            self.layer_path.append(
                CoordConv(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
        else:
            self.layer_path.append(
                nn.Conv2d(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_N *= 2
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_N = curr_N // 2

            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for m in self.layer_path:
            if isinstance(m, CoordConv):
                nn.init.kaiming_normal_(m.conv2d_im.weight, mode='fan_in', nonlinearity='relu')  # 初始化卷积层权重
                nn.init.kaiming_normal_(m.conv2d_coord.weight, mode='fan_in', nonlinearity='relu')
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, xy_field, aber_map_size):
        #print(aber_map_size)

        n_l = 0
        x_bridged = []

        if self.use_coordconv:
            x = F.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
        else:
            x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1
        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = F.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(4):
                if isinstance(list(self.layer_path)[n_l], CoordConv):
                    x = F.elu(list(self.layer_path)[n_l](x, xy_field, aber_map_size))
                else:
                    x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1)  # concatenate


        return x

class OutnetCoordConv(nn.Module):
    """output module"""
    def __init__(self, n_filters, pad=1, ker_size=3):
        super(OutnetCoordConv, self).__init__()

        self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()

        self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu

        self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')  # all in (0, 1)
        nn.init.constant_(self.p_out1.bias, -6.)  # -6

        nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')  # all positive
        nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')  # all in (-1, 1)
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


# class LocalizationCNN(nn.Module):
#     def __init__(self, dilation_flag):
#         super(LocalizationCNN, self).__init__()
#         self.norm = nn.BatchNorm2d(num_features=1, affine=True)
#         self.layer1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.1)
#         self.layer21 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
#         if dilation_flag:
#             self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
#             self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
#             self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
#             self.layer6 = Conv2DLeakyReLUBN(64, 64, 3, (16, 16), (16, 16), 0.2)
#         else:
#             self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
#             self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
#             self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
#             self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
#         self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
#         self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
#         self.pool = nn.MaxPool2d(2, stride=2)
#         self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
#         self.pool1 = nn.MaxPool2d(2, stride=2)
#         self.layer8 = nn.Conv2d(64, 64, kernel_size=1)
#         n_filters = 64
#         ker_size =3
#         pad =1
#         self.p_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
#                                padding=pad).cuda()
#
#         self.xyzi_out = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
#                                   padding=pad).cuda()
#         self.p_out1 = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu
#
#         self.xyzi_out1 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu
#
#         nn.init.kaiming_normal_(self.p_out.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='sigmoid')
#         nn.init.constant_(self.p_out1.bias, -6.)  # -6
#
#         nn.init.kaiming_normal_(self.xyzi_out.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='tanh')
#         nn.init.zeros_(self.xyzi_out1.bias)
#
#         self.xyzis_out1 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
#                                     padding=pad).cuda()
#         self.xyzis_out2 = nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()
#
#         nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
#         nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
#         nn.init.zeros_(self.xyzis_out2.bias)
#
#
#     def forward(self, im):
#
#         # extract multi-scale features
#         im = self.norm(im)
#         out = self.layer1(im)  # Conv2d
#         features = torch.cat((out, im), 1)
#         out2 = self.layer2(features) + out
#         features = torch.cat((out2, im), 1)
#         out21 = self.layer21(features) + out2
#         features = torch.cat((out21, im), 1)
#         out3 = self.layer3(features) + out21
#         features = torch.cat((out3, im), 1)
#         out4 = self.layer4(features) + out3
#         features = torch.cat((out4, im), 1)
#         out5 = self.layer5(features) + out4
#         features = out5
#         out = self.layer6(features) + out3 + out5
#
#         # upsample by 4 in xy
#         features = torch.cat((out, im), 1)
#         out = interpolate(features, scale_factor=2)
#         out = self.deconv1(out)
#         out = interpolate(out, scale_factor=2,mode='area')
#         out = self.deconv2(out)
#
#         # refine z and exact xy
#         out = self.pool(out)
#         out = self.layer7(out)
#         out = self.pool1(out)
#
#         x = self.layer8(out)
#
#         p = elu(self.p_out(x))
#
#         xyzi = elu(self.xyzi_out(x))
#
#         xyzis = elu(self.xyzis_out1(x))
#
#         probs = torch.sigmoid(torch.clamp(self.p_out1(p), -16., 16.))
#
#         xyzi_est = self.xyzi_out1(xyzi)
#         xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
#         xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
#         xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
#         xyzi_sig = torch.sigmoid(self.xyzis_out2(xyzis)) + 0.001
#
#         return probs[:, 0], xyzi_est, xyzi_sig

class LocalizationCNN(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DReLUBN(1, 64, 3, 1, 1)  # replace Conv2d
        self.layer2 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.layer21 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, (8, 8), (8, 8))
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16))
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            # self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 5, 2, 1, 0.2)
            # self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 7, 3, 1, 0.2)
            # self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 17, 8, 1, 0.2)
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 1, 0, 1, 0.2)
            # self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 1, 0, 1, 0.2)
            # self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            # self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            # self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            # self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            # self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 5, 2, 1, 0.2)
            # self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 5, 2, 1, 0.2)
            # self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 7, 3, 1, 0.2)
            # self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 9, 4, 1, 0.2)
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        # self.upsampling1 = nn.ConvTranspose2d(64 + 1, 64 + 1, 4, stride=2, padding=1)
        self.deconv1 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        # self.upsampling2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.deconv2 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, stride=2)
        # self.downsampling1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.layer7 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.downsampling2 = nn.Conv2d(64, 64, 3, 2, padding=1)
        self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)


    def forward(self, im):

        # extract multi-scale features
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out2 = self.layer2(features) + out

        features = torch.cat((out2, im), 1)
        out21 = self.layer21(features) + out2
        features = torch.cat((out21, im), 1)
        out3 = self.layer3(features) + out21
        features = torch.cat((out3, im), 1)
        out4 = self.layer4(features) + out3
        features = torch.cat((out4, im), 1)
        out5 = self.layer5(features) + out4

        features = torch.cat((out5, im), 1)
        out = self.layer6(features) + out3 + out5
        # out6 = self.layer6(features) + out5
        # features = torch.cat((out6, im), 1)

        # out7 = self.layer61(features) + out6
        # features = torch.cat((out7, im), 1)

        # out = self.layer62(features) + out3 + out7

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)

        out = interpolate(features, scale_factor=2)  # c=65, map=256 # default: mode='nearest'
        out = self.deconv1(out)  # c=65, map=256
        out = interpolate(out, scale_factor=2)  # c=65, map=512
        out = self.deconv2(out)  # c=64, map=512

        # refine z and exact xy
        out = self.pool(out)  # c=64, map=256
        out = self.layer7(out)  # out=64, map=256
        out = self.pool1(out)  # c=64,map=128

        out = self.layer8(out)  # c=64, map=128

        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig


class LocalizationCNN_Unet(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DReLUBN(1, 64, 3, 1, 1)  # replace Conv2d
        self.layer2 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16))
            self.layer71 = Conv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)


    def forward(self, im):

        # extract multi-scale features
        out = self.norm(im)

        out = self.layer1(out)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)

        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out4 = self.layer4(features) + out

        features = torch.cat((out4, im), 1)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)
        out = self.layer7(features) + out4 + out6
        features = torch.cat((out, im), 1)  # (65, 128, 128)

        out1 = self.deconv1(features)
        out = self.pool(out1)
        out = self.layerU1(out)
        out = self.layerU2(out)
        out = self.layerU3(out)

        out = interpolate(out, scale_factor=2)
        out = self.layerD3(out)
        out = torch.cat([out, out1], 1)
        out = self.layerD2(out)
        out = self.layerD1(out)

        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        for j in range(5):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(2000):
            with autocast():
                self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')


class LocalizationCNN_Unet_downsample_128(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_downsample_128, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        # self.pool0 = nn.MaxPool2d(2, stride=2)
        self.layer0 = Conv2DReLUBN(1, 64, 4, 1, 1, stride=2)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        # self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = Conv2DReLUBN(64, 64, 4, 1, 1, stride=2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)


    def forward(self, im):

        # extract multi-scale features
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im = self.layer0(im0)  # (10, 64, 128, 128)
        # im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        # out = self.layer30(features) + out
        # features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out4 = self.layer4(features) + out
        features = torch.cat((out4, im), 1)  # (10, 128, 64, 64)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, im), 1)  # (10, 128, 64, 64)

        out1 = self.deconv1(features)  # (10, 64, 64, 64)
        out = self.pool(out1)  # (10, 64, 32, 32)
        # out = self.deconv2(out1)
        out = self.layerU1(out)  # (10, 64, 32, 32)
        out = self.layerU2(out)  # (10, 128, 32, 32)
        out = self.layerU3(out)  # (10, 128, 32, 32)

        out = interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
        out = self.layerD3(out)  # (10, 64, 64, 64)
        out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
        out = self.layerD2(out)  # (10, 64, 64, 64)
        out = self.layerD1(out)  # (10, 64, 64, 64)
        out = interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
        # out = torch.cat((out, im0), 1)
        # out = self.layerD0(out)  # (10, 64, 128, 128)

        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(2000):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 200:.4f} s')


class LocalizationCNN_Unet_downsample_128_Unet(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_downsample_128_Unet, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        # self.pool0 = nn.MaxPool2d(2, stride=2)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        # self.layer2 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU30 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU31 = Conv2DReLUBN(64 * 2, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU32 = Conv2DReLUBN(64 * 4, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD30 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD20 = Conv2DReLUBN(64 * 4, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD10 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.layerD01 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = Conv2DReLUBN(64, 64, 4, 1, 1, stride=2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])


    def forward(self, im):

        # extract multi-scale features
        # extract multi-scale features
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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(200):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')


class LocalizationCNN_Unet_downsample_128_Unet_nodenseconnect(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_downsample_128_Unet_nodenseconnect, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        # self.pool0 = nn.MaxPool2d(2, stride=2)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        # self.layer2 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))
        self.layer2 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(64, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(64, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(64, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(64, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.deconv1 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU30 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU31 = Conv2DReLUBN(64 * 2, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU32 = Conv2DReLUBN(64 * 4, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD30 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD20 = Conv2DReLUBN(64 * 4, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD10 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.layerD01 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = Conv2DReLUBN(64, 64, 4, 1, 1, stride=2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])


    def forward(self, im):

        # extract multi-scale features
        # extract multi-scale features
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im1 = self.layer0(im0)  # (10, 64, 128, 128)
        im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer30(out)

        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)    # (10, 64, 64, 64)
        out1 = self.deconv1(out)  # (10, 64, 64, 64)

        # Unet Stage
        out = self.pool(out)  # (10, 64, 32, 32)
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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(200):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')
# class LocalizationCNN_Unet_downsample_128_Unet_local(nn.Module):
#     def __init__(self):
#         super(LocalizationCNN_Unet_downsample_128_Unet_local, self).__init__()
#         self.norm = nn.BatchNorm2d(num_features=1, affine=True)
#         self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)
#         self.pool1 = nn.MaxPool2d(2, stride=2)  # downsample the input image size
#         #self.layerpool1 = Conv2DReLUBN(64, 64, kernel_size=2, stride=2, padding=0, dilation=1)
#         self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)
#         self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
#         self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
#         # self.layer31 = Conv2DReLUBN(128, 64, 3, 1, 1)
#         self.layer300 = Conv2DReLUBN(256, 128, 3, 1, 1)
#         self.layer301 = Conv2DReLUBN(192, 64, 3, 1, 1)
#         self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)
#
#         self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
#         self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
#         self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
#         self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))
#
#         # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)  # k' = (k+1)*(dilation-1)+k
#         # self.layer5 = Conv2DReLUBN(128, 64, 3, 1, 1)  # padding' = 2*padding-1
#         # self.layer6 = Conv2DReLUBN(128, 64, 3, 1, 1)
#         # self.layer7 = Conv2DReLUBN(128, 64, 3, 1, 1)
#
#         self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
#         self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
#         self.pool = nn.AvgPool2d(2, stride=2)
#         #self.layerpool = Conv2DReLUBN(64, 64, kernel_size=2, stride=2, padding=0, dilation=1)
#         self.pred = OutnetCoordConv(64, 1, 3)
#
#         diag = 0
#         self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
#         self.p_Conv.bias = None
#         self.p_Conv.training = False
#         self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])
#
#     def forward(self, im, test=True):
#
#         img_h, img_w, batch_size = im.shape[-2], im.shape[-1], im.shape[0]
#         # if im.shape[1] > 1:
#         #     img_curr = torch.unsqueeze(im[:, 1], dim=1)  # (10, 1, 128, 128)
#         # else:
#         #     img_curr = im
#         im = im.reshape([-1, 1, img_h, img_w])
#         img_h, img_w = int(img_h / 2), int(img_w / 2)
#
#         # extract multi-scale features
#         im0 = self.norm(im)  # (30, 1, 128, 128)
#         im1 = self.layer0(im0)  # (30, 64, 128, 128)
#         im = self.pool1(im1)  # (30, 64, 64, 64)
#         #im = self.layerpool1(im1)
#
#         if im.shape[0] > batch_size:
#             img_curr = im[torch.arange(1, batch_size * 3 - 1, 3), :, :, :]
#         else:
#             img_curr = im
#
#         out = self.layer1(im)  # (30, 64, 64, 64)
#
#         features = torch.cat((out, im), 1)  # (30, 128, 64, 64)
#
#         out = self.layer2(features) + out  # (30, 64, 64, 64)
#         features = torch.cat((out, im), 1)  # (30, 128, 64, 64)
#         out_ = self.layer3(features) + out  # (30, 64, 64, 64)
#         # features = torch.cat((out_, im), 1)  # (30, 128, 64, 64)
#         # out_ = self.layer31(features) + out_  # (30, 64, 64, 64)
#
#         if test:
#             zeros = torch.zeros_like(out[:1])
#             h_t0 = out_
#             h_tm1 = torch.cat([zeros, out_], 0)[:-1]
#             h_tp1 = torch.cat([out_, zeros], 0)[1:]
#             out_ = torch.cat([h_tm1, h_t0, h_tp1], 1)
#
#         out3_tmp = out_.reshape(-1, 64 * 3, img_h, img_w)  # (10, 64*3, 64, 64)
#         features = torch.cat((out3_tmp, img_curr), 1)  # (10, 64*3+64, 64, 64)  尺寸不匹配
#         out = self.layer300(features)  # (10, 128, 64, 64)
#         features = torch.cat((out, img_curr), 1)  # (10, 64*3, 64, 64)
#         out = self.layer301(features)  # (10, 64, 64, 64)
#         features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)
#
#         if test:
#             out = self.layer30(features) + h_t0[:, :, :, :]
#         else:
#             out = self.layer30(features) + out_[torch.arange(1, batch_size * 3 - 1, 3), :, :, :]  # (10, 128, 64, 64)
#
#         features = torch.cat((out, img_curr), 1)
#         out4 = self.layer4(features) + out
#         features = torch.cat((out4, img_curr), 1)  # (10, 128, 64, 64)
#         out = self.layer5(features) + out4  # (10, 64, 64, 64)
#         features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)
#         out6 = self.layer6(features) + out  # (10, 64, 64, 64)
#         features = torch.cat((out6, img_curr), 1)  # (10, 128, 64, 64)
#         out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
#         features = torch.cat((out7, img_curr), 1)  # (10, 128, 64, 64)
#
#         out1 = self.deconv1(features)  # (10, 64, 64, 64)
#
#         # Unet Stage
#         out = self.pool(out1)  # (10, 64, 32, 32)
#         #out = self.layerpool(out1)
#         out = self.layerU1(out)  # (10, 64, 32, 32)
#         out = self.layerU2(out)  # (10, 128, 32, 32)
#         out = self.layerU3(out)  # (10, 128, 32, 32)
#
#         out = interpolate(out, scale_factor=2)  # (10, 128, 64, 64)
#         out = self.layerD3(out)  # (10, 64, 64, 64)
#         out = torch.cat([out, out1], 1)  # (10, 128, 64, 64)
#         out = self.layerD2(out)  # (10, 64, 64, 64)
#         out = self.layerD1(out)  # (10, 64, 64, 64)
#         out = interpolate(out, scale_factor=2)  # (10, 64, 128, 128)
#         out = self.layerD0(out)  # (10, 64, 128, 128)
#         out = self.layerD00(out)  # (10, 64, 128, 128)
#
#         out = self.pred(out)
#         probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))
#
#         xyzi_est = out['xyzi']
#         xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
#         xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
#         xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
#         xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
#
#         return probs[:, 0], xyzi_est, xyzi_sig
#
#
#     def get_parameter_number(self):
#         # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')
#
#         dummy_input = torch.randn(12, 1, 128, 128).cuda()
#         macs, params = thop.profile(self, inputs=(dummy_input,))
#         macs, params = thop.clever_format([macs, params], '%.3f')
#         print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')
#
#         torch.cuda.synchronize()
#         t0 = time.time()
#         for j in range(5):
#             self.forward(dummy_input)
#         for i in range(1667):
#             self.forward(dummy_input)
#         torch.cuda.synchronize()
#         print(f'Average forward time: {(time.time() - t0):.4f} s')

class LocalizationCNN_Unet_downsample_128_Unet_noresconnect(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_downsample_128_Unet_noresconnect, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        # self.pool0 = nn.MaxPool2d(2, stride=2)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        # self.layer2 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(128, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(128, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(128, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(128, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.deconv1 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU30 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU31 = Conv2DReLUBN(64 * 2, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU32 = Conv2DReLUBN(64 * 4, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD30 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD20 = Conv2DReLUBN(64 * 4, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD10 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.layerD01 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = Conv2DReLUBN(64, 64, 4, 1, 1, stride=2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])


    def forward(self, im):

        # extract multi-scale features
        # extract multi-scale features
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im1 = self.layer0(im0)  # (10, 64, 128, 128)
        im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out = self.layer2(features)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer3(features)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out = self.layer30(features)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)

        out4 = self.layer4(features)
        features = torch.cat((out4, im), 1)  # (10, 128, 64, 64)
        out = self.layer5(features)
        features = torch.cat((out, im), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features)
        features = torch.cat((out6, im), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features)  # (10, 64, 64, 64)
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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(200):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')


class LocalizationCNN_Unet_downsample_128_Unet_resconnect(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_downsample_128_Unet_resconnect, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        # self.pool0 = nn.MaxPool2d(2, stride=2)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)  # replace Conv2d
        # self.layer2 = Conv2DReLUBN(128, 64, 3, (2, 2), (2, 2))
        self.layer2 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(64, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer4 = Conv2DReLUBN(128, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = Conv2DReLUBN(64, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(64, 64, 3, (8, 8), (8, 8))
            self.layer7 = Conv2DReLUBN(64, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer5 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.deconv1 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layerU1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = Conv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU30 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU31 = Conv2DReLUBN(64 * 2, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerU32 = Conv2DReLUBN(64 * 4, 64 * 4, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD30 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD20 = Conv2DReLUBN(64 * 4, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        #self.layerD10 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD0 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD00 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.layerD01 = Conv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        # self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.deconv2 = Conv2DReLUBN(64, 64, 4, 1, 1, stride=2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])


    def forward(self, im):

        # extract multi-scale features
        # extract multi-scale features
        im0 = self.norm(im)  # (10, 1, 128, 128)
        im1 = self.layer0(im0)  # (10, 64, 128, 128)
        im = self.pool1(im1)  # (10, 64, 64, 64)

        out = self.layer1(im)  # (10, 64, 64, 64)

        out = self.layer2(out) + out
        out = self.layer3(out) + out
        out = self.layer30(out) + out

        out4 = self.layer4(out) + out
        out = self.layer5(out4) + out4
        out6 = self.layer6(out) + out
        out7 = self.layer7(out6) + out4 + out6  # (10, 64, 64, 64)

        out1 = self.deconv1(out7)  # (10, 64, 64, 64)

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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(200):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')


class LocalizationCNN_Unet_downsample_128_Unet_local(nn.Module):
    def __init__(self):
        super(LocalizationCNN_Unet_downsample_128_Unet_local, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer30 = Conv2DReLUBN(256, 64, 3, 1, 1)

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
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])

    def forward(self, im, test=True):

        img_h, img_w, batch_size = im.shape[-2], im.shape[-1], im.shape[0]
        # if im.shape[1] > 1:
        #     img_curr = torch.unsqueeze(im[:, 1], dim=1)  # (10, 1, 128, 128)
        # else:
        #     img_curr = im
        im = im.reshape([-1, 1, img_h, img_w])
        img_h, img_w = int(img_h/2), int(img_w/2)

        # extract multi-scale features
        im0 = self.norm(im)  # (30, 1, 128, 128)
        im1 = self.layer0(im0)  # (30, 64, 128, 128)
        im = self.pool1(im1)  # (30, 64, 64, 64)

        out = self.layer1(im)  # (30, 64, 64, 64)
        if im.shape[0] > batch_size:
            img_curr = out[torch.arange(1, batch_size*3-1, 3), :, :, :]
        else:
            img_curr = out
        features = torch.cat((out, im), 1)  # (30, 128, 64, 64)

        out = self.layer2(features) + out  # (30, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (30, 128, 64, 64)
        out = self.layer3(features) + out  # (30, 64, 64, 64)

        if test:
            zeros = torch.zeros_like(out[:1])
            h_t0 = out
            h_tm1 = torch.cat([zeros, out], 0)[:-1]
            h_tp1 = torch.cat([out, zeros], 0)[1:]
            out = torch.cat([h_tm1, h_t0, h_tp1], 1)
        out3_tmp = out.reshape(-1, 64*3, img_h, img_w)  # (10, 64*3, 64, 64)
        features = torch.cat((out3_tmp, img_curr), 1)  # (10, 64*3+64, 128, 128)  尺寸不匹配
        if test:
            out = self.layer30(features) + h_t0[:, :, :, :]
        else:
            out = self.layer30(features) + out[torch.arange(1, batch_size*3-1, 3), :, :, :]

        features = torch.cat((out, img_curr), 1)
        out4 = self.layer4(features) + out
        features = torch.cat((out4, img_curr), 1)  # (10, 128, 64, 64)
        out = self.layer5(features) + out4  # (10, 64, 64, 64)
        features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features) + out  # (10, 64, 64, 64)
        features = torch.cat((out6, img_curr), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, img_curr), 1)  # (10, 128, 64, 64)

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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(12, 1, 128, 128).cuda()
        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        torch.cuda.synchronize()
        t0 = time.time()
        for j in range(5):
            self.forward(dummy_input)
        for i in range(1667):
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0):.4f} s')

class LocalizationCNN_Unet_downsample_128_Unet_local_Impr(nn.Module):
    def __init__(self):
        super(LocalizationCNN_Unet_downsample_128_Unet_local_Impr, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer0 = Conv2DReLUBN(1, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # downsample the input image size
        self.layer1 = Conv2DReLUBN(64, 64, 3, 1, 1)
        self.layer2 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer3 = Conv2DReLUBN(128, 64, 3, 1, 1)
        # self.layer31 = Conv2DReLUBN(128, 64, 3, 1, 1)
        self.layer300 = Conv2DReLUBN(256, 128, 3, 1, 1)
        self.layer301 = Conv2DReLUBN(192, 64, 3, 1, 1)
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
        self.pred = OutnetCoordConv(64, 1, 3)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])

    def forward(self, im, test=True):

        img_h, img_w, batch_size = im.shape[-2], im.shape[-1], im.shape[0]
        # if im.shape[1] > 1:
        #     img_curr = torch.unsqueeze(im[:, 1], dim=1)  # (10, 1, 128, 128)
        # else:
        #     img_curr = im
        im = im.reshape([-1, 1, img_h, img_w])
        img_h, img_w = int(img_h / 2), int(img_w / 2)

        # extract multi-scale features
        im0 = self.norm(im)  # (30, 1, 128, 128)
        im1 = self.layer0(im0)  # (30, 64, 128, 128)
        im = self.pool1(im1)  # (30, 64, 64, 64)

        if im.shape[0] > batch_size:
            img_curr = im[torch.arange(1, batch_size * 3 - 1, 3), :, :, :]
        else:
            img_curr = im

        out = self.layer1(im)  # (30, 64, 64, 64)

        features = torch.cat((out, im), 1)  # (30, 128, 64, 64)

        out = self.layer2(features) + out  # (30, 64, 64, 64)
        features = torch.cat((out, im), 1)  # (30, 128, 64, 64)
        out_ = self.layer3(features) + out  # (30, 64, 64, 64)
        # features = torch.cat((out_, im), 1)  # (30, 128, 64, 64)
        # out_ = self.layer31(features) + out_  # (30, 64, 64, 64)

        if test:
            zeros = torch.zeros_like(out[:1])
            h_t0 = out_
            h_tm1 = torch.cat([zeros, out_], 0)[:-1]
            h_tp1 = torch.cat([out_, zeros], 0)[1:]
            out_ = torch.cat([h_tm1, h_t0, h_tp1], 1)

        out3_tmp = out_.reshape(-1, 64 * 3, img_h, img_w)  # (10, 64*3, 64, 64)
        features = torch.cat((out3_tmp, img_curr), 1)  # (10, 64*3+64, 64, 64)  尺寸不匹配
        out = self.layer300(features)  # (10, 128, 64, 64)
        features = torch.cat((out, img_curr), 1)  # (10, 64*3, 64, 64)
        out = self.layer301(features)  # (10, 64, 64, 64)
        features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)

        if test:
            out = self.layer30(features) + h_t0[:, :, :, :]
        else:
            out = self.layer30(features) + out_[torch.arange(1, batch_size * 3 - 1, 3), :, :, :]  # (10, 128, 64, 64)

        features = torch.cat((out, img_curr), 1)
        out4 = self.layer4(features) + out
        features = torch.cat((out4, img_curr), 1)  # (10, 128, 64, 64)
        out = self.layer5(features) + out4  # (10, 64, 64, 64)
        features = torch.cat((out, img_curr), 1)  # (10, 128, 64, 64)
        out6 = self.layer6(features) + out  # (10, 64, 64, 64)
        features = torch.cat((out6, img_curr), 1)  # (10, 128, 64, 64)
        out7 = self.layer7(features) + out4 + out6  # (10, 64, 64, 64)
        features = torch.cat((out7, img_curr), 1)  # (10, 128, 64, 64)

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

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(10, 1, 128, 128).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        self.eval()
        self.width = torch.arange(0, 128, step=1,
                                  dtype=torch.float32).repeat([10, 128]).reshape(
            [10, 128, 128]).to('cuda')
        self.height = torch.transpose(self.width, 1, 2)
        torch.cuda.synchronize()
        t0 = time.time()
        index = 0
        for i in range(2000):
            # dummy_input = torch.randn(1, 1, 256, 1024).cuda()
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 1:.4f} s')

class LocalizationCNN_Unet_depthwise(nn.Module):
    def __init__(self, dilation_flag):
        super(LocalizationCNN_Unet_depthwise, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = DepthwiseConv2DReLUBN(1, 64, 3, 1, 1)  # replace Conv2d
        self.layer2 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.layer3 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1)
        if dilation_flag:
            # self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = Conv2DReLUBN(64 + 1, 64, 3, (2, 2), (2, 2))  # k' = (k+1)*(dilation-1)+k
            self.layer5 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, (4, 4), (4, 4))  # padding' = 2*padding-1
            self.layer6 = Conv2DReLUBN(64 + 1, 64, 3, (8, 8), (8, 8))
            self.layer7 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16))
            self.layer71 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, (16, 16), (16, 16))
        else:
            self.layer3 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer4 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer5 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = DepthwiseConv2DReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DReLUBN(64 + 1, 64, 3, 1, 1)
        self.layerU1 = DepthwiseConv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU2 = DepthwiseConv2DReLUBN(64, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerU3 = Conv2DReLUBN(64 * 2, 64 * 2, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD3 = Conv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD2 = DepthwiseConv2DReLUBN(64 * 2, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.layerD1 = DepthwiseConv2DReLUBN(64, 64, kernel_size=3, stride=1, padding=1, dilation=1)
        self.norm1 = nn.BatchNorm2d(num_features=64 * 2, affine=True)
        # self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.pool = nn.AvgPool2d(2, stride=2)
        # self.layer7 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        # self.pool1 = nn.MaxPool2d(2, stride=2)
        # self.layer8 = nn.Conv2d(64, 64, kernel_size=1, dilation=1)
        self.pred = OutnetCoordConv(64, 1, 3)


    def forward(self, im):

        # extract multi-scale features
        out = self.norm(im)

        out = self.layer1(out)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)

        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out4 = self.layer4(features) + out

        features = torch.cat((out4, im), 1)
        out = self.layer5(features) + out4
        features = torch.cat((out, im), 1)
        out6 = self.layer6(features) + out
        features = torch.cat((out6, im), 1)
        out = self.layer7(features) + out4 + out6
        features = torch.cat((out, im), 1)

        out1 = self.deconv1(features)
        out = self.pool(out1)
        out = self.layerU1(out)
        out = self.layerU2(out)
        out = self.layerU3(out)

        out = interpolate(out, scale_factor=2)
        out = self.layerD3(out)
        out = torch.cat([out, out1], 1)
        out = self.layerD2(out)
        out = self.layerD1(out)

        out = self.pred(out)
        probs = torch.sigmoid(torch.clamp(out['p'], -16., 16.))

        xyzi_est = out['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # photon
        xyzi_sig = torch.sigmoid(out['xyzi_sig']) + 0.001
        return probs[:, 0], xyzi_est, xyzi_sig


class FdDeeploc(nn.Module):
    def __init__(self, net_pars, feature=64):
        super(FdDeeploc, self).__init__()
        self.net_pars = net_pars

        self.local_context = net_pars['local_flag']
        self.sig_pred = net_pars['sig_pred']
        self.psf_pred = net_pars['psf_pred']
        self.n_filters = net_pars['n_filters']
        # self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

        self.n_inp = 3 if self.local_context else 1
        n_features = self.n_filters * self.n_inp
        self.frame_module = UnetCoordConv(n_inp=1, n_filters=self.n_filters, n_stages=2,
                                          use_coordconv=self.net_pars['use_coordconv'])
        self.context_module = UnetCoordConv(n_inp=n_features, n_filters=self.n_filters, n_stages=2,
                                            use_coordconv=self.net_pars['use_coordconv'])
        self.out_module = OutnetCoordConv_fd(self.n_filters, self.sig_pred, self.psf_pred)

        diag = 0
        self.p_Conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.p_Conv.bias = None
        self.p_Conv.training = False
        self.p_Conv.weight.data = torch.Tensor([[[[diag, 1, diag], [1, 1, 1], [diag, 1, diag]]]])




    def forward(self, X, test=False, index = -1, coord=[], field_xy=[0,0],aber_map_size=[0,0]):

        # extract multi-scale features
        # tt =time.time()
        img_h, img_w, batch_size = X.shape[-2], X.shape[-1], X.shape[0]

        # simple normalization
        scaled_x = (X - self.net_pars['offset']) / self.net_pars['factor']

        if X.ndimension() == 3:  # when test, X.ndimension = 3
            scaled_x = scaled_x[:, None]
            fm_out = self.frame_module(scaled_x, field_xy, aber_map_size)
            if self.local_context:
                zeros = torch.zeros_like(fm_out[:1])
                h_t0 = fm_out
                h_tm1 = torch.cat([zeros, fm_out], 0)[:-1]
                h_tp1 = torch.cat([fm_out, zeros], 0)[1:]
                fm_out = torch.cat([h_tm1, h_t0, h_tp1], 1)
        else:  # when train, X.ndimension = 4
            fm_out = self.frame_module(scaled_x.reshape([-1, 1, img_h, img_w]), field_xy, aber_map_size) \
                .reshape(-1, self.n_filters * self.n_inp, img_h, img_w)

        # cm_in = fm_out

        # layernorm
        fm_out_LN = nn.functional.layer_norm(fm_out, normalized_shape=[self.n_filters * self.n_inp, img_h, img_w])
        cm_in = fm_out_LN

        cm_out = self.context_module(cm_in, field_xy, aber_map_size)
        outputs = self.out_module.forward(cm_out)

        if self.sig_pred:
            xyzi_sig = torch.sigmoid(outputs['xyzi_sig']) + 0.001
        else:
            xyzi_sig = 0.2 * torch.ones_like(outputs['xyzi'])

        probs = torch.sigmoid(torch.clamp(outputs['p'], -16., 16.))

        xyzi_est = outputs['xyzi']
        xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
        xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
        xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
        xyzi_sig = torch.sigmoid(outputs['xyzi_sig']) + 0.001

        return probs[:, 0], xyzi_est, xyzi_sig

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(40, 1, 128, 128).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        self.eval()
        self.width = torch.arange(0, 128, step=1,
                                  dtype=torch.float32).repeat([10, 128]).reshape(
            [10, 128, 128]).to('cuda')
        self.height = torch.transpose(self.width, 1, 2)
        torch.cuda.synchronize()
        t0 = time.time()
        index = 0
        for i in range(5000):
            # dummy_input = torch.randn(1, 1, 256, 1024).cuda()
            self.forward(dummy_input, test=False, index=index+1, coord=[25, 25])
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 1:.4f} s')


# Define the basic Conv-LeakyReLU-BN
class Conv2DLeakyReLUBN(nn.Module):
    def __init__(self, input_channels, layer_width, kernel_size, padding, dilation, negative_slope):
        super(Conv2DLeakyReLUBN, self).__init__()
        self.conv = nn.Conv2d(input_channels, layer_width, kernel_size, 1, padding, dilation)
        self.lrelu = nn.LeakyReLU(negative_slope, inplace=True)
        self.bn = nn.BatchNorm2d(layer_width)

    def forward(self, x):
        out = self.conv(x)
        out = self.lrelu(out)
        out = self.bn(out)
        return out


# Localization architecture
class LocalizationCNN_3D(nn.Module):
    def __init__(self, setup_params):
        super(LocalizationCNN_3D, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=1, affine=True)
        self.layer1 = Conv2DLeakyReLUBN(1, 64, 3, 1, 1, 0.2)
        self.layer2 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        if setup_params['dilation_flag']:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (8, 8), (8, 8), 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (16, 16), (16, 16), 0.2)
        else:
            self.layer3 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (2, 2), (2, 2), 0.2)
            self.layer4 = Conv2DLeakyReLUBN(64 + 1, 64, 3, (4, 4), (4, 4), 0.2)
            self.layer5 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
            self.layer6 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv1 = Conv2DLeakyReLUBN(64 + 1, 64, 3, 1, 1, 0.2)
        self.deconv2 = Conv2DLeakyReLUBN(64, 64, 3, 1, 1, 0.2)
        self.layer7 = Conv2DLeakyReLUBN(64, setup_params['D'], 3, 1, 1, 0.2)
        self.layer8 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer9 = Conv2DLeakyReLUBN(setup_params['D'], setup_params['D'], 3, 1, 1, 0.2)
        self.layer10 = nn.Conv2d(setup_params['D'], setup_params['D'], kernel_size=1, dilation=1)
        self.pred = nn.Hardtanh(min_val=0.0, max_val=setup_params['scaling_factor'])

    def forward(self, im):

        # extract multi-scale features
        im = (im - im.mean()) / im.max()
        im = self.norm(im)
        out = self.layer1(im)
        features = torch.cat((out, im), 1)
        out = self.layer2(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer3(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer4(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer5(features) + out
        features = torch.cat((out, im), 1)
        out = self.layer6(features) + out

        # upsample by 4 in xy
        features = torch.cat((out, im), 1)
        out = interpolate(features, scale_factor=2)
        out = self.deconv1(out)
        out = interpolate(out, scale_factor=2)
        out = self.deconv2(out)

        # refine z and exact xy
        out = self.layer7(out)
        out = self.layer8(out) + out
        out = self.layer9(out) + out

        # 1x1 conv and hardtanh for final result
        out = self.layer10(out)
        out = self.pred(out)
        return out

    def get_parameter_number(self):
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(1, 1, 128, 128).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        self.eval()
        self.width = torch.arange(0, 128, step=1,
                                  dtype=torch.float32).repeat([10, 128]).reshape(
            [10, 128, 128]).to('cuda')
        self.height = torch.transpose(self.width, 1, 2)
        torch.cuda.synchronize()
        t0 = time.time()
        index = 0
        for i in range(2000):
            # dummy_input = torch.randn(1, 1, 256, 1024).cuda()
            self.forward(dummy_input)
        torch.cuda.synchronize()
        print(f'Average forward time: {(time.time() - t0) / 1:.4f} s')