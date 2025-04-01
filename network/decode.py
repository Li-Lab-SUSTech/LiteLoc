import torch
import torch.nn.functional as F
from torch.nn.functional import max_pool2d
import thop
import time


class Outnet(torch.nn.Module):
    """output module"""
    def __init__(self, n_filters, pred_sig=False, pred_bg=False, pad=1, ker_size=3):
        super(Outnet, self).__init__()

        self.pred_sig = pred_sig
        self.pred_bg = pred_bg
        self.p_out1 = torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                padding=pad).cuda()
        self.p_out2 = torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()  # fu
        self.xyzi_out1 = torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                   padding=pad).cuda()
        self.xyzi_out2 = torch.nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()  # fu

        torch.nn.init.kaiming_normal_(self.p_out1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.p_out2.weight, mode='fan_in', nonlinearity='sigmoid')
        torch.nn.init.constant_(self.p_out2.bias, -6.)  # -6

        torch.nn.init.kaiming_normal_(self.xyzi_out1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(self.xyzi_out2.weight, mode='fan_in', nonlinearity='tanh')
        torch.nn.init.zeros_(self.xyzi_out2.bias)

        if self.pred_sig:
            self.xyzis_out1 = torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                        padding=pad).cuda()
            self.xyzis_out2 = torch.nn.Conv2d(in_channels=n_filters, out_channels=4, kernel_size=1, padding=0).cuda()

            torch.nn.init.kaiming_normal_(self.xyzis_out1.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.xyzis_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            torch.nn.init.zeros_(self.xyzis_out2.bias)

        if self.pred_bg:
            self.bg_out1 = torch.nn.Conv2d(in_channels=n_filters, out_channels=n_filters, kernel_size=ker_size,
                                     padding=pad).cuda()
            self.bg_out2 = torch.nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1, padding=0).cuda()

            torch.nn.init.kaiming_normal_(self.bg_out1.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_normal_(self.bg_out2.weight, mode='fan_in', nonlinearity='sigmoid')
            torch.nn.init.zeros_(self.bg_out2.bias)

    def forward(self, x):

        outputs = {}
        p = F.elu(self.p_out1(x))
        outputs['p'] = self.p_out2(p)

        xyzi = F.elu(self.xyzi_out1(x))
        outputs['xyzi'] = self.xyzi_out2(xyzi)

        if self.pred_sig:
            xyzis = F.elu(self.xyzis_out1(x))
            outputs['xyzi_sig'] = self.xyzis_out2(xyzis)

        if self.pred_bg:
            bg = F.elu(self.bg_out1(x))
            outputs['bg'] = self.bg_out2(bg)

        return outputs

class Unet(torch.nn.Module):
    """used for frame analysis module and temporal context module"""
    def __init__(self, n_inp, n_filters=64, n_stages=5, pad=1, ker_size=3):
        super(Unet, self).__init__()
        curr_N = n_filters
        self.n_stages = n_stages
        self.layer_path = torch.nn.ModuleList()

        self.layer_path.append(
            torch.nn.Conv2d(in_channels=n_inp, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        self.layer_path.append(
            torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=2, stride=2, padding=0).cuda())
            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N * 2, kernel_size=ker_size, padding=pad).cuda())
            curr_N *= 2
            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for i in range(n_stages):
            self.layer_path.append(torch.nn.UpsamplingNearest2d(scale_factor=2).cuda())
            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N // 2, kernel_size=ker_size, padding=pad).cuda())

            curr_N = curr_N // 2

            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N * 2, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())
            self.layer_path.append(
                torch.nn.Conv2d(in_channels=curr_N, out_channels=curr_N, kernel_size=ker_size, padding=pad).cuda())

        for m in self.layer_path:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):

        n_l = 0
        x_bridged = []

        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1
        x = F.elu(list(self.layer_path)[n_l](x))
        n_l += 1

        x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(3):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 2 and i < self.n_stages - 1:
                    x_bridged.append(x)

        for i in range(self.n_stages):
            for n in range(4):
                x = F.elu(list(self.layer_path)[n_l](x))
                n_l += 1
                if n == 1:
                    x = torch.cat([x, x_bridged.pop()], 1)  # concatenate
        return x

class DECODE(torch.nn.Module):
    def __init__(self, offset, factor):
        super(DECODE, self).__init__()
        self.offset = offset
        self.factor = factor
        self.local_context = True
        self.sig_pred = True
        self.psf_pred = True
        self.n_filters = 48
        self.n_inp = 3 if self.local_context else 1
        n_features = self.n_filters * self.n_inp
        self.frame_module = Unet(n_inp=1, n_filters=self.n_filters, n_stages=2)
        self.context_module = Unet(n_inp=n_features, n_filters=self.n_filters, n_stages=2)
        self.out_module = Outnet(self.n_filters, self.sig_pred, self.psf_pred, pad=1,
                                 ker_size=3)

    # def forward(self, X):
    #
    #     img_h, img_w = X.shape[-2], X.shape[-1]
    #
    #     # simple normalization
    #     scaled_x = (X - self.offset) / self.factor
    #
    #     if X.ndimension() == 3:  # when test, X.ndimension = 3
    #         scaled_x = scaled_x[:, None]
    #         fm_out = self.frame_module(scaled_x)
    #         if self.local_context:
    #             zeros = torch.zeros_like(fm_out[:1])
    #             h_t0 = fm_out
    #             h_tm1 = torch.cat([zeros, fm_out], 0)[:-1]
    #             h_tp1 = torch.cat([fm_out, zeros], 0)[1:]
    #             fm_out = torch.cat([h_tm1, h_t0, h_tp1], 1)[1:-1]
    #     else:  # when train, X.ndimension = 4
    #         fm_out = self.frame_module(scaled_x.reshape([-1, 1, img_h, img_w])) \
    #             .reshape(-1, self.n_filters * self.n_inp, img_h, img_w)
    #
    #     cm_in = fm_out
    #
    #     cm_out = self.context_module(cm_in)
    #     outputs = self.out_module.forward(cm_out)
    #
    #     if self.sig_pred:
    #         xyzi_sig = torch.sigmoid(outputs['xyzi_sig']) + 0.001
    #     else:
    #         xyzi_sig = 0.2 * torch.ones_like(outputs['xyzi'])
    #
    #     probs = torch.sigmoid(torch.clamp(outputs['p'], -16., 16.))
    #
    #     xyzi_est = outputs['xyzi']
    #     xyzi_est[:, :2] = torch.tanh(xyzi_est[:, :2])  # xy
    #     xyzi_est[:, 2] = torch.tanh(xyzi_est[:, 2])  # z
    #     xyzi_est[:, 3] = torch.sigmoid(xyzi_est[:, 3])  # ph
    #     psf_est = torch.sigmoid(outputs['bg'])[:, 0] if self.psf_pred else None
    #
    #     return probs[:, 0], xyzi_est, xyzi_sig, psf_est

    # this forward use replication method to analyze the temporal context (official DECODE implementation)
    def forward(self, X):

        img_h, img_w = X.shape[-2], X.shape[-1]

        # simple normalization
        scaled_x = (X - self.offset) / self.factor

        if X.ndimension() == 3:  # when test, X.ndimension = 3
            scaled_x = scaled_x[:, None]
            if self.local_context:
                x0 = scaled_x[:-2]
                x1 = scaled_x[1:-1]
                x2 = scaled_x[2:]
                o0 = self.frame_module(x0)
                o1 = self.frame_module(x1)
                o2 = self.frame_module(x2)
                fm_out = torch.cat([o0, o1, o2], 1)
            else:
                fm_out = self.frame_module(scaled_x)
        else:  # when train, X.ndimension = 4
            fm_out = self.frame_module(scaled_x.reshape([-1, 1, img_h, img_w])) \
                .reshape(-1, self.n_filters * self.n_inp, img_h, img_w)

        cm_in = fm_out

        cm_out = self.context_module(cm_in)
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
        psf_est = torch.sigmoid(outputs['bg'])[:, 0] if self.psf_pred else None

        return probs[:, 0], xyzi_est, xyzi_sig, psf_est

    def post_process(self, p, xyzi_est):

        xyzi_est = xyzi_est.to(torch.float32)

        p_clip = torch.where(p > 0.3, p, torch.zeros_like(p))[:, None]

        # localize maximum values within a 3x3 patch
        pool = max_pool2d(p_clip, 3, 1, padding=1)
        max_mask1 = torch.eq(p[:, None], pool).float()

        # Add probability values from the 4 adjacent pixels
        filt = torch.Tensor([[[[0, 1, 0], [1, 1, 1], [0, 1, 0]]]]).half().cuda()
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

        p_index = torch.where(p > 0.99)
        frame_index = torch.unsqueeze(p_index[0], dim=1) + 1

        x = ((xyzi_est[:, 0])[p_index] + p_index[2]).unsqueeze(1)
        y = ((xyzi_est[:, 1])[p_index] + p_index[1]).unsqueeze(1)

        z = ((xyzi_est[:, 2])[p_index]).unsqueeze(1)
        ints = ((xyzi_est[:, 3])[p_index]).unsqueeze(1)
        p = (p[p_index]).unsqueeze(1)

        molecule_array = torch.cat([frame_index, x, y, z, ints, p], dim=1)

        return molecule_array

    # def analyze(self, im):
    #
    #     p, xyzi_est, xyzi_sig, _ = self.forward(im)
    #     infer_dict = self.post_process(p, xyzi_est)
    #
    #     return infer_dict

    def get_parameter_number(self):
        print('-' * 200)
        print('Testing network parameters and multiply-accumulate operations (MACs)')
        # print(f'Total network parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)/1e6:.2f}M')

        dummy_input = torch.randn(12, 128, 128).cuda()

        macs, params = thop.profile(self, inputs=(dummy_input,))
        macs, params = thop.clever_format([macs, params], '%.3f')
        print(f'Params:{params}, MACs:{macs}, (input shape:{dummy_input.shape})')

        t0 = time.time()
        for i in range(1000):
            self.forward(dummy_input)
        print(f'Average forward time: {(time.time() - t0) / 1000:.4f} s')