import torch
import torch.nn as nn
import numpy as np


class LossFuncs:

    def __init__(self, train_size):
        self.train_size = train_size

    # def eval_P_locs_loss(self, P, locs):
    #     loss_cse = -(1 - P) ** 2 * locs * torch.log(P) - P ** 2 * (1 - locs) * torch.log(1 - P)
    #     loss_cse = loss_cse.sum(-1).sum(-1)
    #     return loss_cse
    # cross-entropy--compare difference between two distributions
    def eval_P_locs_loss(self, P, locs):
        loss_cse = - 0.65 * locs * torch.log(P) - 0.35 * (1 - locs) * torch.log(1 - P)
        # loss_cse = -(locs * torch.log(P) + (1 - locs) * torch.log(1 - P))
        loss_cse = loss_cse.sum(-1).sum(-1)
        return loss_cse

    # encourage a detection probability map with sparse but confident predictions.
    def count_loss_analytical(self, P, s_mask):  # s_mask(molecule list): 1-molecule, 0-no molecule
        log_prob = 0
        prob_mean = P.sum(-1).sum(-1)
        prob_var = (P - P ** 2).sum(-1).sum(-1)
        X = s_mask.sum(-1)  # 每帧有多少个molecule
        log_prob += 1 / 2 * ((X - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        return log_prob
    # simultaneously optimize the probability of detection, subpixel localization and standard deviation.
    def loc_loss_analytical(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask):
        # each pixel is a component of Gaussian Mixture Model, with weights prob_normed
        # normalize P in all frames
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        # ensure p in every pixel is greater than zero
        p_inds = tuple((P + 1).nonzero().transpose(1, 0))

        xyzi_mu = xyzi_est[p_inds[0], :, p_inds[1], p_inds[2]]
        xyzi_mu[:, 0] += p_inds[2].type(torch.cuda.FloatTensor) + 0.5  # recover exact coordinate from sub-pixel
        xyzi_mu[:, 1] += p_inds[1].type(torch.cuda.FloatTensor) + 0.5

        xyzi_mu = xyzi_mu.reshape(P.shape[0], 1, -1, 4)
        xyzi_sig = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(P.shape[0], 1, -1, 4)  # >=0.01
        # xyzi_lnsig2 = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        XYZI = xyzi_gt.reshape(P.shape[0], -1, 1, 4).repeat_interleave(self.train_size * self.train_size, 2)

        numerator = -1 / 2 * (abs(XYZI - xyzi_mu))
        denominator = (xyzi_sig ** 2)  # >0
        # denominator = torch.exp(xyzi_lnsig2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(P.shape[0], 1, self.train_size * self.train_size)
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)

        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)  # weights--probability of detection
        # c = torch.sum(p_gauss_4d * gauss_coef,-1)
        # gmm_log = (torch.log(c) * s_mask).sum(-1)
        return (gmm_log * s_mask).sum(-1)

    def final_loss(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask, locs):
        count_loss = torch.mean(self.count_loss_analytical(P, s_mask) * s_mask.sum(-1))
        loc_loss = -torch.mean(self.loc_loss_analytical(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask))
        P_locs_error = torch.mean(self.eval_P_locs_loss(P, locs)) if locs is not None else 0

        loss_total = count_loss + loc_loss + P_locs_error
        # loss_total = count_loss

        return loss_total


class LossFuncs_decode:

    def __init__(self, train_size):
        self.train_size = train_size

    def eval_bg_sq_loss(self, psf_imgs_est, psf_imgs_gt):
        loss = nn.MSELoss(reduction='none')
        cost = loss(psf_imgs_est, psf_imgs_gt)
        cost = cost.sum(-1).sum(-1)
        return cost

    # cross-entropy--compare difference between two distributions
    def eval_P_locs_loss(self, P, locs):
        loss_cse = -(locs * torch.log(P) + (1 - locs) * torch.log(1 - P))
        loss_cse = loss_cse.sum(-1).sum(-1)
        return loss_cse

    # encourage a detection probability map with sparse but confident predictions.
    def count_loss_analytical(self, P, s_mask):  # s_mask(molecule list): 1-molecule, 0-no molecule
        log_prob = 0
        prob_mean = P.sum(-1).sum(-1)
        prob_var = (P - P ** 2).sum(-1).sum(-1)
        X = s_mask.sum(-1)  # 每帧有多少个molecule
        log_prob += 1 / 2 * ((X - prob_mean) ** 2) / prob_var + 1 / 2 * torch.log(2 * np.pi * prob_var)
        return log_prob
    # simultaneously optimize the probability of detection, subpixel localization and standard deviation.
    def loc_loss_analytical(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask):
        # each pixel is a component of Gaussian Mixture Model, with weights prob_normed
        prob_normed = P / (P.sum(-1).sum(-1)[:, None, None])

        p_inds = tuple((P + 1).nonzero().transpose(1, 0))

        xyzi_mu = xyzi_est[p_inds[0], :, p_inds[1], p_inds[2]]
        xyzi_mu[:, 0] += p_inds[2].type(torch.cuda.FloatTensor) + 0.5  # recover exact coordinate from sub-pixel
        xyzi_mu[:, 1] += p_inds[1].type(torch.cuda.FloatTensor) + 0.5

        xyzi_mu = xyzi_mu.reshape(P.shape[0], 1, -1, 4)
        xyzi_sig = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(P.shape[0], 1, -1, 4)  # >=0.01
        # xyzi_lnsig2 = xyzi_sig[p_inds[0], :, p_inds[1], p_inds[2]].reshape(self.batch_size, 1, -1, 4)  # >=0.01
        XYZI = xyzi_gt.reshape(P.shape[0], -1, 1, 4).repeat_interleave(self.train_size * self.train_size, 2)

        numerator = -1 / 2 * (abs(XYZI - xyzi_mu))
        denominator = (xyzi_sig ** 2)  # >0
        # denominator = torch.exp(xyzi_lnsig2)
        log_p_gauss_4d = (numerator / denominator).sum(3) - 1 / 2 * (torch.log(2 * np.pi * denominator[:, :, :, 0]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 1]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 2]) +
                                                                     torch.log(2 * np.pi * denominator[:, :, :, 3]))

        gauss_coef = prob_normed.reshape(P.shape[0], 1, self.train_size * self.train_size)
        gauss_coef_logits = torch.log(gauss_coef)
        gauss_coef_logmax = torch.log_softmax(gauss_coef_logits, dim=2)

        gmm_log = torch.logsumexp(log_p_gauss_4d + gauss_coef_logmax, dim=2)  # weights--probability of detection
        # c = torch.sum(p_gauss_4d * gauss_coef,-1)
        # gmm_log = (torch.log(c) * s_mask).sum(-1)
        return (gmm_log * s_mask).sum(-1)

    def final_loss(self, P, xyzi_est, xyzi_sig, xyzi_gt, s_mask):
        count_loss = torch.mean(self.count_loss_analytical(P, s_mask) * s_mask.sum(-1))
        loc_loss = -torch.mean(self.loc_loss_analytical(P, xyzi_est, xyzi_sig, xyzi_gt, s_mask))
        # P_locs_error = torch.mean(self.eval_P_locs_loss(P, locs)) if locs is not None else 0
        #bg_loss = torch.mean(self.eval_bg_sq_loss(psf_imgs_est, psf_imgs_gt)) if psf_imgs_est is not None else 0

        loss_total = count_loss + loc_loss #+ bg_loss #+ P_locs_error
        # loss_total = count_loss

        return loss_total

