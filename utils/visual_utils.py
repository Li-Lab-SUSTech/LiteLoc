import matplotlib
import pandas as pd
import numpy as np
import torch
import scipy.io as scio
from scipy.stats import norm
import matplotlib.pyplot as plt
import os


from utils.help_utils import cpu
from utils.help_utils import load_h5, zernike45_to_zernike21
from vector_psf.vectorpsf import VectorPSFTorch
from spline_psf.calibration_io import SMAPSplineCoefficient
from utils.data_generator import DataGenerator

def show_sample_psf(psf_pars):

    interval = 21

    I = torch.ones([interval, ])* 5000
    z = torch.linspace(-psf_pars.z_scale, psf_pars.z_scale, interval)

    if psf_pars.simulate_method == 'vector' or psf_pars.simulate_method == 'ui_psf':
        x_offset = torch.zeros([interval, ])
        y_offset = torch.zeros([interval, ])
        if psf_pars.simulate_method == 'vector' and psf_pars.vector_psf.zernikefit_file is None:
            vector_params = psf_pars.vector_psf
            zernike = np.array(psf_pars.vector_psf.zernikefit_map, dtype=np.float32).reshape([21, 3])
            objstage0 = psf_pars.vector_psf.objstage0
        elif psf_pars.simulate_method == 'ui_psf':
            vector_params = psf_pars.ui_psf
            ui_psf, params_psf = load_h5(vector_params.zernikefit_file)
            zernike_coff = zernike45_to_zernike21(ui_psf.res.zernike_coeff[1]) * vector_params.wavelength / (2 * np.pi)
            vector_params.psfrescale = ui_psf.res.sigma[0]
            vector_params.NA = params_psf.option.imaging['NA']
            vector_params.refmed = params_psf.option.imaging['RI']['med']
            vector_params.refcov = params_psf.option.imaging['RI']['cov']
            vector_params.refimm = params_psf.option.imaging['RI']['imm']
            vector_params.wavelength = params_psf.option.imaging['emission_wavelength'] * 1000
            vector_params.Npupil = params_psf.option.model['pupilsize']
            vector_params.pixelSizeX = params_psf.pixel_size['x'] * 1000
            vector_params.pixelSizeY = params_psf.pixel_size['y'] * 1000

            zernike = np.array([2, -2, 0, 2, 2, 0, 3, -1, 0, 3, 1, 0, 4, 0, 0, 3, -3, 0, 3, 3, 0,
                                4, -2, 0, 4, 2, 0, 5, -1, 0, 5, 1, 0, 6, 0, 0, 4, -4, 0, 4, 4, 0,
                                5, -3, 0, 5, 3, 0, 6, -2, 0, 6, 2, 0, 7, 1, 0, 7, -1, 0, 8, 0, 0]).reshape([21, 3])
            zernike[:, 2] = zernike_coff
            objstage0 = psf_pars.ui_psf.objstage0
        else:
            calib_file = scio.loadmat(psf_pars.vector_psf.zernikefit_file, struct_as_record=False, squeeze_me=True)
            if 'vector_psf_model' in calib_file.keys():
                zernikefit_info = calib_file['vector_psf_model']
                zernike = zernikefit_info.aberrations
                vector_params = zernikefit_info.zernikefit
                objstage0 = psf_pars.vector_psf.objstage0
            elif 'psf_params_fitted' in calib_file.keys():
                psf_fit_info = calib_file['psf_params_fitted']
                psf_fit_info.NA = psf_fit_info.na
                psf_fit_info.Npupil = psf_fit_info.npupil
                psf_fit_info.pixelSizeX = psf_fit_info.pixel_size_xy[0]
                psf_fit_info.pixelSizeY = psf_fit_info.pixel_size_xy[1]
                psf_fit_info.psfSizeX = psf_fit_info.psf_size
                psf_fit_info.psfrescale = psf_fit_info.otf_rescale_xy[0]
                zernike = np.column_stack((psf_fit_info.zernike_mode, psf_fit_info.zernike_coef))
                vector_params = psf_fit_info
                objstage0 = psf_fit_info.objstage0
            else:
                zernikefit_info = calib_file['SXY']
                zernike = zernikefit_info.zernikefit.aberrations
                zernikefit_info.zernikefit.wavelength = psf_pars.vector_psf.wavelength
                zernikefit_info.zernikefit.psfrescale = psf_pars.vector_psf.psfrescale
                zernikefit_info.zernikefit.psfSizeX = zernikefit_info.zernikefit.sizeX
                zernikefit_info.zernikefit.psfSizeY = zernikefit_info.zernikefit.sizeY
                vector_params = zernikefit_info.zernikefit
                objstage0 = psf_pars.vector_psf.objstage0
        PSF = VectorPSFTorch(vector_params, zernike, objstage0)

        psf_samples = PSF.simulate_parallel(x=x_offset.cuda(), y=y_offset.cuda(), z=z.cuda(), photons=I.cuda())
        psf_samples /= psf_samples.sum(-1).sum(-1)[:, None, None]
        psf_samples = cpu(psf_samples)
    else:
        calib_file = scio.loadmat(psf_pars.spline_psf.calibration_file, struct_as_record=False, squeeze_me=True)
        if 'cspline_psf_model' in calib_file.keys():
            calib_mat = calib_file['cspline_psf_model']
        else:
            calib_mat = calib_file['SXY'].cspline
        roi_size = calib_mat.coeff.shape[0] + 1
        x_px = torch.ones([interval, ]) * roi_size / 2
        y_px = torch.ones([interval, ]) * roi_size / 2
        spline_params = psf_pars.spline_psf
        psf = SMAPSplineCoefficient(calib_file=spline_params.calibration_file).init_spline(
            xextent=[-0.5, roi_size - 0.5],
            yextent=[-0.5, roi_size - 0.5],
            img_shape=[roi_size, roi_size],
            device='cuda' if torch.cuda.is_available() else 'cpu',  # cuda or cpu, mps not support,
            roi_size=None,
            roi_auto_center=None
            )
        frame_ix = torch.arange(0, interval, 1)
        xyz_px = torch.cat([torch.unsqueeze(x_px, dim=1), torch.unsqueeze(y_px, dim=1), torch.unsqueeze(z, dim=1)], dim=1)
        psf_samples = psf.forward(xyz_px, I.detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=int(frame_ix.max()))
    for j in range(interval):
        plt.subplot(3, 7, j + 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.05)
        plt.tick_params(labelsize=4)
        plt.imshow(psf_samples[j], cmap='turbo')
        plt.title(str(z[j].item()) + ' nm', fontdict={'size': 7})
    plt.show()


def show_train_img(image_num, train_params, camera_params, psf_params):
    DataGen = DataGenerator(train_params, camera_params, psf_params)
    DataGen.batch_size = image_num
    locs, X, Y, Z, I, s_mask, xyzi_gt = DataGen.generate_batch_newest(size=image_num)
    plt.subplots(2, 2, constrained_layout=True)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.squeeze(cpu(locs[i])))
    plt.show()
    imgs_sim = DataGen.simulate_image(s_mask, xyzi_gt, locs, X, Y, Z, I, mode='visualize')
    plt.subplots(2, 2, constrained_layout=True)
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.squeeze(cpu(imgs_sim[i][0])))
        plt.colorbar()
    plt.show()

def plot_emitter_distance_distribution(data_path):
    activation = pd.read_csv(data_path)

    group_activation = activation.groupby("frame")

    dis_count = 0
    frame = np.unique(activation.values[:, 0])
    dis_record = [[] for j in range(len(frame))]  # 有一些帧上可能没有单分子点
    
    count = 0
    for i in frame:
        index = group_activation.indices[i]  # 这里修改过，注意！！！i+1 --> i
        emitter_x = activation['x'][index]
        emitter_y = activation['y'][index]
        emitter = torch.tensor([emitter_x.values, emitter_y.values]).transpose(1, 0)
        dis_matrix = torch.norm(emitter[:, None]-emitter, dim=2, p=2)
        # dis_matrix = dis_matrix - torch.diag_embed(torch.diag(dis_matrix))
        if len(dis_matrix) != 1:
            for k in range(len(dis_matrix)):
                dis_matrix[k][k] = 1e5
        dis_min = torch.min(dis_matrix, dim=1).values  # dim=1表示在每行上进行操作
        # if dis_min.max() > 12800:
        #     print(dis_matrix)
        # dis_matrix = F.pdist(emitter, p=2)
        dis_count += (dis_min < 990).sum()  # math.sqrt(2) * (1400) / 2 ~= 990
        # dis_record[i] = dis_matrix
        dis_record[count] = dis_min
        count += 1

    dis_record = np.array(torch.cat(dis_record, dim=0))
    # pd.DataFrame(dis_record).to_csv('calculate_distance_Astigmatism_density=2.csv', index=False, header=False)

    print('end')
    mean = dis_record.mean()
    variance = dis_record.std()
    fig, ax = plt.subplots()

    # n, bins, patches = ax.hist(dis_record, bins=range(0, 8000, 120), density=True)
    n, bins, patches = ax.hist(dis_record, bins=50, density=True)

    y = norm.pdf(bins, mean, variance)  # y = ((1 / (np.sqrt(2 * np.pi) * variance)) *np.exp(-0.5 * (1 / variance * (bins - mean))**2))
    plt.plot(bins, y, 'r')
    plt.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Proportion')
    ax.set_title('His. of normal distribution: '
                 fr'counts_lt={dis_count:.0f}, $\mu={mean:.0f}$, $\sigma={variance:.0f}$')
    fig.tight_layout()
    plt.show()

def plot_3d_grid():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    # 定义xyz坐标和光强
    x = [0.14961821, 0.216797, -0.06635976, 0.19337428, 0.06218988,
         0.22656065, -0.25518087, 0.48037207, -0.16809013, -0.32201526,
         0.39882308, 0.1707915, -0.22696164, 0.2590645, 0.4019149]
    y = [0.16866839, -0.35525006, 0.18011838, 0.41126794, -0.32057703,
         -0.4846584, 0.32932496, 0.44129652, -0.4986191, -0.04585546,
         0.22736222, -0.24364936, 0.12563914, 0.25639856, -0.40260136]
    z = [1337.6906, -1775.5074, 727.47516, -1133.8007,
         -2798.0103, -1724.7203, 1440.8558, 1768.409,
         2159.7214, -1385.1555, -1943.8789, 92.521194,
         1857.3281, -1522.3153, 506.19543]
    intensity = [0.5154952, 0.20380065, 0.4150588, 0.92753196, 0.55876184,
                 0.4518137, 0.85723543, 0.19585179, 0.08072899, 0.6631173,
                 0.6424689, 0.5053562, 0.65200126, 0.26236552, 0.08536333]

    # 将光强值映射到点的大小，可以根据需要调整比例因子
    sizes = np.array(intensity) * 1000 / 8

    # 创建3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图，使用点的大小表示光强，点的颜色统一为玫红色
    sc = ax.scatter(x, y, z, s=sizes, color='red')
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # 设置轴线颜色为白色
    ax.w_xaxis.line.set_color('white')
    ax.w_yaxis.line.set_color('white')
    ax.w_zaxis.line.set_color('white')

    # 去掉刻度线
    ax.xaxis.set_tick_params(color='white')
    ax.yaxis.set_tick_params(color='white')
    ax.zaxis.set_tick_params(color='white')

    # 隐藏坐标刻度值
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.savefig('./figures/ground_truth.png', dpi=300)

    # 显示图形
    plt.show()






