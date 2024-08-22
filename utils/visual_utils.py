import matplotlib.pyplot as plt
from PSF_vector_gpu.PsfSimulation import *
from utils.help_utils import *
from utils.dataGenerator import *
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm
import tifffile as tiff

def ShowSamplePSF(psf_pars, train_pars):
    interval = 21

    I = torch.ones([interval, ])* 5000
    z = torch.linspace(-psf_pars.z_scale, psf_pars.z_scale, interval)

    if psf_pars.simulate_method == 'vector':
        x_offset = torch.zeros([interval, ])
        y_offset = torch.zeros([interval, ])
        zernike_aber = scio.loadmat(psf_pars.vector_psf.zernike_aber)['aber_map'][0, 0, :21]
        PSF = VectorPSFTorch(psf_pars.vector_psf, zernike_aber)

        psf_samples = PSF.simulate_parallel(x=x_offset.cuda(), y=y_offset.cuda(), z=z.cuda(), photons=I.cuda())
        psf_samples /= psf_samples.sum(-1).sum(-1)[:, None, None]
        psf_samples = cpu(psf_samples)
    else:
        x_px = torch.ones([interval, ]) * 51 / 2
        y_px = torch.ones([interval, ]) * 51 / 2
        spline_params = psf_pars.spline_psf
        psf = SMAPSplineCoefficient(calib_file=spline_params.calibration_file).init_spline(
            xextent=[0, 51],
            yextent=[0, 51],
            img_shape=[51, 51],
            device=spline_params.device_simulation,
            roi_size=None,
            roi_auto_center=None
            ).cuda()
        frame_ix = torch.arange(0, interval, 1)
        xyz_px = torch.cat([torch.unsqueeze(x_px, dim=1), torch.unsqueeze(y_px, dim=1), torch.unsqueeze(z, dim=1)], dim=1)
        psf_samples = psf.forward(xyz_px, I.detach().cpu(), frame_ix, ix_low=int(frame_ix.min()), ix_high=int(frame_ix.max()))

    plt.clf()
    for j in range(interval):
        plt.subplot(3, 7, j + 1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.05)
        plt.tick_params(labelsize=5)
        plt.imshow(psf_samples[j])
        plt.title(str(z[j].item()) + ' nm', fontdict={'size': 8})
    plt.show()

def ShowTrainImg(image_num, train_params, camera_params, psf_params):
    DataGen = DataGenerator(train_params, camera_params, psf_params)
    DataGen.batch_size = 1
    S, X, Y, Z, I, s_mask, xyzi_gt = DataGen.generate_batch(size=image_num, val=False)
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        plt.imshow(np.squeeze(cpu(S[i])))
    plt.show()
    imgs_sim = DataGen.simulatedImg_torch(S, X, Y, Z, I)
    plt.figure()
    for i in range(4):
        plt.subplot(1,4,i+1)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
        plt.imshow(np.squeeze(cpu(imgs_sim[i][0])))
    plt.show()

def GenerateData(psf, img_num, data_params, camera_params, psf_params, type='DMO', z_scale=700, margin=32):
    datagene = DataGenerator(data_params, camera_params, psf_params, type)
    datagene.batch_size = 1
    S, X, Y, Z, I, s_mask, xyzi_gt = datagene.generate_batch(size=img_num, val=True)
    img_sim = datagene.simulatedImg(S, X, Y, Z, I)
    
    frame, x, y, z, In, img_gene = [], [], [], [], [], []
    # output, counts = torch.unique_consecutive(S.nonzero()[:, 0], return_counts=True)
    for i in range(img_num):
        if(i == 10000):
            print("1")
        if min(S[i].nonzero().shape) == 0:
            img_gene.append(np.array(S[i].cpu()))
        else:
            img_gene.append(np.array(torch.squeeze(img_sim[i]).cpu()))
            xyzi = np.array(xyzi_gt[i].cpu())

            for j in range(len(S[i].nonzero())):
                frame.append(i+1)
                x.append(xyzi[j][0] * 100)
                y.append(xyzi[j][1] * 100)
                z.append(xyzi[j][2] * z_scale)
                In.append(xyzi[j][3] * 20000)
    activation = [frame, x, y, z, In]
    activation_df = pd.DataFrame(activation).transpose()
    activation_df = activation_df.rename(columns={0:'frame', 1:'x', 2:'y', 3:'z', 4:'intensity'})
    activation_df.to_csv(psf + '_generate_randomdata_128-4-15k_new.csv', mode='w', index=False)
    tiff.imwrite(psf + '_generate_randomdata_128-4-15k_new' + '.tif', img_gene)

# show training and validation loss at the end of each epoch
def ShowLossAtEndOfEpoch(learning_results):

    # x axis for the plot

    plt.figure(figsize=(9, 6), constrained_layout=True)
    index = 1
    for k,v in learning_results.items():
        plt.subplot(3, 3, index)
        plt.plot(*zip(*sorted(v.items())))
        plt.xlabel('iterations')
        plt.ylabel(k)
        index = index+1

    plt.legend()
    plt.show()

# plot_emitter_distance_distribution (fy)
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

# class cal_CRLB():
#     """
#     Calculate CRLB of given PSF at different axial positions.
#
#     """
#     def __int__(self, params, Nmol=25):
#         self.NA = params['NA']
#         self.refmed = params['refmed']
#         self.refcov = params['refcov']
#         self.refimm = params['refimm']
#         self.wavelength = params['lambda']
#         self.objstage0 = params['initial_obj_stage']
#         self.zemit0 = -1 * self.refmed / self.refimm * self.objstage0
#         self.pixel_size_xy = list(reversed(params['pixel_size_xy']))
#         self.Npupil = params['Npupil']
#         self.zernike_aber = params['zernike_aber']
#         self.Npixels = params['psf_size']
#         self.Nphotons = params['test_photons'] * np.ones(Nmol)
#         self.bg = params['test_bg'] * np.ones(Nmol)
#         self.xemit = 0 * np.ones(Nmol)
#         self.yemit = 0 * np.ones(Nmol)
#         self.zemit = 0 * np.linspace(-params['z_scale'], params['z_scale'], Nmol)
#         self.objstage = 0 * np.linspace(-1000, 1000, Nmol)
#         self.otf_rescale = [params['zernike_aber'][21], params['zernike_aber'][22]]
#         self.PSF_Torch =
#
#     def calculate_CRLB(self):
#         PSF_torch = VectorPSFTorch(psf_params=pa)


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






