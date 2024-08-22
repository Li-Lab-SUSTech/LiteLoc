from utils.eval_utils import assess_file

gt_path = "/home/feiyue/LiteLoc_local_torchsimu/random_points/Tetrapod_6um/hsnr_ld/activations_nonNo.csv"
pred_path = "/home/feiyue/LiteLoc_spline/spline_data/decode_hsnr_ld_spline_tetra6_inverse_z_offset330.csv"
pixel_xy = [108, 108]
frame_size = [128, 128]

eval_params = {'limited_x': [0, pixel_xy[0] * frame_size[0]], 'limited_y': [0, pixel_xy[1] * frame_size[1]],
               'tolerance': 250, 'tolerance_ax': 500, 'min_int': 100}

perf_dict, matches = assess_file(gt_path, pred_path, eval_params )