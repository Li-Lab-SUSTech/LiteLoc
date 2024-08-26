from utils.eval_utils import assess_file

gt_path = "/home/feiyue/LiteLoc_spline/pos4w5_frame5w4_size32_nonNo.csv"
pred_path = "/home/feiyue/LiteLoc_spline/liteloc_pos4w5_frame5w4_size32.csv"
pixel_xy = [108, 108]
frame_size = [32, 32]

eval_params = {'limited_x': [0, pixel_xy[0] * frame_size[0]], 'limited_y': [0, pixel_xy[1] * frame_size[1]],
               'tolerance': 250, 'tolerance_ax': 500, 'min_int': 100}

perf_dict, matches = assess_file(gt_path, pred_path, eval_params )