from utils.eval_utils import assess_file

gt_path = "/home/feiyue/liteloc_git/only_local/grid_data/astig_d50_spline_activation.csv"
pred_path = "/home/feiyue/liteloc_git/only_local/grid_data/results/decode_astig_spline_traind40_datad50_results.csv"
pixel_xy = [108, 108]
frame_size = [64, 64]

eval_params = {'limited_x': [0, pixel_xy[0] * frame_size[0]], 'limited_y': [0, pixel_xy[1] * frame_size[1]],
               'tolerance': 250, 'tolerance_ax': 500, 'min_int': 100}

perf_dict, matches = assess_file(gt_path, pred_path, eval_params)