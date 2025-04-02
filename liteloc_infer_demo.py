import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # If certain GPUs will be used, please set the index. Otherwise, delete this line.

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import torch
import time
from network import multi_process
from utils.help_utils import load_yaml_infer
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--infer_params_path', type=str, default='infer_params_demo_fig3a.yaml')
    args = parser.parse_args()

    # yaml_file = 'infer_params_demo_fig3a.yaml'  # remember to change p probability
    infer_params = load_yaml_infer(args.infer_params_path)

    test_model = torch.load(infer_params.Loc_Model.model_path) # suitable for both DECODE and LiteLoc 

    multi_process_params = infer_params.Multi_Process

    torch.cuda.synchronize()
    t0 = time.time()
    

    liteloc_analyzer = multi_process.CompetitiveSmlmDataAnalyzer_multi_producer(
        loc_model=test_model,
        tiff_path=multi_process_params.image_path,
        output_path=multi_process_params.save_path,
        time_block_gb=multi_process_params.time_block_gb,
        batch_size=multi_process_params.batch_size,
        sub_fov_size=multi_process_params.sub_fov_size,
        over_cut=multi_process_params.over_cut,
        multi_GPU=multi_process_params.multi_gpu,
        end_frame_num=multi_process_params.end_frame_num,
        num_producers=multi_process_params.num_producers,
    )

    torch.cuda.synchronize()
    t1 = time.time()

    print('init time: ' + str(t1 - t0))

    liteloc_analyzer.start()
    print('analyze time: ' + str(time.time() - t1))
