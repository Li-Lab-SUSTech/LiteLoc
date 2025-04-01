import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "5"  # If certain GPUs will be used, please set the index. Otherwise, delete this line.

import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

import torch
import time
from network import multi_process
from utils.help_utils import load_yaml_infer

if __name__ == '__main__':

    yaml_file = 'infer_params_demo_fig3d.yaml'  # remember to change p probability
    infer_params = load_yaml_infer(yaml_file)

    liteloc = torch.load(infer_params.Loc_Model.model_path)

    multi_process_params = infer_params.Multi_Process

    torch.cuda.synchronize()
    t0 = time.time()

    liteloc_analyzer = multi_process.CompetitiveSmlmDataAnalyzer_multi_producer(
        loc_model=liteloc,
        tiff_path=multi_process_params.image_path,
        output_path=multi_process_params.save_path,
        time_block_gb=multi_process_params.time_block_gb,
        batch_size=multi_process_params.batch_size,
        sub_fov_size=multi_process_params.sub_fov_size,
        over_cut=multi_process_params.over_cut,
        multi_GPU=multi_process_params.multi_gpu,
        num_producers=multi_process_params.num_producers,
    )

    torch.cuda.synchronize()
    t1 = time.time()

    print('init time: ' + str(t1 - t0))

    liteloc_analyzer.start()
    print('analyze time: ' + str(time.time() - t1))
