import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
import time
from network import multi_process
from utils.help_utils import load_yaml

if __name__ == '__main__':

    yaml_file = 'infer_template.yaml'
    infer_params = load_yaml(yaml_file)

    liteloc = torch.load(infer_params.Loc_Model.model_path)

    multi_process_params = infer_params.Multi_Process

    torch.cuda.synchronize()
    t0 = time.time()

    liteloc_analyzer = multi_process.CompetitiveSmlmDataAnalyzer_multi_producer(
        loc_model=liteloc,
        tiff_path=multi_process_params.image_path,
        output_path=multi_process_params.save_path,
        time_block_gb=multi_process_params.time_block_gb,
        batch_size=multi_process_params.batch_size,  # 96
        sub_fov_size=multi_process_params.sub_fov_size,  # 336
        over_cut=multi_process_params.over_cut,
        multi_GPU=multi_process_params.multi_gpu,
        end_frame_num=multi_process_params.end_frame_num,
        num_producers=multi_process_params.num_producers,
        # num_producers should be divisible by num_consumers, e.g. num_consumers=8, num_producers can be 1,2,4,8; if num_consumers=7, num_producers can be 1 or 7.
    )

    torch.cuda.synchronize()
    t1 = time.time()

    print('init time: ' + str(t1 - t0))

    liteloc_analyzer.start()
    print('analyze time: ' + str(time.time() - t1))
