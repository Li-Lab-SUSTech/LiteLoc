# LiteLoc
## Scalable and lightweight deep learning for efficient high accuracy single-molecule localization microscopy

LiteLoc is a Python and [Pytorch](http://pytorch.org/) based scalable and lightweight deep learning for efficient high accuracy single-molecule localization microscopy (SMLM). \
LiteLoc includes a scalable and competitive data analysis framework and a lightweight deep learning network which has small number of parameters (only 1.33 M) and low computational complexity (71.08 GFLOPs) while maintaining comparable or even better localization precision. \
With the help of high parallelism between the data loader/analyzer/saver processes, the total analysis speed is ~25 times faster than that of DECODE and more than 560 MB/s data analysis throughput could be achieved with eight NVIDIA GTX 4090 GPUs.



### System Requirements
**OS:** Linux(GPU accelerated) / Windows(GPU accelerated)\
**CUDA version:** 12.1\
**Software:** conda, anaconda

### Python Environment Configuration
#### 1. create a virtual environment using conda  
```
conda create -n liteloc_env python==3.8.19
``` 
_**Note:** generally python=3.8 is okay, but should be lower than 3.9_ 
#### 2. activate liteloc environment:  
```
conda activate liteloc_env
```
#### 3. install packages imported in liteloc  
***Note**: If you cuda version is not 12.1, please download the version of torch from [Pytorch](https://pytorch.org/get-started/previous-versions/) according to your operation system and cuda version.*

```
pip install -r requirements.txt
```
```
conda install -c turagalab -c conda-forge spline
```

### Quick Start! (Demo of Figure 3a and Figure 3d in LiteLoc paper)
**Demo of Figure 3a:** train a network based on our experimental astigmatism NPC data.\
**Demo of Figure 3d:** train a network based on our experimental 6μm DMO-tetrapod NPC data.\
**Download link (PSF model and data):** [![image](https://zenodo.org/badge/DOI/10.5281/zenodo.13886596.svg)](https://zenodo.org/records/13886596)
#### 1. get PSF model and experimental data.
We also provide GUI in Matlab to generate your own PSF model. You can run **/PSF Modeling/Fit_PSF_model/calibrate_psf_model_GUI.m** program, load your own experimental beads **.tif** file, 
set parameters and finally get a **.mat** file, which includes both vectorial and cspline PSF model. This file will be loaded automatically in the
training process to generate training data.

#### 2. train your own LiteLoc
Please uncompress the downloaded data in step 1, then put astigmatism NPC data into the directory **/datasets/demo-fig3a/**, 6μm DMO-tetrapod NPC into **/datasets/demo-fig3d/**, put astigmatic PSF model into the directory **/calibrate_mat/demo-fig3a/** and 6μm tetrapod PSF into **/calibrate_mat/demo-fig3d/**.\
Then check the **'calibration_path'**, **'result_path'** and **'infer_data'** in **'train_params_demo_fig3a/3d.yaml'**. Please pay attention to the parameters with notes.\
_**Note:** If you run the program in terminal directly, please close the figure window that program plots at the beginning to continue the training._

#### 3. infer your data and get results
Similar to the training setup, you need to set parameters in the template file for inference according your computation resource.
