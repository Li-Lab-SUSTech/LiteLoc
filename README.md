# LiteLoc
### Scalable and lightweight deep learning for high-throughput single-molecule localization microscopy

### System Requirements
OS: Linux(GPU accelerated) / Windows(GPU accelerated)\
CUDA: 12.1\
Software: conda, anaconda

### Python Environment Configuration
#### 1. create a virtual environment using conda  
`conda create -n liteloc_env python==3.8.19  # generally python=3.8 is okay, but must be lower than 3.9`  
#### 2. activate liteloc environment:  
`conda activate liteloc_env`
#### 3. install packages needed in liteloc  
Note: You should check cuda version of your device and modify the version of torch.

`pip install -r requirements.txt`  
`conda install -c haydnspass -c conda-forge spline`

### Quick Start! (Demo of Figure 3a and Figure 3d)
**Demo of Figure 3a:** train a network based on our experimental astigmatism NPC data.\
**Demo of Figure 3d:** train a network based on our experimental 6μm DMO-tetrapod NPC data.
#### 1. get PSF model and experimental data.
The model and data can be downloaded from [zenodo: 10.5281/zenodo.13886596](https://zenodo.org/records/13886596).\
We also provide GUI in Matlab to generate PSF model. You can run '/PSF Modeling/Fit_PSF_model/calibrate_psf_model_GUI.m' program, load your own experimental beads '.tif' file, 
set parameters and finally get a '.mat' file, which includes both vectorial and cspline PSF model. This file will be loaded automatically in the
training process to generate training data.

#### 2. train your own liteloc
Please uncompress the downloaded data in step 1 and put astigmatism data into the directory '/datasets/demo-fig3a/', 6μm DMO-tetrapod NPC into '/datasets/demo-fig3d'.\
Then check the 'calibration_path', 'result_path' and 'infer_data' in 'train_params_demo_fig3a/3d.yaml'.You can also set parameters in the template '.train_params_demo_fig3a/3d.yaml' file for training according to your optical setup. Please pay attention to the parameters with notes.

#### 3. infer your data and get results
Similar to the training setp, you need to set parameters in the template file for inference according your computation resource.
