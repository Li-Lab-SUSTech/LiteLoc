# LiteLoc
A parallel scalable framework with lightweight deep learning for SMLM

### System Requirements
OS: Linux(GPU accelerated) / Windows(GPU accelerated) / MacOS
CUDA:
Software: conda, anaconda

### Python Environment Configuration
#### 1. create a virtual environment using conda  
`conda create -n liteloc_env python==3.8.19  # generally python=3.8 is okay, but must be lower than 3.9`  
#### 2. activate liteloc environment:  
`conda activate liteloc_env`
#### 3. install packages needed in liteloc  
`pip install -r requirements.txt`  
`conda install -c haydnspass -c conda-forge spline`

### Get Started
#### 1. get your own calibration map
We provide GUI in Matlab to generation calibration map. You can run 'calibrate_map_GUI.mat' program, load your experimental beads '.tif' file, 
set parameters and finally get a file, which includes both vectorial and cspline PSF model. This file will be used in the
training process below to generate training data.

#### 2. train your own liteloc
You can set parameters in 'train_template.yaml' file according to your optical setup. Please pay attention to the parameters with notes.

#### 3. infer your data and get results
Similar to the training setp, you need to set parameters in 'infer_template.yaml' file according your computation resource.
