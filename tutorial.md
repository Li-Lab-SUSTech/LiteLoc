Outline:
1.OS: windows or linux
2.创建虚拟环境 + 配置环境 (spline需要额外一条命令：conda install -c haydnspass -c conda-forge spline)
3.运行demo-下载训练好的模型 & 示例数据
4.开始训练自己的模型
1) calibate aberration: Matlab load beads --> calibrate --> 得到包含map和spline model参数的.mat文件
2) 在.yaml中写参数，包括Camera, PSF_model以及Training的参数
3) 开始训练模型
5.推理数据
1) 在.yaml中添加推理数据的路径、保存结果的路径
2) 开始推理数据

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



