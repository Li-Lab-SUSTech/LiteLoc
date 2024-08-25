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


