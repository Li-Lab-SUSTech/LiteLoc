<!-- ABOUT THE PROJECT -->
## LiteLoc 
**LiteLoc** is a Python and [Pytorch](http://pytorch.org/) based scalable and lightweight deep learning for efficient high accuracy single-molecule localization microscopy (SMLM). 

**LiteLoc** includes a scalable and competitive data analysis framework and a lightweight deep learning network which has small number of parameters (**only 1.33 M**) and low computational complexity (**71.08 GFLOPs**, 10 images with 128x128 pixels) while maintaining comparable or even better localization precision. \
With the help of high parallelism between the data loader/analyzer/saver processes, the total analysis speed is **~25 times** faster than that of DECODE and **more than 560 MB/s** data analysis throughput could be achieved with eight NVIDIA GTX 4090 GPUs.


<!-- GETTING STARTED -->

## Getting Started

### Installation
### Option 1: Using Docker (If macOS, recommend this)

1. Install Docker 
2. Clone/Download the repo
   ```
   git clone https://github.com/Li-Lab-SUSTech/LiteLoc.git
   ```
3. Run Docker mirror in the terminal (assuming program's path: C:\codes\LiteLoc)
   ``` 
   # for GPU-accelerated device
   docker run -it --rm --shm-size=10gb  --gpus all -v  C:\codes\LiteLoc:/app in win of ubuntu:/app --name liteloc-gpu-container terence133/liteloc-gpu:latest
   
   # for CPU-only device
   docker run -it --rm --shm-size=10gb -v C:\codes\LiteLoc:/app --name liteloc-cpu-container terence133/liteloc-cpu:latest
   ```

### Option 2: Using pip
1. Clone the repo
   ```
   git clone https://github.com/Li-Lab-SUSTech/LiteLoc.git
   ```
3. Create a virtual environment 
   ```
   conda create -n liteloc_env python==3.9.21
   ```
4. Activate liteloc environment and install the required packages
   ```
   conda activate liteloc_env
   cd LiteLoc-main
   
   # for Ubuntu and Windows
   pip install -r requirements.txt
   
   # for macOS
   pip install -r requirements-mac.txt
   
   conda install -c turagalab -c conda-forge spline
   ```
The project files should be organized as the following hierarchy:
   ```
.
|-- PSF_Modeling                                
|   |-- Fit_PSF_model
|       |-- calibrate_psf_model_GUI.m          // MATLAB GUI for vectorial PSF calibration from beads images.
|       `-- ...
|-- demo                                       // several demos for showing how to use LiteLoc to train and infer.
|   |-- demo1_astig_npc                        // demo for training and inference of astigmatic PSF-based NPC imaging experiments.
|   |-- demo2_tetrapod_npc                     // demo for training and inference of Tetrapod PSF-based NPC imaging experiments.
|   |-- demo3_calibrate_psf                    // demo for PSF calibration using Python.
|   |-- demo4_decode                           // demo for integrate DECODE in LiteLoc's acceleration framework.
|   `-- general_training_inference.ipynb       // a general training and inference process with intermediate results.
|-- vector_psf                                 // module for generating vectorial PSF. 
|-- spline_psf                                 // module for generating C-spline interpolation PSF (from DECODE).
|-- network
|   |-- liteloc.py                             // architecture of LiteLoc network.
|   |-- decode.py                              // architecture of DECODE network.
|   |-- loc_model.py                           // entire training process of LiteLoc.
|   |-- loc_model_decode.py                    // entire training process of DECODE.
|   |-- loss_utils.py                          // loss function.
|   |-- multi_process.py                       // scalable and competitive inference framework.         
|   `-- ...
|-- utils
|   |-- data_generator.py                      // generate training data.
|   |-- eval_utils.py                          // evaluate localization results and output metrics.
|   |-- help_utils.py                          // various functions that can be easily invoked.
    `-- ...
|-- calibrate_mat
|   |-- astig_npc_psf_model.mat                // PSF model of demo1.
|   `-- tetrapod_npc_psf_model.mat             // PSF model of demo2.
|-- datasets                                   // directory for placing inference dataset.
|-- results                                    // directory for placing training results.
   ```


<!-- USAGE EXAMPLES -->
### Demos
We provide several examples in ```demos``` directory, illustrating how to train LiteLoc and infer based on different PSF models.
For all demos: 
* []() Please download data from [![image](https://zenodo.org/badge/DOI/10.5281/zenodo.13886596.svg)](https://zenodo.org/records/13886596) and put in ```demos``` directory.
* []() Training and inference results will be automatically saved in ```results``` directory.
* []() Localizations are saved as ```.csv``` file.

We recommend users to use [SMAP](https://www.nature.com/articles/s41592-020-0938-1) software for post-processing and 
rendering the results. A ```general_training_inference.ipynb``` file for demo1 is also provided to show the parameter definition 
and intermediate results of the entire process.
#### Demo1: LiteLoc for astigmatic PSF-based NPC imaging.
Demo1 is based on the experimental Nup96 NPC dataset with a 1.4 μm astigmatic PSF.
* []() ```train_params_demo1.yaml```: Parameter setting for training LiteLoc, which will be automatically invoked before training.
* []() ```liteloc_train_demo1.py```: Program for training LiteLoc.
* []() ```infer_params_demo1.yaml```: Parameter setting for inference.
* []() ```liteloc_infer_demo1.py```: Program for inference.

Expected runtime on NVIDIA RTX 4090: Training ~30 mins. Inference ~85 s.
#### Demo2: LiteLoc for Tetrapod PSF-based NPC imaging.
Demo2 is based on the experimental Nup96 NPC dataset with a 6μm Tetrapod PSF. The files in this directory are the 
similar to those in ```demo1_astig_npc```.

Expected runtime on NVIDIA RTX 4090: Training ~80 mins. Inference ~225 s.

#### Demo3: PSF calibration using Python.
Considering some users prefer using purely Python program to realize the entire data analysis, we also provide 
a Python-based PSF calibration approach in demo3.

#### Demo4: Deploy DECODE on LiteLoc acceleration framework.
To further enhance usability, we provide a clear example demonstrating how DECODE can be accelerated using LiteLoc’s framework. 
Users who wish to leverage LiteLoc for accelerating their own deep learning networks can directly refer to demo4.


<!-- LICENSE-MIT -->
## License

This work is dual-licensed under MIT and GPL v3.0.
If you use the code in the ```spline_psf``` directory, you must comply with the GPL v3.0 license. 
For other codes, the MIT license applies.


<!-- CONTACT -->
## Contact

For any questions about this software, please contact [Li-Lab](https://li-lab-sustech.github.io/).