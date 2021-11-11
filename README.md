# Open Source Library on Audio Analysis and Recognition with Asthma Medication Adherence Monitoring Algorithms: State of the Art
 
<img src="http://www.ece.upatras.gr/images/ENGLISH_VERSION/LOGO/LogoVersionEN.png" align=mid />


# Take a breath
This repository contains a Python with Tensorflow implementation of [Take a breath: Smart Platform for self-management and support of patients with chronic respiratory diseases](http://www.vvr.ece.upatras.gr/). 
Many studies have shown that the performance on deep learning is significantly affected by volume of training data. The Take-a-Breath project provides the TaB dataset with inhaler's sounds, respiratory sounds and environmental sounds, to build relatively large deep learning programs. Based on this dataset, a state of the art deployment, on pre-trained machine learning models and corresponding code are provided.

### License
TaB dataset is released under the VVR Group License (refer to the LICENSE file for detailso).

### Citing TaB
If you use this code or pre-trained models, please cite the following:
```
    @article{ntalianis2020deep,
  title={Deep CNN Sparse Coding for Real Time Inhaler Sounds Classification},
  author={Ntalianis, Vaggelis and Fakotakis, Nikos Dimitris and Nousias, Stavros and Lalos, Aris S and Birbas, Michael and Zacharaki, Evangelia I and Moustakas, Konstantinos},
  journal={Sensors},
  volume={20},
  number={8},
  pages={2363},
  year={2020},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```
### Update(2021/11/11)
We uploaded 10 pre-trained models based on TaB dataset (1 dataset).
```
Model name             : parameters settings
 
```
 


### Contents
1. [Requirements](#Requirements)
2. [Installation](#Installation)
3. [Demo](#Demo)
4. [Experiments](#Experiments)
5. [TODO](#TODO)
6. [Acknowledgement](#Acknowledgement)

### Requirements
- Python 3.8.0
- PyTorch-0.4.1
- CUDA Version 9.0
- CUDNN 7.0.5

### Installation
- Install Python 3.8.0
- pip install -r requirements.txt


### Demo
- Structure of data directories
```
MedicalNet is used to transfer the pre-trained model to other datasets (here the MRBrainS18 dataset is used as an example).
MedicalNet/
    |--datasets/：Data preprocessing module
    |   |--brains18.py：MRBrainS18 data preprocessing script
    |	|--models/：Model construction module
    |   |--resnet.py：3D-ResNet network build script
    |--utils/：tools
    |   |--logger.py：Logging script
    |--toy_data/：For CI test
    |--data/：Data storage module
    |   |--TaB/：TaB dataset
    |	|   |--images/：source image named with patient ID
    |	|   |--labels/：mask named with patient ID
    |   |--train.txt: training data lists
    |   |--val.txt: validation data lists
    |--pretrain/：Pre-trained models storage module
    |--model.py: Network processing script
    |--setting.py: Parameter setting script
    |--train.py: TaB training demo script
    |--test.py: TaB testing demo script
    |--requirement.txt: Dependent library list
    |--README.md
```

- Network structure parameter settings
```
Model name   : parameters settings
resnet_10.pth: --model resnet --model_depth 10 --resnet_shortcut B
resnet_18.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet_34.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet_50.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet_101.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet_152.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet_200.pth: --model resnet --model_depth 200 --resnet_shortcut B
```

- After successfully completing basic installation, you'll be ready to run the demo.
1. Clone the TaB repository
```
git clone https://github.com//
```
2. Download data & pre-trained models ([Google Drive](https://drive.google.com/file/usp=sharing) or [VVR Group Datasets](https://share.vvrgroup.com/55sZyIx))

    Unzip and move files
```
mv TaB_pytorch_files.zip TakeABreath/.
cd TakeABreath
unzip TakeABreath_pytorch_files.zip
```
3. Run the training code (e.g. 3D-ResNet-50)
```
python train.py --gpu_id 0 1    # multi-gpu training on gpu 0,1
or
python train.py --gpu_id 0    # single-gpu training on gpu 0
```
4. Run the testing code (e.g. 3D-ResNet-50)
```
python test.py --gpu_id 0 --resume_path trails/models/resnet_50_epoch_110_batch_0.pth.tar --img_list data/val.txt
```

### Experiments
- Computational Cost 
```
GPU：NVIDIA Tesla P40
```
<table class="dataintable">
<tr>
   <th class="dataintable">Network</th>
   <th>Paramerers (M)</th>
   <th>Running time (s)</th>
</tr>
<tr>
   <td>3D-ResNet10</td>
   <td>14.36</td>
   <td>0.18</td>
</tr class="dataintable">
<tr>
   <td>3D-ResNet18</td>
   <td>32.99</td>
   <td>0.19</td>
</tr>
<tr>
   <td>3D-ResNet34</td>
   <td>63.31</td>
   <td>0.22</td>
</tr>
<tr>
   <td>3D-ResNet50</td>
   <td>46.21</td>
   <td>0.21</td>
</tr>
<tr>
   <td>3D-ResNet101</td>
   <td>85.31</td>
   <td>0.29</td>
</tr>
<tr>
   <td>3D-ResNet152</td>
   <td>117.51</td>
   <td>0.34</td>
</tr>
<tr>
   <td>3D-ResNet200</td>
   <td>126.74</td>
   <td>0.45</td>
</tr>
</table>

- Performance
```
Visualization of the classification results of our approach vs. the comparison ones after the same training epochs. 
It has demonstrated that the efficiency for training convergence and accuracy based on our TakeABreath pre-trained models.
```
<img src="images/efficiency.gif" width="812" hegiht="294" align=mid />


```
Results of transfer TakeABreath pre-trained models to respiratory sounds classification (TaB) and accuracy evaluation metrics, respectively.
```
<table class="dataintable">
<tr>
   <th>Network</th>
   <th>Pretrain</th>
   <th>LungSeg(Dice)</th>
   <th>NoduleCls(accuracy)</th>
</tr>
<tr>
   <td rowspan="2">3D-ResNet10</td>
   <td>Train from scratch</td>
   <td>71.30%</td>
   <td>79.80%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>87.16%</td>
    <td>86.87%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet18</td>
   <td>Train from scratch</td>
   <td>75.22%</td>
   <td>80.80%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>87.26%</td>
    <td>88.89%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet34</td>
   <td>Train from scratch</td>
   <td>76.82%</td>
   <td>83.84%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>89.31%</td>
    <td>89.90%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet50</td>
   <td>Train from scratch</td>
   <td>71.75%</td>
   <td>84.85%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>93.31%</td>
    <td>89.90%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet101</td>
   <td>Train from scratch</td>
   <td>72.10%</td>
   <td>81.82%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>92.79%</td>
    <td>90.91%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet152</td>
   <td>Train from scratch</td>
   <td>73.29%</td>
   <td>73.74%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>92.33%</td>
    <td>90.91%</td>
</tr>
<tr>
   <td rowspan="2">3D-ResNet200</td>
   <td>Train from scratch</td>
   <td>71.29%</td>
   <td>76.77%</td>
</tr>
<tr>
    <td>TakeABreath</td>
    <td>92.06%</td>
    <td>90.91%</td>
</tr>
</table>

- Please refer to [Open Source Library on Audio Analysis and Recognition with Asthma Medication Adherence Monitoring Algorithms: Review Paper](https://5) for more details：

### TODO
- [ ] Decision Trees (2014 app.)
- [ ] Random Forests
- [ ] Support Vector Machines
- [ ] AdaBoost
- [ ] GradBoost
- [ ] Hiddel Markov Models
- [ ] Convolutional Neural Networks
- [X] LSTM
- [ ] Linear Discriminant Analysis
- [ ] Quadratic Discriminant Analysis
- [ ] Sparse Convolutional Neural Networks

### Acknowledgement
We thank [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch) and [MRBrainS18](https://mrbrains18.isi.uu.nl/) which we build MedicalNet refer to this releasing code and the dataset.

### Contribution
If you want to contribute to VVR Group TaB, be sure to review the [contribution guidelines](https://github.com/tab/CONTRIBUTING.md)
