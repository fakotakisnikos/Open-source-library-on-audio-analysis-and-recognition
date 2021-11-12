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
- Tensorflow-2.6.1
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
    |--TaB dataset/：Data analysis
    |   |--participant_no1
        |--participant_no2
        |--participant_no3
    |	|--models/：Model construction module
    |--utils/：tools
    |   |--logger.py：Logging script
    |--pretrain/：Pre-trained models storage module
    |--model.py: Network processing script
    |--setting.py: Parameter setting script
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
mv TaB_tensorflow_files.zip TakeABreath/.
cd TakeABreath
unzip TakeABreath_pytorch_files.zip
```
3. Run the training code (e.g. SparseCNN)
```
python train.py --gpu_id 0 1    # multi-gpu training on gpu 0,1
or
python train.py --gpu_id 0    # single-gpu training on gpu 0
```
4. Run the testing code (e.g. SparseCNN)
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
   <th>Accuracy (%)</th>
   <th>Running time (s)</th>
</tr>
<tr>
   <td>LSTM</td>
   <td>98.6</td>
   <td>0.18</td>
</tr class="dataintable">
</table>

- Performance
```
Visualization of the classification results of our approach vs. the comparison ones after the same training epochs. 
It has demonstrated that the efficiency for training convergence and accuracy based on our TakeABreath pre-trained models.
```
<img src="images/efficiency.gif" width="812" hegiht="294" align=mid />


```
Results of TakeABreath pre-trained models to respiratory sounds classification (TaB) and accuracy evaluation metrics, respectively.
```
<table class="dataintable">
<tr>
   <th>Network</th>
   <th>Pretrain</th>
   <th>LungSeg(Dice)</th>
   <th>NoduleCls(accuracy)</th>
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
We thank [3D-ResNets-PyTorch](https://github.com/kenshohara/3D-ResNets-PyTorch).

### Contribution
If you want to contribute to VVR Group TaB, be sure to review the [contribution guidelines](https://github.com/tab/CONTRIBUTING.md)
