# The whole code will be available once the paper is accepted. 
<br/>
<br/>
<br/>

## G2L-Stereo: Global to Local Two-Stage Real-Time Stereo Matching Network

## Results on KITTI 2015 and KITTI 2012 leaderboard
[Leaderboard Link](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

| Method | Scene Flow <br> (EPE) | KITTI 2012 <br> (3-all) | KITTI 2015 <br> (D1-all) | Runtime (ms) |
|:-:|:-:|:-:|:-:|:-:|
| G2L-Stereo | 0.47 | 1.74 % | 1.88 % | 50 |
| Fast-ACVNet+ | 0.59 | 1.85 % | 2.01 % | 45 |
| HITNet | - | 1.89 % |1.98 % | 54 |
| CoEx | 0.69 | 1.93 % | 2.13 % | 33 |
| BGNet+ |  - | 2.03 % | 2.19 % | 35 |
| AANet |  0.87 | 2.42 % | 2.55 % | 62 |
| DeepPrunerFast | 0.97 | - | 2.59 % | 50 |





# How to use

## Environment
* Python 3.8
* Pytorch 2.1.0

## Install

### Create a virtual environment and activate it.

```
conda create -n g2l python=3.8
conda activate g2l
```
### Dependencies

```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install ruamel.yaml
pip install opencv-python
pip install scikit-image
pip install tensorboard==2.12.0
pip install matplotlib 
pip install tqdm
pip install timm==0.6.5
```

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

### Data Directories

In our setup, the dataset is organized as follows
```
../project
└── datasets
    └── SceneFlow
        ├── driving
        │   ├── disparity
        │   └── frames_finalpass
        ├── flyingthings3d_final
        │   ├── disparity
        │   └── frames_finalpass
        ├── monkaa
        │   ├── disparity
        │   └── frames_finalpass
        ├── kitti12
        │   ├── testing
        │   └── training
        └── kitti15
            ├── testing
            └── training
```

## Train
Use the following command to train G2L-Stereo on Scene Flow

Firstly, train the global disparity initialization network for 20 epochs,
```
python main_train_sf.py --datapath ../SceneFlow --epochs 20 --lrepochs 14,18:3\
    --only_disp4
```
Secondly, freeze the global disparity initialization network parameters, train the remaining network for another 20 epochs,
```
python main_train_sf.py --datapath ../SceneFlow --epochs 20 --lrepochs 14,18:3\
    --freezen_disp4 --loadckpt xxxx/checkpoint_000019.ckpt
```
Finally, train the complete network for 40 epochs,
```
python main_train_sf.py --datapath ../SceneFlow --epochs 40 --lrepochs 20,30,35:3\
    --whole_with_ckpt --loadckpt xxxx/checkpoint_000019.ckpt
```

Use the following command to train G2L-Stereo on KITTI (using pretrained model on Scene Flow)
```
python main_train_kitti.py\
    --trainlist filenames/kitti12_15_train_all.txt\
    --testlist filenames/kitti15_val.txt\
    --epochs 400 --lrepochs 300:10\
    --whole_with_ckpt --loadckpt pretrained_model/G2L_sceneflow.ckpt 
```

## Test
```
python main_test.py\
    --dataset sceneflow\
    --datapath ../SceneFlow\
    --loadckpt pretrained_model/G2L_sceneflow.ckpt\
    --test_batch_size 4 --logdir ./testlog --stage 0
```



# Acknowledgements

Part of the code is adopted from previous works:[CoEx](https://github.com/antabangun/coex), [HITNet](https://github.com/MJITG/PyTorch-HITNet-Hierarchical-Iterative-Tile-Refinement-Network-for-Real-time-Stereo-Matching), [AcfNet](https://github.com/youmi-zym/AcfNet),[ACVNet](https://github.com/gangweiX/ACVNet)



