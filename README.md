# COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification

This repository is the official implementation of [COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification]()

## Requirements
1. Create an environment with python3.8.13 and activate. Further steps require this environment to be activated.
```
conda create -n COSE python=3.8.13
conda activate COSE
```
2. Install cuda 11.3 and pytorch:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
```
3. Install additional packages with: `pip install -r requirements.txt`

## Downloading Required Datasets and Checkpoints
Most datasets will be downloaded automatically to your local upon running the code using `torchvision.datasets`. For the EuroSAT, you'd need to download the dataset.
1. EuroSAT Dataset:
    - Go to `data/` folder and run the following in order:
        - `mkdir eurosat`
        - `cd eurosat`
        - `wget -c --no-check-certificate "https://madm.dfki.de/files/sentinel/EuroSAT.zip"`
        - `unzip EuroSAT.zip`
        - `rm EuroSAT.zip`
    - Go to [this gdrive folder](https://drive.google.com/drive/u/2/folders/1vwCpDrpeUZeyVQM5u-_ZVth34Cbd2mAC) to download `train_eurosat.csv` and `test_eurosat.csv` and place them in `../data/eurosat/2750` folder
2. MoCov3 Checkpoints
    - Go to [this link](https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md)
        - Download ResNet-50 with 1000 pretrain epochs (pretrain file, not linear file) - [this link](https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar)
        - Download ViT-Base (pretrain file, not linear file) - [this link](https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar)
    - Create a new directory in `data/`: `mkdir mocov3`
    - Place both files in `data/mocov3`
3. iBOT Checkpoints
    - Create a new directory in `data/`: `mkdir ibot`
    - Go to [this link](https://github.com/bytedance/ibot#pre-trained-models)
        - Download ViT-B/16 (with 84.1% Fin.) backbone(t) checkpoint (save as `vitb16.pth`) - [this link](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/vitb_16_rand_mask/checkpoint_teacher.pth)
        - Download Swin-T/14 backbone(t) checkpoint (save as `swint14.pth`) - [this link](https://lf3-nlp-opensource.bytetos.com/obj/nlp-opensource/archive/2022/ibot/swint_14/checkpoint_teacher.pth)
    - Place both files in `data/ibot` 
4. ConvNext Checkpoints
    - Create a new directory in `data/`: `mkdir convnext`
    - Go to [this link](https://github.com/facebookresearch/ConvNeXt)
        - Download the ConvNeXt-S model under "ImageNet-1K trained models"
    - Place the file in `data/convnext`

## Pre-trained Image Classification Models
Checkpoints for pre-trained image classification models will be provided