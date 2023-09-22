# COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification

This repository is the official implementation of [COSE: A Consistency-Sensitivity Metric for Saliency on Image Classification](https://rangeldaroya.github.io/projects/cose)

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
Download required datasets and checkpoints [here](https://drive.google.com/file/d/1AI2ZBOBUD6M860UpxfmQ8nMAGVCTuMdt/view?usp=sharing). After downloading, extract the zip file and place the subfolders in `data/`

## Pre-trained Image Classification Models
Checkpoints for pre-trained image classification models are provided [here](). Download and extract the zip file. Place the subfolders in the `ckpts/` folder.
The accuracies for the different pretrained models are enumerated below (this is also available in the appendix of our paper):
|  	| **Caltech101** 	| **CIFAR10** 	| **CUB200** 	| **EuroSAT** 	| **Oxford102** 	|
|---	|---	|---	|---	|---	|---	|
| **ViT-L/16** 	| 94.82% 	| 94.86% 	| 84.09% 	| 97.06% 	| 84.14% 	|
| **ResNet101** 	| 94.41% 	| 88.07% 	| 81.46% 	| 93.63% 	| 82.22% 	|
| **ViT-B//16** 	| 96.00% 	| 95.00% 	| 81.00% 	| 96.00% 	| 81.00% 	|
| **DINO ViT-B//16** 	| 97.00% 	| 95.00% 	| 75.00% 	| 98.00% 	| 91.00% 	|
| **MoCov3 ViT-B//16** 	| 91.00% 	| 93.00% 	| 76.67% 	| 96.00% 	| 78.00% 	|
| **iBOT ViT-B//16** 	| 96.00% 	| 95.00% 	| 73.00% 	| 97.00% 	| 89.00% 	|
| **ResNet50** 	| 94.00% 	| 85.00% 	| 81.00% 	| 94.00% 	| 82.00% 	|
| **DINO ResNet50** 	| 93.00% 	| 84.00% 	| 73.00% 	| 96.00% 	| 85.00% 	|
| **MoCov3 ResNet50** 	| 96.00% 	| 90.00% 	| 72.00% 	| 95.00% 	| 83.00% 	|
| **Swin-T** 	| 96.00% 	| 92.00% 	| 84.00% 	| 96.00% 	| 84.00% 	|
| **iBOT Swin-T** 	| 96.00% 	| 95.00% 	| 80.27% 	| 97.00% 	| 88.00% 	|
| **ConvNext** 	| 95.56% 	| 93.94% 	| 82.86% 	| 95.37% 	| 79.53% 	|
| **SparK ConvNext** 	| 75.58% 	| 78.27% 	| 53.19% 	| 87.20% 	| 48.04% 	|
|  	| **93.49%** 	| **90.70%** 	| **76.73%** 	| **95.25%** 	| **81.15%** 	|

## Using this repository

### Compiling Results across methods and generating metrics
1. Download our results for different methods [here](https://drive.google.com/file/d/10qZOKNYgqIP0UCr-ItSOVz9kVakkZCQs/view?usp=drive_link), extract the zip file, place all subfolders in the `outputs/` directory.
2. To compile metrics across different saliency methods and models, go to `postprocess/` and run the following (make sure the global variables are updated to reflect the models, saliency methods, and datasets to be compiled)
```
python 02_compile_explanations.py
```
3. Compiling the provided outputs would give the following metrics:

| **Dataset** 	| **Model** 	| **BlurIG** 	| **GradCAM** 	| **GradCAM++** 	| **GuidedIG** 	| **IG** 	| **LIME** 	| **SmoothGrad** 	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|
| **Caltech101** 	| **ConvNext** 	| 63.01% 	| 63.50% 	| 65.28% 	| 54.25% 	| 62.53% 	| 61.51% 	| 60.90% 	|
|  	| **ResNet50** 	| 65.77% 	| 61.86% 	| 45.41% 	| 52.41% 	| 58.80% 	| 56.69% 	| 59.60% 	|
|  	| **Swin-T** 	| 67.81% 	| 67.15% 	| 57.50% 	| 52.54% 	| 64.68% 	| 63.17% 	| 61.95% 	|
|  	| **ViT** 	| 69.60% 	| 68.66% 	| 60.30% 	| 57.48% 	| 66.67% 	| 61.12% 	| 66.41% 	|
| **CIFAR10** 	| **ConvNext** 	| 61.46% 	| 66.74% 	| 66.72% 	| 53.02% 	| 60.91% 	| 62.06% 	| 58.85% 	|
|  	| **ResNet50** 	| 60.47% 	| 62.29% 	| 42.75% 	| 48.27% 	| 51.20% 	| 59.86% 	| 52.02% 	|
|  	| **Swin-T** 	| 66.35% 	| 69.66% 	| 58.29% 	| 50.11% 	| 63.05% 	| 65.76% 	| 56.95% 	|
|  	| **ViT** 	| 66.46% 	| 71.54% 	| 59.11% 	| 55.72% 	| 66.68% 	| 63.85% 	| 61.00% 	|
| **CUB200** 	| **ConvNext** 	| 54.15% 	| 60.61% 	| 59.81% 	| 59.59% 	| 61.20% 	| 56.90% 	| 47.89% 	|
|  	| **ResNet50** 	| 58.63% 	| 44.01% 	| 40.74% 	| 56.50% 	| 60.05% 	| 55.50% 	| 52.38% 	|
|  	| **Swin-T** 	| 62.37% 	| 62.51% 	| 49.78% 	| 60.14% 	| 64.02% 	| 59.06% 	| 56.05% 	|
|  	| **ViT** 	| 59.07% 	| 64.80% 	| 56.65% 	| 61.26% 	| 60.42% 	| 58.31% 	| 53.70% 	|
| **EuroSAT** 	| **ConvNext** 	| 59.17% 	| 65.47% 	| 63.58% 	| 52.83% 	| 61.45% 	| 60.23% 	| 57.17% 	|
|  	| **ResNet50** 	| 57.28% 	| 62.27% 	| 45.17% 	| 40.87% 	| 46.14% 	| 59.47% 	| 47.96% 	|
|  	| **Swin-T** 	| 64.74% 	| 66.49% 	| 53.82% 	| 47.51% 	| 61.99% 	| 59.60% 	| 60.43% 	|
|  	| **ViT** 	| 67.85% 	| 70.31% 	| 57.95% 	| 57.90% 	| 68.12% 	| 60.63% 	| 62.48% 	|
| **Oxford102** 	| **ConvNext** 	| 58.60% 	| 62.78% 	| 61.74% 	| 58.73% 	| 60.71% 	| 57.23% 	| 57.77% 	|
|  	| **ResNet50** 	| 61.13% 	| 61.90% 	| 40.30% 	| 54.93% 	| 55.07% 	| 58.32% 	| 56.60% 	|
|  	| **Swin-T** 	| 66.57% 	| 67.38% 	| 56.87% 	| 52.86% 	| 63.42% 	| 62.06% 	| 59.41% 	|
|  	| **ViT** 	| 66.90% 	| 68.31% 	| 59.14% 	| 59.67% 	| 65.92% 	| 61.44% 	| 62.95% 	|
|  	|  	| **63.23%** 	| **64.66%** 	| **54.59%** 	| **54.73%** 	| **61.33%** 	| **60.11%** 	| **57.94%** 	|

### Training a set of models for a given dataset
If you wish to use the pre-trained models used in the paper, please refer to the section on Pre-trained Image Classification Models above.
1. Go to `src/` and run the following
```
python train_model.py --config_path configs/pred_models/<dataset-name>.yaml
```
Sample use:
```
python train_model.py --config_path configs/pred_models/cifar10.yaml
```
2. The config file specifies the learning rate, type of optimizer, number of epochs, and other parameters for training.
3. The parameters used for the pre-trained classification models are provided in the `src/configs/pred_models/` folder. Note that certain datasets like CUB could have multiple config files corresponding to different models with different parameters.
4. Once training is done and you are satisfied with the accuracy for image classification, you can modify `src/configs/trained_models/<dataset-name>.yaml` to specify the name of the checkpoint, and the interval epochs that were saved.


### Generating Explanations using a given saliency method
If you wish to use results from the paper, please refer to the section on Compiling results above.
1. Go to `src/` and run the following
```
python explain.py --config_path configs/<saliency-method>/<dataset-name>.yaml
```
Sample use that generates explanations for CIFAR10 using LIME:
```
python explain.py --config_path configs/lime/cifar10.yaml
```
2. Each run produces a csv file and a set of images. Each csv file contains the metrics of the generated explanations and the corresponding transformation that was applied. The generated images show sample explanation results for qualitative analysis.
3. If a summarized set of metrics is needed, please follow instructions on Compiling Results


### How do I train new models on the datasets?
Coming soon.

### How do I add a new dataset for training and extracting explanations?
Coming soon.

### How do I add new saliency methods for benchmarking?
Coming soon.

## References
- https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md
- https://github.com/bytedance/ibot#pre-trained-models
- https://github.com/facebookresearch/ConvNeXt
