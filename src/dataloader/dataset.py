import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

from dataloader.augmentation import ModifiedTrivial

from dataloader.cifar import CIFAR10
from dataloader.cub import Cub
from dataloader.flowers102 import Flowers102
from dataloader.eurosat import EuroSAT
from dataloader.caltech import Caltech101

def get_dataset(dataset, model_name=None):
    pil_transf, preprocess_trans = get_transforms(model_name)
    train_trans = transforms.Compose([ 
        pil_transf,
        ModifiedTrivial(),
        preprocess_trans
    ])
    test_aug = ModifiedTrivial()
    test_aug.fixed_augment = "Identity"
    test_trans = transforms.Compose([
        pil_transf,
        test_aug,
        preprocess_trans
    ])

    if dataset == "cub200":
        # CUB has additional data augmentations in training to improve performance
        trainset = Cub(train=True, transform=train_trans, return_image_only=False)
        testset = Cub(train=False, transform=test_trans, return_image_only=False)
    elif dataset == "cifar10":
        trainset = CIFAR10(
            root='../data', train=True, download=True, transform=train_trans)
        testset = CIFAR10(
            root='../data', train=False, download=True, transform=test_trans)
    elif dataset == "oxford102":
        trainset = Flowers102(
            root='../data', split='train', download=True, transform=train_trans)
        testset = Flowers102(
            root='../data', split='test', download=True, transform=test_trans)
    elif dataset == "eurosat":
        trainset = EuroSAT(root="../data", train=True, transform=train_trans)
        testset = EuroSAT(root="../data", train=False, transform=test_trans)
    elif dataset == 'caltech101':
        trainset = Caltech101(
            root='../data', split='train', download=True, transform=train_trans)
        testset = Caltech101(
            root='../data', split='test', download=True, transform=test_trans)
    else:
        raise NotImplementedError(f"Dataset specified [{dataset}] not implemented")
    
    return trainset, testset

def get_cve_dataset(dataset, transform_type, rot_vals_deg, trans_vals, scales, model_name=None):  # for both cve and scve
    pil_transf, preprocess_trans = get_transforms(model_name)
    trans = transforms.Compose([
        pil_transf,
        preprocess_trans
    ])
    # get args
    dataset_args = {
        "transform": trans,
        "rot_vals_deg": rot_vals_deg,
        "trans_vals": trans_vals,
        "scales": scales,
        "to_bgr": transform_type=="bgr",
        "to_rrr": transform_type=="rrr",
        "to_double_data_only": transform_type=="random",    # random model
    }

    if dataset=="cub200":
        dataset = Cub(**dataset_args, train=False)
    elif dataset=="cifar10":
        dataset = CIFAR10(**dataset_args, train=False, root="../data", download=True)
    elif dataset=="oxford102":
        dataset = Flowers102(**dataset_args, split="test", root="../data", download=True)
    elif dataset=="eurosat":
        dataset = EuroSAT(**dataset_args, train=False, root="../data")
    elif dataset=="caltech101":
        dataset = Caltech101(**dataset_args, split="test", root="../data", download=True)
    else:
        raise NotImplementedError(f"Dataset specified [{dataset}] not implemented")
    
    return dataset

def get_transforms(model_name):
    if model_name in ["ResNet50", "DINO_ResNet50", "MoCov3_ResNet50"]:
        pil_transf = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])
    elif model_name in ["ViT_B_16", "DINO_ViT_B_16", "MoCov3_ViT_B_16", "iBOT_ViT_B_16", "VGG16", "DenseNet121"]:
        # from https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights
        pil_transf = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224)
        ])
    elif model_name in ["Swin_T_14", "iBOT_Swin_T_14", "convnext_small", "SparK_convnext_small"]:
        # from https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html#torchvision.models.swin_t
        pil_transf = transforms.Compose([
            transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224)
        ])
    else:
        raise NotImplementedError(f"Model specified [{model_name}] not implemented")

    preprocess_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return pil_transf, preprocess_trans
