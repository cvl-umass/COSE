from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    vit_b_16,
    ViT_B_16_Weights,
    swin_t,
    Swin_T_Weights,
    vgg16,
    VGG16_Weights,
    densenet121,
    DenseNet121_Weights,
)

from model.dino_vision_transformer import vit_base
from model.swin_transformer import swin_tiny
from model.spark_convnext import convnext_small as cvnext_small

def get_gradcam_target_layer(model_name, model):
    if model_name in ["ResNet50", "DINO_ResNet50", "MoCov3_ResNet50"]:
        target_layers = [model.layer4[-1]]
    elif model_name == "DenseNet121":
        target_layers = [model.features[-1]]     # based on https://github.com/jacobgil/pytorch-grad-cam#chosing-the-target-layer
    elif model_name == "VGG16":
        target_layers = [model.features[-1]] # based on https://github.com/jacobgil/pytorch-grad-cam#chosing-the-target-layer
    elif model_name == "ViT_B_16":
        target_layers = [model.encoder.layers[-1].ln_1] # see https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py
    elif model_name in ["DINO_ViT_B_16", "MoCov3_ViT_B_16", "iBOT_ViT_B_16"]:
        target_layers = [model.blocks[-1].norm1]
    elif model_name == "Swin_T_14":
        target_layers = [model.features[-1][-1].norm1]  # See https://github.com/jacobgil/pytorch-grad-cam/issues/84
    elif model_name == "iBOT_Swin_T_14":
        target_layers = [model.layers[-1].blocks[-1].norm1]
    elif model_name in  ["convnext_small", "SparK_convnext_small"]:
        # target_layers = [model.features[-1][-1].block[2]]   # first layernorm in the last block of convnext
        target_layers = [model.stages[-1][-1]]   # first layernorm in the last block of convnext
    else:
        raise NotImplementedError(f"Model specified [{model_name}] not implemented")

    return target_layers

def get_pretrained_model(model_name, is_random=False):
    if model_name == "ResNet50":
        w = None if is_random else ResNet50_Weights.DEFAULT
        model = resnet50(weights=w)
        num_feats = model.fc.in_features
    elif model_name == "convnext_small":
        model = cvnext_small()
        num_feats = model.head.in_features
        if not is_random:
            state_dict = torch.load("../data/convnext/convnext_small_1k_224_ema.pth")["model"]
            model.load_state_dict(state_dict, strict=True)
    elif model_name == "SparK_convnext_small":
        model = cvnext_small()
        num_feats = model.head.in_features
        if not is_random:
            state_dict = torch.load("../data/spark/official_convnext_small_1kpretrained.pth")
            model.load_state_dict(state_dict, strict=False)
    elif model_name == "DenseNet121":
        w = None if is_random else DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=w)
        num_feats = model.classifier.in_features
    elif model_name == "VGG16":
        w = None if is_random else VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=w)
        num_feats = model.classifier[-1].in_features
    elif model_name == "DINO_ResNet50":
        model = resnet50(weights=None)
        if not is_random:
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=False)
        num_feats = model.fc.in_features
    elif model_name == "MoCov3_ResNet50":
        model = resnet50(weights=None)
        num_feats = model.fc.in_features
        # Ckpts and parsing based on https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py
        if not is_random:
            ckpt = torch.load("../data/mocov3/r-50-1000ep.pth.tar")
            state_dict = ckpt["state_dict"]
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.fc = None # remove since there are no weights for the classifier head (only backbone)
            model.load_state_dict(state_dict, strict=True)
    elif model_name == "MoCov3_ViT_B_16":
        model = vit_base(patch_size=16)
        num_feats = model.blocks[-1].mlp.fc2.out_features
        # Ckpts and parsing based on https://github.com/facebookresearch/moco-v3/blob/main/main_lincls.py
        if not is_random:
            ckpt = torch.load("../data/mocov3/vit-b-300ep.pth.tar")
            state_dict = ckpt["state_dict"]
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.head'):
                    # remove prefix
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]
            model.load_state_dict(state_dict, strict=True)
    elif model_name == "ViT_B_16":
        w = None if is_random else ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=w)
        num_feats = model.heads.head.in_features
    elif model_name == "DINO_ViT_B_16":
        # model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        model = vit_base(patch_size=16)
        if not is_random:
            # from https://github.com/facebookresearch/dino/blob/main/hubconf.py#L59-L63
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
                map_location="cpu",
            )
            model.load_state_dict(state_dict, strict=True)
        num_feats = model.blocks[-1].mlp.fc2.out_features
    elif model_name == "iBOT_ViT_B_16":
        model = vit_base(patch_size=16)
        num_feats = model.blocks[-1].mlp.fc2.out_features
        if not is_random:
            # Following based on https://github.com/bytedance/ibot/blob/main/utils.py#L111-L122
            state_dict = torch.load("../data/ibot/vitb16.pth")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
    elif model_name == "Swin_T_14":
        w = None if is_random else Swin_T_Weights.IMAGENET1K_V1
        model = swin_t(weights=w)
        num_feats = model.head.in_features
    elif model_name == "iBOT_Swin_T_14":
        model = swin_tiny(window_size=14)
        num_feats = model.head.in_features
        if not is_random:
            model.head = None
            # Following based on https://github.com/bytedance/ibot/blob/main/utils.py#L111-L122
            state_dict = torch.load("../data/ibot/swint14.pth")["state_dict"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=True)
    else:
        raise NotImplementedError(f"Model specified [{model_name}] not implemented")

    return model, num_feats


def get_changed_model(model_name, model, last_layer):
    """Changes last layer/head of the model"""
    if model_name in ["ResNet50", "DINO_ResNet50", "MoCov3_ResNet50"]:
        model.fc = last_layer
    elif model_name in ["DenseNet121"]:
        model.classifier = last_layer
    elif model_name == "VGG16":
        model.classifier[-1] = last_layer
    elif model_name == "ViT_B_16":
        model.heads = last_layer
    elif model_name in ["DINO_ViT_B_16", "MoCov3_ViT_B_16", "iBOT_ViT_B_16", "Swin_T_14", "iBOT_Swin_T_14", "SparK_convnext_small", "convnext_small"]:
        model.head = last_layer
    else:
        raise NotImplementedError(f"Model specified [{model_name}] not implemented")

    return model


def get_model(model_name, dataset, num_classes, device, is_random=False, model_path=None, requires_grad=False):
    model, num_feats = get_pretrained_model(model_name=model_name, is_random=is_random)
    model = model.to(device)

    if dataset == "cub200":
        if model_name == "iBOT_Swin_T_14" and (not requires_grad):
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
        last_layer = nn.Sequential(
            nn.Linear(num_feats, num_classes) # 200 CUB categories
        )
    elif dataset == "cifar10":
        for param in model.parameters():
            param.requires_grad = requires_grad
        # Change last layer
        last_layer = nn.Sequential(
            nn.Linear(num_feats, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # 10 CIFAR classes
        )
    elif dataset == "oxford102":
        for param in model.parameters():
            param.requires_grad = requires_grad
        # if model_name == "SparK_convnext_small":
        #     # Change last layer
        #     last_layer = nn.Sequential( # newmodel2
        #         nn.Linear(num_feats, num_classes),   # 102 Oxford102 Flower categories
        #     )
        # else:
        # Change last layer
        last_layer = nn.Sequential(
            nn.Linear(num_feats, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),   # 102 Oxford102 Flower categories
        )
    elif dataset == "eurosat":
        for param in model.parameters():
            param.requires_grad = requires_grad
        # Change last layer
        last_layer = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    elif dataset == "caltech101":
        for param in model.parameters():
            param.requires_grad = requires_grad
        # Change last layer
        last_layer = nn.Sequential(
            nn.Linear(num_feats, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes) # 101 Caltech101 categories
        )
    else:
        raise NotImplementedError(f"Dataset specified [{dataset}] not implemented for model [{model_name}]")

    model = get_changed_model(model_name, model, last_layer)
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    return model


def get_optimizer(model, optimizer_name, params):
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **params)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **params)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **params)
    else:
        raise NotImplementedError(f"Optimizer not found:[{params}]. Available optimizers: ['SGD', 'Adam', 'RMSprop']")
    return optimizer


def get_model_feats_logits(model_name, model, inp): # for CVE/SCVE
    classifier = get_classifier_head(model_name, model)
    feats = get_model_feats(model_name, model, inp)
    logits = classifier(feats)
    return {"features": feats, "logits": logits}


def get_model_feats(model_name, model, inp): # for CVE/SCVE
    if model_name in ["ResNet50", "DINO_ResNet50", "MoCov3_ResNet50"]:
        feat_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,

            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4[0],
        )
        feats = feat_layers(inp)
    elif model_name == "DenseNet121":
        x = model.features(inp)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feats = torch.flatten(x, 1)
    elif model_name == "VGG16":
        x = model.features(inp)
        x = model.avgpool(x)
        feats = torch.flatten(x, 1)
    elif model_name == "ViT_B_16": 
        # Paper on "Making Heads or Tails: Towards Semantically Consistent Visual Counterfactuals" says:
        # "Note that any neural network can be divided into such components by selecting an 
        # arbitrary layer to split at. In our setup, we split a network after the final 
        # down-sampling layer [for resnet/vgg]"
        x = model._process_input(inp)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)
        # Classifier "token" as used by standard language architectures
        feats = x[:, 0]
    elif model_name == "Swin_T_14":
        x = model.features(inp)
        x = model.norm(x)
        x = model.permute(x)
        x = model.avgpool(x)
        feats = model.flatten(x)
    elif model_name == "iBOT_Swin_T_14":
        x = model.patch_embed(inp)
        x = x.flatten(2).transpose(1, 2)

        if model.ape:
            x = x + model.absolute_pos_embed
        x = model.pos_drop(x)

        for layer in model.layers:
            x = layer(x)

        x_region = model.norm(x)  # B L C
        x = model.avgpool(x_region.transpose(1, 2))  # B C 1
        feats = torch.flatten(x, 1)
    elif model_name in ["DINO_ViT_B_16", "MoCov3_ViT_B_16", "iBOT_ViT_B_16"]:
        # Following lines from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L209-L214
        x = model.prepare_tokens(inp)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        feats = x[:, 0]
    else:
        raise NotImplementedError(f"No implementation for model [{model_name}]")
    return feats


def get_classifier_head(model_name, model): # for CVE/SCVE
    if model_name in ["ResNet50", "DINO_ResNet50", "MoCov3_ResNet50"]:
        head = nn.Sequential(
            model.layer4[1:],
            model.avgpool,
            nn.Flatten(start_dim=1),
            model.fc,
        )
    elif model_name in ["VGG16", "DenseNet121"]:
        head = model.classifier
    elif model_name == "ViT_B_16":
        head = model.heads
    elif model_name in ["DINO_ViT_B_16", "MoCov3_ViT_B_16", "iBOT_ViT_B_16", "Swin_T_14", "iBOT_Swin_T_14", "SparK_convnext_small", "convnext_small"]:
        head = model.head
    else:
        raise NotImplementedError(f"No implementation for model [{model_name}]")
    return head