import argparse
import yaml
import numpy as np
from loguru import logger
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
import saliency.core as saliency
import torch
import torch.nn.functional as F
import pandas as pd
import os

from dataloader.dataset import get_dataset
from dataloader.augmentation import set_dataset_augmentation, get_dataset_augmentations, _apply_op
from utils.model import get_model, get_gradcam_target_layer
from utils.results import save_and_update_results
from utils.transforms import get_similarity, get_reverse_affine_explanations, explain_img, get_reverse_rot_explanation

RANDOM_SEED = 0     # set this to have consistent results over runs
device = "cuda"

parser = argparse.ArgumentParser(description="Run explainability framework")
parser.add_argument("--config_path", type=str, required=True)

np.random.seed(RANDOM_SEED)

# Load config file
args = parser.parse_args()
with open(args.config_path, "r") as stream:
    config = yaml.safe_load(stream)

def ibot_swin_reshape_transform(tensor, height=7, width=7): # See https://github.com/jacobgil/pytorch-grad-cam/issues/84
    result = tensor.reshape(tensor.size(0), 
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def swin_reshape_transform(tensor): # See https://github.com/jacobgil/pytorch-grad-cam/issues/84
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result

def reshape_transform(tensor, height=14, width=14): # for transformers (see https://github.com/jacobgil/pytorch-grad-cam/blob/master/usage_examples/vit_example.py#L55)
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def batch_predict(images, m=None):
    m.eval()

    batch = torch.Tensor(images)
    if batch.shape[1] != 3:
        batch = torch.moveaxis(batch, 3, 1)

    m.to(device)
    batch = batch.to(device)
    
    logits = m(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

conv_layer_outputs = {}
def get_call_model_function(call_model):
    # Higher order function used to specify which model to run saliency with
    def call_model_function(images, call_model_args=None, expected_keys=None):
        # Taken from https://github.com/PAIR-code/saliency/blob/master/Examples_core.ipynb
        images = torch.Tensor(images).to(device)
        if images.shape[1] != 3:
            images = torch.moveaxis(images, 3, 1)

        images.requires_grad = True

        class_idx_str = 'class_idx_str'
        target_class_idx =  call_model_args[class_idx_str]
        output = call_model(images)
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            outputs = output[:,target_class_idx]
            grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs))
            grads = torch.movedim(grads[0], 1, 3)
            gradients = grads.cpu().detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            one_hot = torch.zeros_like(output)
            one_hot[:,target_class_idx] = 1
            call_model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return lambda m, i, o: conv_layer_outputs.update((saliency.base.CONVOLUTION_OUTPUT_GRADIENTS, torch.movedim(o[0], 1, 3).detach().numpy()))
    return call_model_function


def get_cam(gradcam_func, model_name, model, dev):
    target_layers = get_gradcam_target_layer(model_name, model)
    if "vit" in model_name.lower():
        cam = gradcam_func(model=model, target_layers=target_layers, use_cuda=(dev == "cuda:0"), reshape_transform=reshape_transform)
    elif model_name == "iBOT_Swin_T_14":
        cam = gradcam_func(model=model, target_layers=target_layers, use_cuda=(dev == "cuda:0"), reshape_transform=ibot_swin_reshape_transform)
    elif model_name == "Swin_T_14":
        cam = gradcam_func(model=model, target_layers=target_layers, use_cuda=(dev == "cuda:0"), reshape_transform=swin_reshape_transform)
    else: 
        cam = gradcam_func(model=model, target_layers=target_layers, use_cuda=(dev == "cuda:0"))
    return cam


def get_t_explanation(i, model_name, magnitude, sign, t_type, t_inputs, predict_model, predict_cam, target_class, idx2label):
    predict_func = lambda x: batch_predict(x, predict_model)
    call_fn = get_call_model_function(predict_model)

    t_pred_class, t_explanation = explain_img(
        i, 
        t_inputs, 
        t_type, 
        predict_func, 
        idx2label, 
        target_class, 
        config,
        model_name,
        cam=predict_cam, 
        mag=magnitude * sign,
        call_fn=call_fn
    )
    valid_mask = None

    if t_type in ["TranslateX", "TranslateY"]:
        t_ones_mask = np.array(_apply_op(torch.ones(t_inputs.shape), t_type, float(magnitude * sign), fill=0))
        t_ones_mask = t_ones_mask[0]
        trans_val = np.array([0, 0])
        trans_val[0 if t_type=="TranslateX" else 1] += magnitude * sign
        t_explanation, valid_mask = get_reverse_affine_explanations(t_explanation, t_ones_mask, rot_val=0, scale_val=np.array([1, 1]), trans_val=trans_val)
    elif t_type in ["Rotate"]:
        t_ones_mask = np.array(_apply_op(torch.ones(t_inputs.shape), t_type, float(magnitude * sign), fill=0))
        t_ones_mask = t_ones_mask[0]
        t_explanation, valid_mask = get_reverse_rot_explanation(t_explanation, t_ones_mask, rot_val=float(magnitude * sign))
    elif t_type in ["FlipLR"]:
        t_explanation = np.fliplr(t_explanation)
    elif t_type in ["FlipUD"]:
        t_explanation = np.flipud(t_explanation)
    
    return t_pred_class, t_explanation, valid_mask

def main(config, model_name, ckpt_name, model_epochs):
    results_fp = f"{config['out_dir']}/{config['dataset']}/{model_name}/results_{config['start_idx']}_{config['end_idx']}.csv"
    logger.debug(f"{config['method']} run with {config}")
    if config["dataset"] == "cub200":
        idx2label = open("../data/CUB_200_2011/classes.txt", "r").readlines()
        idx2label = [x[:-1].split(" ")[1].split(".")[1] for x in idx2label]
    else:
        idx2label = config['idx2label']

    # Define output directories
    IMG_OUT_DIR = "../image_outputs"
    if not os.path.exists(IMG_OUT_DIR):   # location to save image samples
        os.makedirs(IMG_OUT_DIR)
    if not os.path.exists(f"{IMG_OUT_DIR}/{config['dataset']}"):
        os.makedirs(f"{IMG_OUT_DIR}/{config['dataset']}")
    if not os.path.exists(f"{IMG_OUT_DIR}/{config['dataset']}/{model_name}"):
        os.makedirs(f"{IMG_OUT_DIR}/{config['dataset']}/{model_name}")

    if not os.path.exists(config['out_dir']):
        os.makedirs(config['out_dir'])
    if not os.path.exists(f"{config['out_dir']}/{config['dataset']}"):
        os.makedirs(f"{config['out_dir']}/{config['dataset']}")
    if not os.path.exists(f"{config['out_dir']}/{config['dataset']}/{model_name}"):   # location to save image samples
        os.makedirs(f"{config['out_dir']}/{config['dataset']}/{model_name}")

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    device = torch.device(dev)

    # Define data loaders
    _, testset = get_dataset(dataset=config["dataset"], model_name=model_name)
    num_imgs = len(testset)

    is_grad_method = config["method"] in ["gradcam", "gradcamPP", "ig"]

    # Define model
    model_path = os.path.join("../ckpts", config["dataset"], model_name, ckpt_name)
    model = get_model(model_name, config["dataset"], config["num_classes"], device, model_path=model_path, requires_grad=is_grad_method)
    random_model = get_model(model_name, config["dataset"], config["num_classes"], device, is_random=True, requires_grad=is_grad_method)
    model.eval()
    random_model.eval()

    cam, random_cam = None, None
    if config["method"] in ["gradcam", "gradcamPP"]:
        gradcam_func = GradCAM if config["method"] == "gradcam" else GradCAMPlusPlus
        cam = get_cam(gradcam_func, model_name, model, dev)
        random_cam = get_cam(gradcam_func, model_name, random_model, dev)

    # Load previous results to append to
    results = []
    if config['to_append_results']:
        prev_results = pd.read_csv(results_fp)
        results = prev_results.values.tolist()
    
    testset_idxs = np.arange(len(testset))
    np.random.shuffle(testset_idxs)
    for ctr, i in enumerate(testset_idxs):
        if ctr < config['start_idx']:
            continue
        
        set_dataset_augmentation(testset, "Identity", 0, 1)
        inputs, targets = testset[i]
        target_class = idx2label[targets]

        # Get result for original image
        pred_class, orig_explanation = explain_img(
            i, 
            inputs, 
            "Identity", 
            lambda x: batch_predict(x, model), 
            idx2label, 
            target_class, 
            config, 
            model_name,
            cam=cam, 
            call_fn=get_call_model_function(model)
        )

        # Get result for each of the transformed images
        for t_type in config['transform_types']:
            logger.debug(f"Getting explanation for {t_type} image")
            signed_mags = [0]

            if (t_type != 'random') and (t_type != "mid_ckpt"):
                magnitudes, is_signed = get_dataset_augmentations(testset)[t_type]
                signs = [1, -1] if is_signed else [1]
                signed_mags = [0]
                if len(signs) > 1:
                    signed_mags = torch.Tensor(signs).reshape(-1, 1) @ torch.Tensor(magnitudes).reshape(1, -1)
                    signed_mags = np.random.choice(signed_mags[signed_mags != 0], size=config["mag_per_transform"], replace=False)

            for signed_mag in list(signed_mags):
                magnitude = abs(signed_mag)
                sign = np.sign(signed_mag)
                set_dataset_augmentation(testset, "Identity" if (t_type == "random" or t_type == "mid_ckpt") else t_type, magnitude, sign)
                t_inputs, _ = testset[i]
                if t_type == "random":
                    pred_models, pred_cams = [random_model], [random_cam]
                elif t_type == "mid_ckpt":
                    pred_models, pred_cams = [], []
                    for e in model_epochs:
                        ckpt_prefix = ckpt_name.split(".pth")[0]
                        mid_model_path = os.path.join("../ckpts", config["dataset"], model_name, f"{ckpt_prefix}_e{int(e):03d}.pth")
                        mid_model = get_model(model_name, config["dataset"], config["num_classes"], device, model_path=mid_model_path, requires_grad=is_grad_method)
                        mid_model.eval()
                        pred_models.append(mid_model)
                        
                        mid_cam = None
                        if config["method"] in ["gradcam", "gradcamPP"]:
                            gradcam_func = GradCAM if config["method"] == "gradcam" else GradCAMPlusPlus
                            mid_cam = get_cam(gradcam_func, model_name, mid_model, dev)
                        pred_cams.append(mid_cam)
                else:
                    pred_models, pred_cams = [model], [cam]

                for pred_ctr, (predict_model, predict_cam) in enumerate(zip(pred_models, pred_cams)):
                    if len(pred_models) > 1:   # only happens for mid_ckpt
                        t_type = f"mid_ckpt_e{model_epochs[pred_ctr]:03d}"
                    t_pred_class, t_explanation, valid_mask = get_t_explanation(
                        i, model_name, magnitude, sign, t_type, t_inputs, predict_model, predict_cam, target_class, idx2label
                    )
                    
                    if valid_mask is not None:
                        valid_mask = valid_mask.astype(bool)

                    ssim = get_similarity(orig_explanation, t_explanation, method="ssim", mask=valid_mask)
                    spearman = get_similarity(orig_explanation , t_explanation, method="spearman", mask=valid_mask)

                    # Log results to csv file
                    results, results_df = save_and_update_results(
                        i,
                        results,
                        target_class,
                        pred_class,
                        t_type,
                        magnitude * sign,
                        t_pred_class,
                        ssim,
                        spearman
                    )

        results_df.to_csv(results_fp, index=False)
        logger.info(f"Done marking img {ctr+1:02d}/{len(testset)} [idx={i}, ctr={ctr}]")
        
        if ctr >= config['end_idx']:
            break

if __name__=="__main__":
    model_config_fp = f"configs/trained_models/{config['dataset']}.yaml"
    with open(model_config_fp, "r") as stream:
        model_config = yaml.safe_load(stream)
    model_ckpts = model_config["model_ckpts"]
    models_epochs = model_config["model_epochs"]

    logger.info(f"models_epochs: {models_epochs}, model_ckpts: {model_ckpts}")
    for model_name, ckpt_name in model_ckpts.items():
        logger.info(f"Generating explanations for model: {model_name}, {ckpt_name}")
        main(config, model_name, ckpt_name, models_epochs[model_name])