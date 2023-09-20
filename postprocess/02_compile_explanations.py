import pandas as pd
import numpy as np
from loguru import logger

import os
from datetime import datetime

DATETIME_NOW = datetime.now().strftime("%Y-%m-%d.%H.%M.%S")
OUT_DIR = "../outputs"
SALIENCY_METHODS = [
    "blur_ig", "gradcam", "gradcamPP", "lime", "smoothgrad",
    "guided_ig", "ig"
]
DATASETS = ["caltech101", "cifar10", "cub200", "eurosat", "oxford102"]
MODELS = [
  "DenseNet121",
  "DINO_ResNet50",
  "DINO_ViT_B_16",
  "iBOT_Swin_T_14",
  "iBOT_ViT_B_16",
  "MoCov3_ResNet50",
  "MoCov3_ViT_B_16",
  "ResNet50",
  "Swin_T_14",
  # "VGG16",
  "ViT_B_16",
]

GEOMETRIC_TRANS = ["TranslateX", "TranslateY", "Rotate", "FlipLR", "FlipUD"]    # 5
PHOTOMETRIC_TRANS = ["Brightness", "Color", "Contrast", "Sharpness", "AutoContrast", "Equalize", "Blur", "Smooth"]  # 8
MODEL_TRANS = ["random", "mid_ckpt"]

UNSUPERVISED_TERMS = ["dino", "ibot", "mocov3"]


def get_model_type(model_name):
    feat_type = "supervised"
    model_name = model_name.lower()
    for term in UNSUPERVISED_TERMS:
        if term in model_name:
            feat_type = "unsupervised"
            model_name = model_name.split(term+"_")[-1]
            break
    return feat_type, model_name

def get_metrics(results_df, saliency_method, dataset, model):
    metrics = {
        "geom_consistency": [],
        "photo_consistency": [],
        "overall_consistency": [],

        "geom_fidelity": [],
        "photo_fidelity": [],
        "img_fidelity": [],

        "random_fidelity": [],
        "midckpt_fidelity": [],
        "model_fidelity": [],

        "overall_fidelity": [],

        "geom_mean_consfid": [],

    }
    for idx, row in results_df.iterrows():
        t_type = row["transform_type"]
        ssim = row["ssim"]
        # If model prediction is consistent despite the transformation, 
        # explanation should also be consistent/the same
        if (row["target_class"] == row["pred_class"]) and (
            row["target_class"] == row["t_pred_class"]):
            if t_type in GEOMETRIC_TRANS:
                metrics["geom_consistency"].append(ssim)
            if t_type in PHOTOMETRIC_TRANS:
                metrics["photo_consistency"].append(ssim)
            if (t_type in GEOMETRIC_TRANS) or (t_type in PHOTOMETRIC_TRANS):       # both photometric and geometric
                metrics["overall_consistency"].append(ssim)

        # If model is originally wrong in the prediction (due to random model weights or middle of training or image transforms) 
        # but eventually gets it right, explanation should be different
        elif (row["target_class"] == row["pred_class"]) and (
            row["target_class"] != row["t_pred_class"]):
            # For changes due to model weights
            if t_type == "random":
                metrics["random_fidelity"].append(ssim)
            elif "mid_ckpt" in t_type:
                metrics["midckpt_fidelity"].append(ssim)
            if (t_type == "random") or ("mid_ckpt" in t_type):
                metrics["model_fidelity"].append(ssim)

            # For changes due to image transformation
            if t_type in GEOMETRIC_TRANS:
                metrics["geom_fidelity"].append(ssim)
            if t_type in PHOTOMETRIC_TRANS:
                metrics["photo_fidelity"].append(ssim)
            if (t_type in GEOMETRIC_TRANS) or (t_type in PHOTOMETRIC_TRANS):       # both photometric and geometric
                metrics["img_fidelity"].append(ssim)

            metrics["overall_fidelity"].append(ssim)

    feature_type, base_model = get_model_type(model)
    overall_metrics = {
        "saliency_method": saliency_method,
        "dataset": dataset,
        "model": model,
        "feature_type": feature_type,
        "base_model": base_model,

        "geom_consistency": np.mean(metrics["geom_consistency"]),
        "photo_consistency": np.mean(metrics["photo_consistency"]),
        "overall_consistency": np.mean(metrics["overall_consistency"]),
        # "overall_consistency_25p": np.percentile(metrics["overall_consistency"], 25),
        # "overall_consistency_50p": np.percentile(metrics["overall_consistency"], 50),
        # "overall_consistency_75p": np.percentile(metrics["overall_consistency"], 75),

        "geom_fidelity": np.mean(metrics["geom_fidelity"]),
        "photo_fidelity": np.mean(metrics["photo_fidelity"]),
        "img_fidelity": np.mean(metrics["img_fidelity"]),

        "random_fidelity": np.mean(metrics["random_fidelity"]),
        "midckpt_fidelity": np.mean(metrics["midckpt_fidelity"]),
        "model_fidelity": np.mean(metrics["model_fidelity"]),

        "overall_fidelity": np.mean(metrics["overall_fidelity"]),
    }
    overall_metrics["geom_mean_consfid"] = np.sqrt(overall_metrics["overall_consistency"] * 
                                                    (1 - overall_metrics["overall_fidelity"]))
    return overall_metrics


if __name__ == "__main__":
    logger.info("Processing output csv files from saliency methods")
    start_idx = 0
    summarized_results = []
    for saliency_method in SALIENCY_METHODS:
        if saliency_method in ["lime", "blur_ig", "smoothgrad"]:
            end_idx = 50
        else:
            end_idx = 500

        for dataset in DATASETS:
            for model in MODELS:
                output_fp = os.path.join(OUT_DIR, saliency_method, dataset, model, f"results_{start_idx}_{end_idx}.csv")
                try:
                    results_df = pd.read_csv(output_fp)
                except FileNotFoundError:
                    logger.error(f"Did not see results: {output_fp}")
                    continue
                metrics = get_metrics(results_df, saliency_method, dataset, model)

                # print(metrics)
                summarized_results.append(metrics)
        #         break
        #     break
        # break
    summarized_results_df = pd.DataFrame(summarized_results)
    summarized_results_df.to_csv(f"{DATETIME_NOW}_overall_results.csv", index=True)
    print(summarized_results_df)