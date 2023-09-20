import pandas as pd
import numpy as np
import os
from loguru import logger
import yaml
import json

CKPTS_DIR = "../ckpts"
CONFIG_DIR = "../src/configs"
DATE_FILTER = "2023-02-14"  # "yyyy-mm-dd" or ""yyyy-mm-dd.HH:MM:SS" to filter checkpoints; can set to None to not filter

def get_filtered_files(files):
    filtered_files = []
    for file in files:
        if DATE_FILTER and not file.startswith(DATE_FILTER):
            continue
        filtered_files.append(file)
    return filtered_files

def get_config_data(dataset):
    # Get training configurations
    config_path = os.path.join(CONFIG_DIR, "pred_models", f"{dataset}.yaml")
    # logger.debug(f"config_path: {config_path}")
    with open(config_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.pop("model_names")
    config.pop("out_dir")
    config.pop("datapath")
    return config

def update_metrics_summary(metrics_summary, filtered_files):
    # group by date and time
    file_groups = {}
    for file in filtered_files:
        datetime_key = file.split("_")[0]
        if datetime_key not in file_groups:
            file_groups[datetime_key] = {
                "sorted_ckpts": [],
                "metrics": None,
                "losses": None,
            }
        # For each file group, find checkpoint files, metric file, and loss file
        if file.endswith(".pth"):
            file_groups[datetime_key]["sorted_ckpts"].append(file)
            file_groups[datetime_key]["sorted_ckpts"].sort()
        elif file.endswith("_metrics.txt"):
            # NOTE: assumes single metric file
            file_groups[datetime_key]["metrics"] = file
            metrics_fp = os.path.join(CKPTS_DIR, dataset, model_name, file)
            with open(metrics_fp, "r") as f:
                metrics_data = json.load(f)
            metrics_summary[dataset][model_name] = metrics_data["accuracy"]
            metrics_summary[dataset]["epochs"] = metrics_data["epochs"]
            metrics_summary[dataset]["datetime"] = datetime_key
        elif file.endswith("_losses.txt"):
            file_groups[datetime_key]["losses"] = file
    return metrics_summary


if __name__ == "__main__":
    logger.info(f"Processing training data from {CKPTS_DIR} and {CONFIG_DIR}")

    metrics_summary = {}
    for dir, mid_dir, files in os.walk(CKPTS_DIR):
        if mid_dir != []:
            continue
        dataset = dir.split("/")[-2]
        model_name = dir.split("/")[-1]
        logger.debug(f"dataset: {dataset}, model_name: {model_name}")
        if dataset not in metrics_summary:
            metrics_summary[dataset] = {}
        if model_name not in metrics_summary[dataset]:
            metrics_summary[dataset][model_name] = None
        
        config = get_config_data(dataset)
        metrics_summary[dataset]["config"] = config
        
        filtered_files = get_filtered_files(files)
        metrics_summary = update_metrics_summary(metrics_summary, filtered_files)
        
    metrics_summary = pd.DataFrame.from_dict(metrics_summary)
    metrics_summary.to_csv(f"{DATE_FILTER}_metrics_summary.csv", index=True)
    print(metrics_summary)