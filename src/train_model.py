import json
import argparse
import torch
from loguru import logger
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

from utils.model import get_model, get_optimizer
from dataloader.dataset import get_dataset


parser = argparse.ArgumentParser(description="Train image classification model")
parser.add_argument("--config_path", type=str, required=True)

DATE_STR = datetime.now().strftime("%Y-%m-%d.%H:%M:%S")         # for naming output files

def train_model(train_dl, test_dl, model, optimizer, scheduler, num_epochs, config, model_name, ckpt_interval):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    test_accs, train_accs = [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_losses_epoch = 0
        num_total, num_correct = 0, 0
        for i, (inputs, targets) in enumerate(train_dl):
            targets = targets.to(device)
            inputs = inputs.to(device)
            
            optimizer.zero_grad()
            
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            if config.get("grad_clip"):
                nn.utils.clip_grad_value_(model.parameters(), config["grad_clip"])

            optimizer.step()
            train_losses_epoch += loss.item()
            _, predicted = yhat.max(1)
            num_total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

        train_losses.append(train_losses_epoch/len(train_dl))
        test_loss, test_acc = evaluate_model(test_dl, model)
        scheduler.step(test_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        train_accs.append(num_correct/num_total)
        plot_losses(train_losses, test_losses, epoch+1, config, model_name)  # update plot for loss every epoch
        if (epoch + 1) % ckpt_interval == 0:
            save_model_and_metrics(model, epoch, config, train_losses, test_losses, train_accs, test_accs)
    save_model_and_metrics(model, epoch, config, train_losses, test_losses, train_accs, test_accs)

    return train_losses, test_losses


def evaluate_model(val_dl, model):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    test_loss, num_correct, num_total = 0, 0, 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_dl):
            targets = targets.to(device)
            inputs = inputs.to(device)
            yhat = model(inputs)
            
            loss = criterion(yhat, targets)
            test_loss += loss.item()
            _, predicted = yhat.max(1)
            num_total += targets.size(0)
            num_correct += (predicted == targets).sum().item()

            # logger.debug(f"test_loss: {test_loss}, num_correct: {num_correct}, num_total: {num_total}")
    test_loss = test_loss/len(val_dl)
    accuracy = num_correct/num_total
    return test_loss, accuracy


def plot_losses(train_losses, test_losses, epoch, config, model_name):
    plt.figure()
    plt.plot([i+1 for i in range(epoch)], train_losses, label = "train loss")
    plt.plot([i+1 for i in range(epoch)], test_losses, label = "test loss")
    # plt.yscale("log")
    plt.legend()
    plt.savefig(f"{config['out_dir']}/{config['dataset']}/{model_name}/{DATE_STR}_losses.jpg", bbox_inches='tight')
    plt.close()


def save_model_and_metrics(model, epoch, config, train_losses, test_losses, train_accs, test_accs):
    # Save model
    torch.save(model.state_dict(), f"{config['out_dir']}/{config['dataset']}/{model_name}/{DATE_STR}_e{epoch:03d}.pth")

    metrics = {
        "accuracy": str(test_accs[-1]), # put the latest accuracy for easy referencing
        "epochs": str(epoch+1),
        "train_losses": str(train_losses),
        "test_losses": str(test_losses),
        "train_accs": str(train_accs),
        "test_accs": str(test_accs),
    }
    with open(f"{config['out_dir']}/{config['dataset']}/{model_name}/{DATE_STR}_metrics.txt", "w") as fp:
        json.dump(metrics, fp, indent=4)


def main(config, model_name, device, optimizer_name, ckpt_interval):
    # Define data loaders
    trainset, testset = get_dataset(dataset=config["dataset"], model_name=model_name)
    train_dl = torch.utils.data.DataLoader(
        trainset, batch_size=config["train_batch_size"], shuffle=True#, num_workers=8
        )
    test_dl = torch.utils.data.DataLoader(
        testset, batch_size=config["test_batch_size"], shuffle=False#, num_workers=2
    )

    # Define model
    model = get_model(model_name, config["dataset"], config["model_kwargs"]["num_classes"], device)
    
    # Train and evaluate model
    optimizer = get_optimizer(model, optimizer_name, config['pred_module'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.00001)
    train_losses, test_losses = train_model(
        train_dl, test_dl, model, optimizer, scheduler, config['trainer']['max_epochs'], config, model_name, ckpt_interval,
    )
    test_loss, accuracy = evaluate_model(test_dl, model)
    logger.info(f"test_loss: {test_loss}, accuracy: {accuracy}")
    torch.save(model.state_dict(), f"{config['out_dir']}/{config['dataset']}/{model_name}/{DATE_STR}.pth")


if __name__=="__main__":
    # Load config file
    args = parser.parse_args()
    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    logger.debug(f"Training prediction model with {config}")
    if not os.path.exists(config['out_dir']):
        logger.warning(f"{config['out_dir']} does not exist. Creating directory.")
        os.makedirs(config["out_dir"])
    if not os.path.exists(f"{config['out_dir']}/{config['dataset']}"):
        os.makedirs(f"{config['out_dir']}/{config['dataset']}")

    # Save model checkpoint every ckpt_interval
    ckpt_interval = max(
        1,
        int(np.floor(config["trainer"]["max_epochs"] * config["trainer"]["ckpt_interval_pct"])),
    )

    # Check for GPU
    if torch.cuda.is_available():  
        dev = "cuda:0"
        logger.debug(f"Found GPU. Using: {dev}")
    else:  
        dev = "cpu"
    device = torch.device(dev)
    optimizer_name = config['pred_module'].pop("optimizer")
    for model_name in config["model_names"]:
        logger.info(f"Training model: {model_name}")
        if not os.path.exists(f"{config['out_dir']}/{config['dataset']}/{model_name}"):
            os.makedirs(f"{config['out_dir']}/{config['dataset']}/{model_name}")
        main(config, model_name, device, optimizer_name, ckpt_interval)