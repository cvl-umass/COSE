# Code below is from https://pytorch.org/vision/stable/_modules/torchvision/datasets/eurosat.html#EuroSAT
import os
from typing import Callable, Optional
import pandas as pd
import numpy as np

import PIL
from PIL import Image
from torch.utils.data import Dataset
import torchvision


class EuroSAT(Dataset):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,

        rot_vals_deg=None,
        trans_vals=None,
        scales=None,

        to_bgr=False,
        to_rrr=False,

        to_double_data_only=False,   # setting this to True will just double the data length (no transformations)
    ) -> None:
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        self.rot_vals_deg = rot_vals_deg
        self.trans_vals = trans_vals
        self.scales = scales

        self.to_bgr = to_bgr
        self.to_rrr = to_rrr

        self.to_double_data_only = to_double_data_only

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # Load train and test data
        if self.train:
            data_path = os.path.join(self._data_folder, "train_eurosat.csv")
        else:
            data_path = os.path.join(self._data_folder, "test_eurosat.csv")
        self.samples = pd.read_csv(data_path)
        self.len_orig = len(self.samples)
        self.targets = self.samples["target"]

    def __len__(self) -> int:
        if (self.rot_vals_deg is not None) or self.to_double_data_only or self.to_rrr or self.to_bgr:
            return len(self.samples) * 2
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

    def __getitem__(self, idx):
        sample = self.samples.iloc[idx % self.len_orig]
        path = os.path.join(self._data_folder, sample.filepath)
        
        img = Image.open(path)
        width, height = img.size

        if (idx > (self.len_orig-1)):
            if self.rot_vals_deg is not None:
                # Add data augmentations with random rotations/scale/trans defined
                # print(f"self.rot_vals_deg[index]: {self.rot_vals_deg[index]}, self.trans_vals[index]: {list(self.trans_vals[index])}, self.scales[index]: {self.scales[index]}")
                img = torchvision.transforms.functional.affine(
                    img,
                    angle=self.rot_vals_deg[idx],
                    translate=list(self.trans_vals[idx]),
                    scale=self.scales[idx],
                    shear=0,
                )
            elif self.to_rrr:
                rrr_img = np.array(img).astype(float)
                rrr_img[:,:,1] = 0
                rrr_img[:,:,2] = 0
                img = PIL.Image.fromarray(rrr_img.astype(np.uint8)) # convert to PIL image
            elif self.to_bgr:
                rgb_img = np.array(img).astype(float)
                bgr_img = rgb_img[...,::-1]
                img = PIL.Image.fromarray(bgr_img.astype(np.uint8)) # convert to PIL image

        return self.transform(img), sample.target