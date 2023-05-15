import os.path as osp
from typing import List, Optional, Union

import cv2
from PIL import Image

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .utils import scandir


class Resize(object):
    def __init__(self, resize, resize_small=False):
        self.resize = resize
        self.resize_small = resize_small

    def __call__(self, img):
        w, h = img.width, img.height
        if max(h, w) > self.resize or self.resize_small:
            ratio = self.resize / max(h, w)
            h_new, w_new = round(h * ratio), round(w * ratio)
            img = img.resize((w_new, h_new), Image.Resampling.BICUBIC)
        return img


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.hflip = opt["hflip"] if "hfilp" in opt else None
        self.resize = opt["resize"] if "resize" in opt else None
        self.crop_size = opt["crop_size"] if "crop_size" in opt else None

        self.gt_folder = opt["dataroot_gt"]
        self.enlarge_ratio = opt.get("enlarge_ratio", 1)

        if "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.gt_paths = [
                    osp.join(self.gt_folder,
                             line.strip().split(" ")[0]) for line in fin
                ]
        else:
            self.gt_paths = sorted(
                list(scandir(self.gt_folder, full_path=True)))
        self.gt_paths = self.gt_paths * self.enlarge_ratio

        if self.hflip:
            self.fn_hflip = transforms.RandomHorizontalFlip()
        if self.resize:
            if self.resize["type"] == "long":
                self.fn_resize = Resize(self.resize["size"])
            elif self.resize["type"] == "short":
                self.fn_resize = transforms.Resize(self.resize["size"])
            else:
                raise NotImplementedError
        if self.crop_size:
            self.fn_crop = transforms.RandomCrop(self.crop_size)
        self.fn_totensor = transforms.ToTensor()

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        img = cv2.imread(gt_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, "RGB")

        if self.hflip:
            img = self.fn_hflip(img)
        if self.resize:
            img = self.fn_resize(img)
        if self.crop_size:
            img = self.fn_crop(img)
        img = self.fn_totensor(img)

        return img, 0

    def __len__(self):
        return min(self.opt.get("num_img", float("inf")), len(self.gt_paths))


class CustomVAEDataset(LightningDataModule):
    def __init__(
        self,
        opt_dataset: str,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.opt_dataset = opt_dataset
        self.train_batch_size = opt_dataset["train_batch_size"]
        self.val_batch_size = opt_dataset["val_batch_size"]
        self.num_workers = opt_dataset["num_workers"]
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = CustomDataset(self.opt_dataset["train"])
        self.val_dataset = CustomDataset(self.opt_dataset["val"])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
