import os.path as osp
from typing import List, Optional, Union

import cv2
from PIL import Image

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .utils import scandir


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class CustomDataset(Dataset):

    def __init__(self, opt, transform):
        super(CustomDataset, self).__init__()
        self.opt = opt
        self.transform = transform
        self.gt_folder = opt["dataroot_gt"]

        if "meta_info_file" in self.opt:
            with open(self.opt["meta_info_file"], "r") as fin:
                self.gt_paths = [osp.join(self.gt_folder, line.strip().split(" ")[0]) for line in fin]
        else:
            self.gt_paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):

        # Load gt images. Dimension order: HWC; channel order: BGR
        gt_path = self.gt_paths[index]
        img = cv2.imread(gt_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img, "RGB")

        if self.transform:
            img = self.transform(img)

        return img, 0

    def __len__(self):
        return min(self.opt.get("num_img", float("inf")), len(self.gt_paths))


class CustomVAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

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
        self.resize = opt_dataset["resize"]
        self.patch_size = opt_dataset["patch_size"]
        self.num_workers = opt_dataset["num_workers"]
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.Resize(self.resize),
                                              transforms.RandomCrop(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.Resize(self.resize),
                                            transforms.RandomCrop(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = CustomDataset(
            self.opt_dataset["train"],
            transform=train_transforms,
        )

        self.val_dataset = CustomDataset(
            self.opt_dataset["val"],
            transform=val_transforms,
        )
#       ===============================================================

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
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
