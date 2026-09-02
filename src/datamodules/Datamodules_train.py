from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import src.datamodules.create_dataset as create_dataset
from typing import Optional


class data_create(LightningDataModule):

    def __init__(self, cfg, fold = None):
        super(data_create, self).__init__()
        self.cfg = cfg
        self.preload = cfg.get("preload", True)
      
    def setup(self, stage: Optional[str] = None):
        folder_path_train = self.cfg.get("train_data_dir", self.cfg.path.pathBase)
        folder_path_val = self.cfg.get("val_data_dir", folder_path_train)
        if hasattr(self, "train"):
            return

        self.train = create_dataset.Train(folder_path_train, self.cfg)
        self.val = create_dataset.Val(folder_path_val, self.cfg)


    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=True, drop_last=self.cfg.get("droplast", False))

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True, shuffle=False, drop_last=False)
