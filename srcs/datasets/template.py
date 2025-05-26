import pytorch_lightning as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=512,
                 num_workers=8,
                 persistent_workers=True):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        raise NotImplementedError()

    def train_dataloader(self):
        return DataLoader(
                self.train_set,
                shuffle=True,
                pin_memory=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(
                self.test_set,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(
                self.test_set,
                shuffle=False,
                pin_memory=False,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                persistent_workers=self.persistent_workers)
