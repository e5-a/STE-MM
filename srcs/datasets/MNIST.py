from torchvision import transforms
from torchvision.datasets import MNIST
from .template import DataModule as TmpDataModule


class DataModule(TmpDataModule):
    def __init__(self,
                 batch_size=512,
                 num_workers=8,
                 persistent_workers=True,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers

    def setup(self, stage=None):
        trans = transforms.Compose([
            transforms.ToTensor()
            ])
        if stage == 'fit' or stage is None:
            self.train_set = MNIST(
                    root='../datasets', train=True, download=True, transform=trans)

        self.test_set = MNIST(
                root='../datasets', train=False, download=True, transform=trans)
