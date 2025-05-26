from torchvision import transforms
from torchvision.datasets import CIFAR10
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
        train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        test_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        if stage == 'fit' or stage is None:
            self.train_set = CIFAR10(
                    root='../datasets', train=True, download=True, transform=train_trans)

        self.test_set = CIFAR10(
                root='../datasets', train=False, download=True, transform=test_trans)
