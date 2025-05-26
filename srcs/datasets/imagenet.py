from torchvision import transforms
import torchvision
from .template import DataModule as TmpDataModule
from torch.utils.data import random_split
import torch
from pathlib import Path
from typing import List
import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.pipeline.operation import Operation


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


def create_train_loader(train_dataset, num_workers, batch_size, in_memory, resolution, device):
    this_device = device
    train_path = Path(train_dataset)
    assert train_path.is_file()

    res = resolution
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device))
    ]

    order = OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    })

    return loader


def create_val_loader(val_dataset, num_workers, batch_size, resolution, device):
    this_device = device
    val_path = Path(val_dataset)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device)),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device))
    ]

    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    })
    return loader


class DataModule(TmpDataModule):
    def __init__(self,
                 batch_size=512,
                 num_workers=8,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.num_workers == 0:
            self.num_workers = 1

    #def setup(self, stage=None):
    #    normalize = transforms.Normalize(
    #        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #    )
    #    if stage == 'fit' or stage is None:
    #        self.train_set = torchvision.datasets.ImageFolder(
    #                './datasets/ILSVRC/train',
    #                transform=transforms.Compose([transforms.RandomResizedCrop(224),
    #                                              transforms.RandomHorizontalFlip(),
    #                                              transforms.ToTensor(),
    #                                              normalize]))
    #        #self.train_set, _ = random_split(self.train_set, [0.01, 0.99], generator=torch.Generator().manual_seed(42))

    #    self.test_set = torchvision.datasets.ImageFolder(
    #            './datasets/ILSVRC/val',
    #            transform=transforms.Compose([transforms.Resize(256),
    #                                          transforms.CenterCrop(224),
    #                                          transforms.ToTensor(),
    #                                          normalize]))

    def setup(self, stage=None):
        return

    def val_dataloader(self):
        return create_val_loader('../datasets/ffcv_imagenet/val_500_0.5_90.ffcv', self.num_workers, self.batch_size, 256, 'cuda:0')

    def test_dataloader(self):
        return create_val_loader('../datasets/ffcv_imagenet/val_500_0.5_90.ffcv', self.num_workers, self.batch_size, 256, 'cuda:0')

    def train_dataloader(self):
        return create_train_loader('../datasets/ffcv_imagenet/train_500_0.5_90_50000images.ffcv', self.num_workers, self.batch_size, 1, 192, 'cuda:0')

    def full_train_dataloader(self):
        return create_train_loader('../datasets/ffcv_imagenet/train_500_0.5_90.ffcv', self.num_workers, self.batch_size, 1, 192, 'cuda:0')
