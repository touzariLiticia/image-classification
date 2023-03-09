import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import imgaug as ia
import imgaug.augmenters as iaa
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms


BN_IMAGES_TRAIN = 111430
BN_IMAGES_TEST = 10130
HEIGHT = 56
WIDTH = 56
CHANNELS = 3


def get_imgaug_seq():
    """This function return an augmentation pipeline """
    seq = iaa.Sequential(
        [
            iaa.Sometimes(
                0.2,
                iaa.OneOf(
                    [
                        iaa.GaussianBlur((0, 0.1)),
                        iaa.CoarseDropout(
                            (0.02, 0.1),
                            size_percent=(0.1, 0.5),
                            per_channel=0.2,
                        ),
                    ]
                ),
            ),
            iaa.LinearContrast((0.8, 1.3)),
            iaa.Sometimes(
                0.1,
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),
            ),
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.0, 1.0))),
        ],
        random_order=True,
    )  # apply augmenters in random order
    return seq


class FaceDataset(Dataset):
    """Face Dataset for training
    Parameters
    ----------
    data : array
        Dataset of images
    labels : array
        Dataset of labels
    transform : callable, optional
        A function/transform that takes in a image and returns a transformed
        version, by default None
    augmentation : callable, optional
        Augmentation function for training images, by default None
    """

    def __init__(self, data, labels, transform=None, augmentation=None):
        self.data = data
        self.targets = labels
        self.transform = transform
        self.augmentation = augmentation

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if (y == 0) and self.augmentation:  # check for minority class
            x = self.augmentation.augment_image(image=x)

        if self.transform:
            x = self.transform(x)

        x = x.transpose(2, 0, 1).astype('float32')
        return x, y

    def __len__(self):
        return len(self.data)


class FaceDatasetTest(Dataset):
    """Face Dataset for testing
    Parameters
    ----------
    data : array
        Dataset of images
    transform : callable, optional
        A function/transform that takes in a image and returns a transformed
        version, by default None
    """

    def __init__(self, data, transform=None):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        x = x.transpose(2, 0, 1).astype('float32')
        return x

    def __len__(self):
        return len(self.data)


def get_train_dataloader(
    data_path,
    labels_path=None,
    sample=True,
    augment=True,
    batch_size=64,
    num_workers=24
):
    """This function returns a dataloader
    Parameters
    ----------
    data_path : str
        Path to training dataset
    labels_path : str
        Path to training labels
    sample : bool
        Apply a WeightedRandomSampler or not
    augment : 
        Apply data augmentation or not
    batch_size : int
        Size of the batch
    num_workers : int
        Number of workers   
    Return: Dataloader
    """
    labels = pd.read_csv(labels_path, header=None, sep=" ")
    labels = np.array(labels[0])

    scene_infile = open(data_path, 'rb')
    scene_image_array = np.fromfile(
        scene_infile, dtype=np.uint8, count=BN_IMAGES_TRAIN*HEIGHT*WIDTH*CHANNELS)
    data = scene_image_array.reshape(
        (BN_IMAGES_TRAIN, HEIGHT, WIDTH, CHANNELS))

    if sample:
        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.unique(labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(
            samples_weight, len(samples_weight), replacement=True)
    else:
        sampler = None

    if augment:
        augmentation = get_imgaug_seq()
    else:
        augmentation = None

    dataset = FaceDataset(data, labels, augmentation=augmentation)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, sampler=sampler)
    return dataloader


def get_test_dataloader(data_path, batch_size=64, num_workers=24):
    """This function returns a dataloader
    Parameters
    ----------
    data_path : str
        Path to test dataset
    batch_size : int
        Size of the batch
    num_workers : int
        Number of workers   
    Return: Dataloader
    """
    scene_infile = open(data_path, 'rb')
    scene_image_array = np.fromfile(
        scene_infile, dtype=np.uint8, count=BN_IMAGES_TEST*HEIGHT*WIDTH*CHANNELS)
    data = scene_image_array.reshape((BN_IMAGES_TEST, HEIGHT, WIDTH, CHANNELS))

    dataset = FaceDatasetTest(data)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)
    return dataloader
