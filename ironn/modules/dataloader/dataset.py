
import torch
import cv2
from torch.utils.data import Dataset
import os
import numpy as np
import sys
from torchvision import transforms
import torchvision.transforms.functional as TF
import math
import numbers
import random
from skimage import transform, io
from PIL import Image, ImageOps
# sys.path.append(os.path.abspath('../modules'))

from ..preprocessing.utils import get_image_paths, normalize, get_segmentation_arr
# from ..modules.utils import get_image_paths, normalize, get_segmentation_arr

np.random.seed(1234)
torch.cuda.manual_seed_all(1234)

class CustomTransform(object):
    """
    Class to add transformations to images such as rotation and cropping.
    This allows doing data augmentation to increase the number of images
    by creating multiple images from a single image.
    """
    def __init__(self, size):
        """
        Parameters:
        ----------
        size: int
            input size to rescale the image
        """
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Parameters:
        ----------
        sample: (PIL Image)
            a dict of image and label to be cropped.

        Returns:
        sample: dict
            a dict of image and label after transformation.
        """

        image = sample['image']
        mask = sample['label']
        resize = transforms.Resize(size=(self.size))
        image = resize(image)
        mask = resize(mask)
         # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(224, 224))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        return {'image': image, 'label': mask}

class QioTrain(Dataset):
    """
    Class for loading the training dataset.
    Inherits from the PyTorch Dataset class.
    """
    def __init__(self, rootdir, transform=None, is_val=False, ignore_ids=None):
        """
        Init function for the train dataset class
        Parameters:
        ----------
        rootdir: str
            path to the root directory of the project
        transform:
            a torchvision.transform object containing standard or custom
            transformations for the data
        is_val: bool
            true if the dataset being loaded  from the class is the
            validation set
        ignore_ids: list
            a list of image ids to ignore (example: outliers from iCA)
        """
        self.transform = transform
        self.rootdir = rootdir
        self.traindir = os.path.join(rootdir, "training")
        self.labeldir = os.path.join(rootdir, "labels")
        self.transform = transform
        self.ignore_ids = ignore_ids

    def __getitem__(self, index):
        """
        function to get a single item from the dataset based on the index passed

        Parameters:
        ----------
        index: int
            index of the image from the data folder
        """
        image_paths = get_image_paths(self.traindir, self.ignore_ids)
        img_path = image_paths[index]
        image_name = img_path.split("/")[-1]
        label_path = os.path.join(self.labeldir, image_name.split(".")[0] + ".png")
        img = Image.open(img_path)
        label = Image.open(label_path)
        resize = transforms.Resize(size=(224, 224))
        img = resize(img)
        label = resize(label)

        label = np.array(label)
        img = np.array(img)
        label = label.astype('int')
        img = normalize(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        label = get_segmentation_arr(label, 4)
        img = img.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        sample = {'image': torch.from_numpy(img),
                'label': torch.from_numpy(label)}

        return sample

    def __len__(self):
        """
        returns the length of the dataset (the number of files in the folder)
        """
        path, dirs, files = next(os.walk(self.traindir))
        n = len(files)
        return n # of how many examples(images?)

class QioVal(Dataset):
    def __init__(self, rootdir, transform=None):
        """
        Init function for the validation dataset class
        Parameters:
        ----------
        rootdir: str
            path to the root directory of the project
        transform:
            a torchvision.transform object containing standard or custom
            transformations for the data
        """
        self.transform = transform
        self.rootdir = rootdir
        self.valdir = os.path.join(rootdir, "validation")
        self.labeldir = os.path.join(rootdir, "validation_labels")

    def __getitem__(self, index):
        """
        function to get a single item from the dataset based on the index passed

        Parameters:
        ----------
        index: int
            index of the image from the data folder
        """
        image_paths = get_image_paths(self.valdir)
        img_path = image_paths[index]
        image_name = img_path.split("/")[-1]
        label_path = os.path.join(self.labeldir, image_name.split(".")[0] + ".png")
        img = Image.open(img_path)
        label = Image.open(label_path)
        resize = transforms.Resize(size=(224, 224))
        img = resize(img)
        label = resize(label)
        label = np.array(label)
        # print(label.shape)
        label = label.astype('int')
        img = np.array(img)
        img = normalize(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img),
                'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
        returns the length of the dataset (the number of files in the folder)
        """
        path, dirs, files = next(os.walk(self.valdir))
        n = len(files)
        return n

class QioTest(Dataset):
    def __init__(self, rootdir, transform=None):
        """
        Init function for the train dataset class
        Parameters:
        ----------
        rootdir: str
            path to the root directory of the project
        transform:
            a torchvision.transform object containing standard or custom
            transformations for the data
        """
        self.transform = transform
        self.rootdir = rootdir
        self.testdir = os.path.join(rootdir, "testing")

    def __getitem__(self, index):
        """
        function to get a single item from the dataset based on the index passed

        Parameters:
        ----------
        index: int
            index of the image from the data folder
        """
        test_paths = get_image_paths(self.testdir)
        image_path = test_paths[index]
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = normalize(img, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        img = img.transpose((2, 0, 1))

        img = torch.from_numpy(img)
        sample = {'image': img}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """
        returns the length of the dataset (the number of files in the folder)
        """
        path, dirs, files = next(os.walk(self.testdir))
        n = len(files)
        return n
