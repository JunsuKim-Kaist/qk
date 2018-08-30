''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class CUSTOM(data.Dataset):
    """`FER2013 Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Test', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('./data/custom.h5', 'r', driver='core')
        # now load the picked numpy arrays

        self.data = self.data['pixel']
        self.labels = self.data['label']
        self.data = np.asarray(self.data)
        self.data = self.data.reshape((140, 10, 48, 48, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]
        imgList = np.empty((3, 44, 44, 0), int)
        concat_axis = 3

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        #print(img.shape)

        tuple_zero = (0,)
        # imgList = np.empty(tuple_zero + img.shape[1:], int)
        #imgList = np.empty((0, 44, 44, 3), int)
        #print(img.shape)
        #print(imgList.shape)
        for frame in range(img.shape[0]):
            if frame == 5:
                imgFrame = Image.fromarray(img[frame,:,:,:])
                #print(imgFrame.size)
                if self.transform is not None:
                    imgFrame = self.transform(imgFrame)
                image_np = np.array(imgFrame)

                return image_np, target


    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
