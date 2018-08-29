''' QIA Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class QIA(data.Dataset):
    """`QIA Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data = h5py.File('../../../data/qia.h5', 'r', driver='core')
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data['Training_pixel']
            self.train_labels = self.data['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((4974, 10, 48, 48, 3))

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data['PublicTest_pixel']
            self.PublicTest_labels = self.data['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((622, 10, 48, 48, 3))

        else:
            self.PrivateTest_data = self.data['PrivateTest_pixel']
            self.PrivateTest_labels = self.data['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((622, 10, 48, 48, 3))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
            imgList = np.empty((3, 44, 44, 0), int)
            concat_axis = 3
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
            imgList = np.empty((10, 3, 44, 44, 0), int)
            concat_axis = 4
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]
            imgList = np.empty((10, 3, 44, 44, 0), int)
            concat_axis = 4

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
                '''
                #print(image_np.shape)
                #image_np = np.swapaxes(np.swapaxes(image_np, 0, 2), 0, 1)
                #print(image_np[:,:,:,np.newaxis].shape)
                #imgList = np.append(imgList, image_np[np.newaxis, :], axis = 0)
                #print("test")
                #print(imgList.shape)
                #print(image_np.shape)
                if self.split == 'Training':
                    imgList = np.append(imgList, image_np[:,:,:,np.newaxis], axis=concat_axis)
                else:
                    imgList = np.append(imgList, image_np[:, :, :, :, np.newaxis], axis=concat_axis)
            #print(imgList.shape)
        if self.split == 'Training':
            img = imgList[:,:,:,0:1]
        else:
            img = imgList[:, :, :, :, 0:1]
        return imgList[:,:,:,:,0:1], target
        '''
        '''
        48, 48
        img = img[:, :, np.newaxis]
        48, 48, 3
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
        '''

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)

