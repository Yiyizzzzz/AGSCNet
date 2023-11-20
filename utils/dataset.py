import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import numpy as np
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.tif'))
        self.transform = transform

    def augment(self, image, flipCode):
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, intex):
        image1_path = self.imgs_path[intex]

        label_path = image1_path.replace('image', 'label')

        image1 = cv2.imread(image1_path)
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = label.reshape(label.shape[0], label.shape[1], 1)


        """
        flipCote = random.choice([-1, 0, 1, 2])
        if flipCote != 2:
            image1 = self.augment(image1, flipCote)
            label = self.augment(label, flipCote)
            label = label.reshape(label.shape[0], label.shape[1], 1)
            ClipImage1 = self.augment(ClipImage1, flipCote)
            Cliplabel = self.augment(Cliplabel, flipCote)
            Cliplabel = Cliplabel.reshape(Cliplabel.shape[0], Cliplabel.shape[1], 1)
        """

        fimage = self.transform(image1)
        flabel = self.transform(label)

        return fimage, flabel

    def __len__(self):
        return len(self.imgs_path)

