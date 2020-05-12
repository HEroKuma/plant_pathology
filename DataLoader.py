import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import get_transform
from scipy.io import loadmat


class Data(Dataset):
    def __init__(self, root, resize=(256, 256), phase='train'):
        self.file = pd.read_csv(os.path.join(root, 'train1.csv'))
        self.index = self.file['image_id']
        self.tag = self.file[['healthy', 'multiple_diseases', 'rust', 'scab']]
        self.transform = get_transform(resize, phase)

    def __getitem__(self, index):
        image = self.transform(Image.open(os.path.join('images', self.index[index % len(self.file)]) + '.jpg').convert('RGB'))
        tag = self.tag.values[index % len(self.file)]
        tag = np.where(tag == 1)[0][0]
        tag = torch.tensor(tag)

        return {'X': image, 'Y': tag}

    def __len__(self):
        return len(self.file)


class testData(Dataset):
    def __init__(self, root, resize=(256, 256), phase='test'):
        self.file = pd.read_csv(os.path.join(root, 'test.csv'))
        self.index = self.file['image_id']
        self.transform = get_transform(resize, phase)

    def __getitem__(self, index):
        image = self.transform(Image.open(os.path.join('images', self.index[index % len(self.file)]) + '.jpg').convert('RGB'))
        file_name = self.index[index % len(self.file)]

        return image, file_name

    def __len__(self):
        return len(self.file)


if __name__ == '__main__':
    a = Data('./')
    print(a[380])
