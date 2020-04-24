import os
import pandas as pd
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Data(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.file = pd.read_csv(os.path.join(root, 'train.csv'))
        self.index = self.file['image_id']
        self.tag = self.file[['healthy', 'multiple_diseases', 'rust', 'scab']]

    def __getitem__(self, index):
        image = self.transform(Image.open(os.path.join('images', self.index[index % len(self.file)]) + '.jpg'))
        tag = self.tag.values[index % len(self.file)]
        tag = np.where(tag == 1)[0][0]
        tag = torch.tensor(tag)

        return {'X': image, 'Y': tag}

    def __len__(self):
        return len(self.file)

class testData(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.file = pd.read_csv(os.path.join(root, 'test.csv'))
        self.index = self.file['image_id']

    def __getitem__(self, index):
        image = self.transform(Image.open(os.path.join('images', self.index[index % len(self.file)]) + '.jpg'))
        file_name = self.index[index % len(self.file)]

        return image, file_name

    def __len__(self):
        return len(self.file)