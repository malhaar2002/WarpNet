from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import config

class CocoDataset(Dataset):
    def __init__(self, root, transforms=None, subset='train'):
        self.root = root
        self.transforms = transforms
        self.subset = subset  # 'train' or 'test'
        self.imgs = os.listdir(self.root)

    def __getitem__(self, idx):
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}

    def __len__(self):
        return len(self.imgs)
