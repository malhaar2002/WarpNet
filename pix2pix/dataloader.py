from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
import config

class CocoDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = os.listdir(self.root)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = np.array(Image.open(img_path))
        input_img = img[:, :256, :]
        target_img = img[:, 256:, :]
        augemntations = config.transform_both(image=input_img, image0=target_img)
        input_img, target_img = augemntations["image"], augemntations["image0"]
        input_img = config.transform_only_input(image=input_img)["image"]
        target_img = config.transform_only_mask(image=target_img)["image"]
        
        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)

        return img, target

    def __len__(self):
        return len(self.imgs)