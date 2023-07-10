import os
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch


class ImageDatasets(Dataset):
    def __init__(self, path):
        self.images = [os.path.join(path, img) for img in os.listdir(path)]

        print('"End reading dataset..."')

    def apple_LR(self, img):
        img = img[np.newaxis, :]
        l_img = img[:, 5:60, 15:134, 15:102]
        r_img = img[:, 60:115, 15:134, 15:102]
        return l_img, r_img, img

    def __getitem__(self, index):
        img = nib.load(self.images[index])
        img = np.squeeze(img.get_data())
        pos = img > 0.05
        img = img * pos

        l_img, r_img, raw_img = self.apple_LR(img)
        return torch.from_numpy(l_img), torch.from_numpy(r_img), torch.from_numpy(raw_img)

    def __len__(self):
        return len(self.images)


