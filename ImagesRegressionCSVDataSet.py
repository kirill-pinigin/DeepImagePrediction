import pandas as pd
import numpy as np
from PIL import Image
import os

import torch

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def load_image(filepath, channels = 3):
    if channels == 3:
        image = Image.open(filepath).convert('RGB')
    else:
        image = Image.open(filepath).convert('YCbCr')
        image, _, _ = image.split()
    return image

class ImagesRegressionCSVDataSet(Dataset):
    def __init__(self, dir, csv_path, channels, transforms):
        self.root_dir = dir
        self.transforms = transforms
        self.data_info = pd.read_csv(csv_path, header=0)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)
        self.phase = 'train'

    def __getitem__(self, index):
        single_image_name = os.path.join(self.root_dir,self.image_arr[index])
        img_as_img = load_image(single_image_name)
        image = self.transforms[self.phase](img_as_img)
        target = torch.FloatTensor(1)
        target[0] =  float(self.label_arr[index])
        return image, target

    def __len__(self):
        return self.data_len


def make_validation_split(dataset, batch_size, ratio = 0.2):
    dataset_size = len(dataset)
    print(dataset_size)
    indices = list(range(dataset_size))
    split = int(np.floor(ratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size , num_workers=4,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                    sampler=valid_sampler)
    dataloaders = {'train' : train_loader, 'val' : validation_loader}
    return dataloaders