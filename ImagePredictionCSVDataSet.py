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
    if channels == 1:
        image = Image.open(filepath).convert('YCbCr')
        image, _, _ = image.split()
    else:
        image = Image.open(filepath).convert('RGB')
    return image

MAXIMUM_VALUE = float(4.31)
MINIMUM_VALUE = float(1.096)

class ImagePredictionCSVDataSet(Dataset):
    def __init__(self, dir, csv_path, channels, transforms):
        self.root_dir = dir
        self.channels = channels
        self.transforms = transforms
        self.data_info = pd.read_csv(csv_path, header=0)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)
        self.statistics()

    def __getitem__(self, index):
        single_image_name = os.path.join(self.root_dir,self.image_arr[index])
        img_as_img = load_image(single_image_name, self.channels)
        image = self.transforms(img_as_img)
        target = torch.FloatTensor([float((self.label_arr[index] - MINIMUM_VALUE)/(MAXIMUM_VALUE - MINIMUM_VALUE))])
        return image, target

    def __len__(self):
        return self.data_len

    def statistics(self):
        print('maximum value = ', np.max(self.label_arr))
        self.max = float(np.max(self.label_arr))
        print('minimum value = ', np.min(self.label_arr))
        self.min = float(np.min(self.label_arr))
        print('average value = ', np.mean(self.label_arr))
        self.mean = float(np.mean(self.label_arr))
        print('dispersion value = ', np.std(self.label_arr))
        self.std = float(np.std(self.label_arr))


def make_dataloaders (dataset, batch_size, splitratio = 0.2):
    print(' split ratio ', splitratio)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(splitratio * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    # print(train_indices, val_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    print(train_sampler, valid_sampler)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4,
                                                    sampler=valid_sampler)
    print(train_loader, validation_loader)
    dataloaders = {'train': train_loader, 'val': validation_loader}
    return dataloaders