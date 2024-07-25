import torch 
import torchvision
from torch.utils.data import Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]])

        self.transform = transform
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples

class toTensor:
    def __call__(self, sample):
        inputs, targets = sample
        # return torch.from_numpy(inputs), torch.from_numpy(targets)
        return torch.from_numpy(np.asarray(inputs)), torch.from_numpy(np.asarray(targets))

class mulTransform:
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        input *= self.factor 
        return inputs, target

dataset = WineDataset(transform=toTensor())
first = dataset[0]
features, labels = first
print(type(features), type(labels))

composed = torchvision.transforms.Compose([toTensor(), mulTransform(2)])
dataset = WineDataset(transform=composed)
features, labels = first
print(type(features), type(labels))