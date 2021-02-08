from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    
    def __init__(self, cifar_obj, meta_dict, class_to_idx, transform):
        self.cifar_obj = cifar_obj
        self.meta_dict = meta_dict
        self.class_to_idx = class_to_idx
        self.transform = transform
    
    def __len__(self):
        return len(self.cifar_obj['labels'])
    
    def __getitem__(self, idx):

        img = self.cifar_obj['data'][idx]
        
        img = self.transform(img)

        cifar_label = self.cifar_obj['labels'][idx]
        label_name = self.meta_dict['class_names'][cifar_label]

        label_idx = self.class_to_idx[label_name]

        sample = {"image": img, "label": label_idx}
        
        return sample
