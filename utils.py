import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
import re
import numpy as np
from torchvision.transforms import RandomRotation, Resize
from torchvision.io import read_image
from tqdm import *

data_dict = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25,
    'del': 26,
    'space': 27,
    'nothing': 28,
}

def ohe(y: torch.tensor, num_keys = 26):
    output = torch.zeros(y.shape[0], num_keys)
    for i, char in enumerate(y):
        output[i,data_dict[char]] = 1.
    return output


class ASL_Dataset(Dataset):
    def __init__(self, transform = None, img_shape = 64):
        self.path = './ASL_Data/asl_alphabet_train/asl_alphabet_train/'
        self.fnames = []
        for dir in tqdm(os.listdir(self.path)):
            for item in os.listdir(os.path.join(self.path, dir)):
                if dir not in ['nothing','space','del']: self.fnames.append(f'{dir}/{item}')
        self.transform = transform
        self.img_shape = img_shape

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, self.fnames[idx])
        #print(img_path, self.fnames[idx])
        img = torch.tensor(read_image(img_path),dtype=torch.float32)
        img = Resize((self.img_shape,self.img_shape))(img) / 255.0
        if self.transform: img = self.transform(img)
        label = re.search(fr'[A-Za-z]*', self.fnames[idx])[0]
        return img, ohe(np.array([label]))

