from unicodedata import name
import pandas as pd
filename='/home/zrf2022/GANs/Merged_ElasticNetExprs.txt'

data=pd.read_table(filename)
print(data)

import torch
import numpy as np
import pandas as pd

import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PatchDataloader(object):
    def __init__(self,train_df, batchsize,mylen=1000):
        train_transform=None
        train_dataset = SurvDataset(train_df, train_transform,mylen)
        self.mylen=mylen
        self.train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=1,drop_last=True)

    def get_loader(self):
        return self.train_dataloader

    def __len__(self):
        return self.mylen


class SurvDataset(Dataset):
    def __init__(self, df, transform,mylen):
        self.df=df
        self.mylen=mylen
        self.transform = transform

    def __getitem__(self, idx):
        thisdata = self.df.iloc[idx]
        patch,ostime,osstate = thisdata.iloc[3:], thisdata.iloc[1],  thisdata.iloc[2]
        patch,ostime,osstate =patch.astype('float32'),ostime.astype('float32'),osstate.astype('float32')
        return  torch.tensor(patch),torch.tensor(ostime),torch.tensor(osstate)

    def __len__(self):
        return self.mylen

if __name__ == '__main__':
    
    filename='/home/zrf2022/GANs/Merged_ElasticNetExprs.txt'
    data=pd.read_table(filename)
    genedataloder=PatchDataloader(data,128)
