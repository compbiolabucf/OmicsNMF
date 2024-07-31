from logging import raiseExceptions
import torch, re, random, csv,os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from random import randrange, sample
from sklearn.model_selection import train_test_split
import time
from .data_prep import get_data
from .utils import nmf_U_V
from tqdm import tqdm 
"trainXm trainXmi inferXm train_samples infer_samples mRNA_feat miRNA_feat"
normalize_XS = True

class miRNA2mRNA_Dataset(Dataset):
    def __init__(self,data_dict,train_samples):
        
        self.X1 = data_dict['trainXm']
        self.Y1 = data_dict['trainXmi']
        self.Xt = data_dict['sourceXm']
        self.train_samples = train_samples


    def __len__(self):
        return len(self.train_samples)
    def __getitem__(self, idx):
        sample = self.train_samples[idx]
        x = self.Xt[sample].values
        label = 1
        if sample in self.Y1:
            y = self.Y1[sample].values
            label = 1
        else:
            y = np.random.normal(0,1,20530)
            y = y + np.min(y)
            label = 0
        xs = 0
        s = 0
        return np.float32(np.asarray(x)),np.asarray(s),np.float32(xs),np.float32(y), np.float32(label)

def get_dataloaders(source_path, target_path, batch_size=1, random_state = 1111):
    data_dict = get_data(source_path, target_path)
    X1_samples = data_dict['train_samples']
    Xinfer_samples = data_dict['infer_samples']

    Xt_samples, Xv_samples = train_test_split(X1_samples, train_size=.8, random_state=random_state)
    tds = miRNA2mRNA_Dataset(data_dict,Xt_samples)
    vds = miRNA2mRNA_Dataset(data_dict,Xv_samples)
    test_ds = miRNA2mRNA_Dataset(data_dict,Xinfer_samples)
    traindl =  DataLoader(tds, batch_size = batch_size, shuffle=True)
    validdl = DataLoader(vds, batch_size = batch_size, shuffle=False)
    testdl = DataLoader(test_ds, batch_size = batch_size, shuffle=False)
    return traindl,validdl,testdl,data_dict['source_feat'],data_dict['target_feat']


if __name__ == '__main__':
    td, vd, _, a, b = get_dataloaders("../data/mRNA.csv", "../data/miRNA.csv")
    print(a,b)
    print("Train")
    for i,x in tqdm(enumerate(td)):
        j = 0
        print()
        print(len(x))
        for y in x:
            print(y.shape)
        break
    print("Valid")
    for i,x in tqdm(enumerate(vd)):
        j = 0
        print()
        for y in x:
            print(y.shape)
        break
