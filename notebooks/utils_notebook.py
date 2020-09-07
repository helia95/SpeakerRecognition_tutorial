import pandas as pd
import os
import numpy as np

import torch

def load_enroll(path):
    X = {}
    df = pd.read_csv(path)
    
    for idx, row in df.iterrows():
        path = row['_path']
        spk = row['spk_id']
        th_embd = torch.load(path)        
        X[spk] = th_embd
        
    return X

def load_test(path):
    X = []
    df = pd.read_csv(path)
    
    for idx, row in df.iterrows():
        path = row['_path']
        spk = row['spk_id']
        th_embd = torch.load(path)        
        X.append((th_embd, spk))
        
    return X


#############################
def load_subset(path, spks_to_keep):
    X = []
    labels = []
    
    
    df = pd.read_csv(path)
    idx_to_keep = df['spk_id'].isin(spks_to_keep)
    
    
    for idx, row in df[idx_to_keep].iterrows():
        path = row['_path']
        spk = row['spk_id']
        th_embd = torch.load(path)
        X.append(th_embd.numpy())
        labels.append(spk)
    return np.concatenate(X), labels



