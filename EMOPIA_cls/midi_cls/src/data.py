import os
import pandas as pd
import torch

from torch.utils.data import Dataset

class PEmo_Dataset(Dataset):
    def __init__(self, feature_path, labels, split, cls_type, pad_idx):
        self.pt_dir = feature_path
        self.labels = labels
        self.split = split
        self.cls_type = cls_type
        self.get_fl()
        self.pad_idx = pad_idx

    def get_fl(self):
        if self.split == "TRAIN":
            self.fl = pd.read_csv("../dataset/split/train.csv", index_col=0)
        elif self.split == "VALID":
            self.fl = pd.read_csv("../dataset/split/val.csv", index_col=0)
        elif self.split == "TEST":
            self.fl = pd.read_csv("../dataset/split/test.csv", index_col=0)
        else:
            print("Split should be one of [TRAIN, VALID, TEST]")
        if self.cls_type == "HL":
            self.fl = self.fl[self.fl["label"].isin(('Q1', 'Q3'))]

    def __getitem__(self, index):
        audio_fname = self.fl.iloc[index].name
        label = self.fl.iloc[index]['label']
        if self.cls_type == "AV":
            labels = self.labels.index(label)
        elif self.cls_type == "A":
            if label in ['Q1','Q2']:
                labels = self.labels.index('HA')
            elif label in ['Q3','Q4']:
                labels = self.labels.index('LA')
        elif self.cls_type == "V":
            if label in ['Q1','Q4']:
                labels = self.labels.index('HV')
            elif label in ['Q2','Q3']:
                labels = self.labels.index('LV')
        elif self.cls_type == "HL":
            if label == 'Q1':
                labels = self.labels.index('H')
            elif label == 'Q3':
                labels = self.labels.index('L')
        processed_midi = torch.load(os.path.join(self.pt_dir, audio_fname + ".pt"))
        return processed_midi, labels, audio_fname

    def __len__(self):
        return len(self.fl)
    
    def batch_padding(self, data):
        texts, labels, audio_fname = list(zip(*data))
        max_len = max([len(s) for s in texts])
        texts = [s+[self.pad_idx]*(max_len-len(s)) if len(s) < max_len else s for s in texts]
        return torch.LongTensor(texts), torch.LongTensor(labels), audio_fname