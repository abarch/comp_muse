import torch
import os
import numpy as np
from torch.utils.data import Dataset


class EncoderDataSet(Dataset):
    def __init__(self, image_base_dir, transformer=None, target_transformer=None):
        count = -1 # the first entry has index 0
        files = []
        sep = os.sep

        for x in os.walk(image_base_dir):
            count += 1
            if count == 0:
                continue

            path, _, new_files = x
            files.append([path + sep + file for file in new_files])
        self.files = files

    def __len__(self):
        # 0-indexed
        return sum([len(files) for files in self.files])

    def __getitem__(self, item):
        if item >= len(self):
            raise IndexError(f"Index {item} out of range!")
        i_class = 0
        psum = 0  # partial sum (to avoid sum as parameter name)

        while psum + len(self.files[i_class]) <= item:
            psum += len(self.files[i_class])
            i_class += 1

        path = self.files[i_class][item - psum]
        hidden = torch.load(path)

        # we only want to reconstruct one bar, not all.
        # in later approaches this might be changes, so one
        # can index all bars as separate items
        i = np.random.randint(hidden.shape[0])

        # return (x, y)
        return hidden[i], torch.tensor(i_class)


