# -*- coding: utf-8 -*-

import re
import os
import sys
import numpy as np
import pickle

from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, opt, case=0):
        # 0:train, 1:dev, 2 :test
        data_root = opt.npy_data_root
        if case == 0:
            now = 'train/'
        elif case == 1:
            now = 'dev/'
        elif case == 2:
            now = 'test1/'
        elif case == 3:
            now = 'test2/'
        data_path = data_root + now
        sens = np.load(data_path+'sens.npy')
        if case < 2:
            rels = np.load(data_path+'relations.npy')
            tags = np.load(data_path+'tags.npy')
            assert len(sens) == len(tags)
            self.data = list(zip(sens, tags, rels))
        else:
            self.data = list(zip(sens, sens, sens))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self)
        return self.data[idx]

