import h5py
import json
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
 
class CCDataset(Dataset):
    def __init__(self, data_folder, split='train'):
        self.image_file = os.path.join(data_folder,f'{split}_image_attention_pool.pickle')
        self.cation_file = os.path.join(data_folder,f'{split}_caption_encoded.json')
        self.split = split
        

        with open(self.cation_file, 'r') as f:
            self.caption_data = json.load(f)
        
        self.image_list = list(self.caption_data.keys())

        with open(self.image_file, 'rb') as file:
            self.image_data = pickle.load(file)

    
    def __len__(self):#计数所有caption句子数目来作为dataset的长度
        if self.split == 'test':
            return len(self.caption_data)
        else:
            return sum(len(captions) for captions in self.caption_data.values())
        # return len(self.caption_data)
    
    def __getitem__(self, idx):
        if self.split == 'test':
            img_idx = idx
        else:
            # img_idx = idx
            img_idx = idx//5
        image_key = self.image_list[img_idx] #图像文件名
 
        image_info = self.image_data[img_idx]

        image_before = np.transpose(image_info[image_key]['image_before'], (2, 0, 1))
        image_after = np.transpose(image_info[image_key]['image_after'], (2, 0, 1))

        # image_before =image_info[image_key]['feat_bef']
        # image_after = image_info[image_key]['feat_aft']

        
        if self.split == 'test':
            text_info = self.caption_data[image_key]
            return image_before,image_after,np.array(text_info),image_key
        else:
            sentence_count = 0
            for _, captions in self.caption_data.items():
                if idx < sentence_count + len(captions):

                    caption_idx = idx - sentence_count
                    text_info = captions[caption_idx]
                    break
                sentence_count += len(captions)
           
            # text_info = self.caption_data[image_key]
            return image_before,image_after, np.array(text_info),image_key

