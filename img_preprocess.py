import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn as nn
from transformers import CLIPModel


class image_pre:
    def __init__(self, default_data_path, default_data_processed_path, split="train"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.A_path = os.path.join(default_data_path, split, "A")
        self.B_path = os.path.join(default_data_path, split, "B")
        self.out_path = os.path.join(default_data_processed_path, split + '_image_attention_pool.pickle')
        self.split = split

    def process_images(self, folder_a_path, folder_b_path, output_pickle):
        images_a = [f for f in os.listdir(folder_a_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        images_b = [f for f in os.listdir(folder_b_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Extract the numeric part from the filenames and sort
        images_a.sort(key=lambda x: int(os.path.splitext(x)[0]))
        images_b.sort(key=lambda x: int(os.path.splitext(x)[0]))

        data = []
        
        for image_name in tqdm(images_a):
            image_path_a = os.path.join(folder_a_path, image_name)
            image_path_b = os.path.join(folder_b_path, image_name)

            # Load images
            image_a = Image.open(image_path_a)
            image_b = Image.open(image_path_b)

            data_all = {
                image_name: {
                    'image_before': np.asarray(image_a).astype(np.float32),
                    'image_after': np.asarray(image_b).astype(np.float32)
                }
            }
            data.append(data_all)

        # Save data to pickle file
        with open(output_pickle, 'wb') as file:
            pickle.dump(data, file)

    def process_folders(self):
        self.process_images(self.A_path, self.B_path, self.out_path)
        print(f"The result will be saved at {self.out_path}")

# Usage example
default_data_path = 'data/images'
default_data_processed_path = 'datasets'

train_processor = image_pre(default_data_path, default_data_processed_path, split="train")
train_processor.process_folders()

test_processor = image_re(default_data_path, default_data_processed_path, split="test")
test_processor.process_folders()

val_processor = image_re(default_data_path, default_data_processed_path, split="val")
val_processor.process_folders()
