import os
import h5py
from PIL import Image
import torch
from torchvision import transforms
import clip_new as clip
from visiontransfomers import VisionTransformer,Transformer
import numpy as np
import pickle
from tqdm import tqdm
from torch import nn as nn
from transformers import CLIPProcessor, CLIPModel,CLIPImageProcessor,CLIPVisionModel

class MiniClip(nn.Module):
    def __init__(self):
        super().__init__()
        Clip = CLIPModel.from_pretrained("remoteclip").cuda()
        self.vision_model = Clip.vision_model
        self.visual_projection = Clip.visual_projection
    
    
    def forward(self, **x):
        vision_outputs = self.vision_model(**x)

        pooled_output = vision_outputs[1]  # pooled_output
        # print(vision_outputs[0].shape)
        # print(vision_outputs[1].shape)
        image_features = self.visual_projection(pooled_output)
        
        return image_features


class image_CLIP:
    def __init__(self, default_data_path,default_data_processed_path,split="train"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.A_path = os.path.join(default_data_path, split,"A")
        self.B_path = os.path.join(default_data_path, split,"B")
        self.out_path =  os.path.join(default_data_processed_path,split+'_image_attention_pool.pickle')
        # self.image_processor = CLIPImageProcessor.from_pretrained("/root/mj/DiffusionCC/remoteclip")
        # self.image_embedding = MiniClip().cuda()
        self.split = split

    def process_images(self, folder_a_path, folder_b_path, output_hdf5):
        images_a = [f for f in os.listdir(folder_a_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # images_b = [f for f in os.listdir(folder_b_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        data = []
        
        for image_name in tqdm(images_a):
            image_path_a = os.path.join(folder_a_path, image_name)
            image_path_b = os.path.join(folder_b_path, image_name)

            # clip start
            image_a = Image.open(image_path_a)
            image_b = Image.open(image_path_b)
            # x_p = self.image_processor(images=Image.open(image_path_a), return_tensors="pt", padding=True)
            # y_p = self.image_processor(images=Image.open(image_path_b), return_tensors="pt", padding=True)
            # x_p['pixel_values'] = x_p['pixel_values'].cuda()
            # y_p['pixel_values'] = y_p['pixel_values'].cuda()
            # features_a = self.image_embedding(**x_p).cpu().detach().numpy()
            # features_b = self.image_embedding(**y_p).cpu().detach().numpy()
            
            data_all = {
                image_name:{
                    'image_before':np.asarray(image_a).astype(np.float32),
                    'image_after':np.asarray(image_b).astype(np.float32)
                    # 'feat_bef':x_p,
                    # 'feat_aft':y_p,
                    # 'feature_before':features_a.astype(np.float32),
                    # 'feature_after':features_b.astype(np.float32)
                    }
                }
            data.append(data_all)
            #print(data_all)
        with open(self.out_path, 'wb') as file:
            pickle.dump(data, file)
        file.close()
        
            

    def process_folders(self):

        # output_hdf5 = h5py.File(self.out_path, "w")

        self.process_images(self.A_path, self.B_path, self.out_path)
        
        # output_hdf5.close()   
        print(f"处理完成，并将结果保存在 {self.out_path}")

# 使用示例
default_data_path = 'data/images'
default_data_processed_path = 'datasets'
train_processor = image_CLIP(default_data_path,default_data_processed_path,split="train")
train_processor.process_folders()

test_processor = image_CLIP(default_data_path,default_data_processed_path,split="test")
test_processor.process_folders()

test_processor = image_CLIP(default_data_path,default_data_processed_path,split="val")
test_processor.process_folders()
