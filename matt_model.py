#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 00:24:21 2024

@author: timofey
"""

import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import os

# Загрузка модели и извлекателя признаков
feature_extractor = SegformerFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Переключаем модель на GPU, если доступно
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Устанавливаем модель в режим оценки

def decode_segmentation_mask(output):
    output=output.logits.argmax(dim=1).squeeze().cpu().numpy()
    mask=np.where(output!=0,255,0).astype(np.uint8)
    return mask
def proccess_image(image_path):
    pic=Image.open(image_path).convert("RGB")
    inputs=feature_extractor(images=pic,return_tensors="pt").to(device)
    with torch.no_grad():
        output=model(**inputs)
    mask=decode_segmentation_mask(output=output)
    mask_image=Image.fromarray(mask,mode="L")
    return mask_image


matt_segment="/home/timofey/computer_vision/matt_images"
os.makedirs(matt_segment,exist_ok=True)
folder_dir="/home/timofey/computer_vision/sirius_data"
folder_path = "/home/timofey/computer_vision/matt_images"
os.makedirs(folder_path,exist_ok=True)

# Clearing folder, checking if it is empty

if os.path.exists(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
else:
    print(f"Folder {folder_path} does not exist.")
for image in os.listdir(folder_dir):
    image_path=os.path.join(folder_dir,image)
    try:
        res=proccess_image(image_path=image_path)
        save_path=os.path.join(matt_segment,image)
        res.save(save_path)
    except Exception as e:
        print(f"Type of error {e}")
        continue
