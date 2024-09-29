#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:55:44 2024

@author: timofey
"""

import torch 
import os 
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np

feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

nvidia_segment="/home/timofey/computer_vision/nvidia_images"
os.makedirs(nvidia_segment,exist_ok=True)
folder_dir="/home/timofey/computer_vision/sirius_data"

folder_path = "/home/timofey/computer_vision/nvidia_images"
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
        pic=Image.open(image_path).convert("RGB")
        inputs=feature_extractor(images=pic,return_tensors="pt").to(device)
        with torch.no_grad():
            outputs=model(**inputs)
        logits=outputs.logits
        mask=logits.argmax(dim=1).squeeze().cpu().numpy()
        binary_mask = np.where(mask == 0, 0, 255).astype(np.uint8)
        mask_image=Image.fromarray(binary_mask)
        save_path=os.path.join(nvidia_segment,image)
        mask_image.save(save_path)
    except Exception as e:
        print(f"Type of error {e}")
        continue