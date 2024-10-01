# -*- coding: utf-8 -*-
"""cv_task_redone.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1fxctKLn-JZ3KB6kWvdxqzGbo_UaDu61f
"""






"""Удаление фона с помощью briaai"""

from transformers import pipeline
import torch
from transformers import AutoModelForImageSegmentation
from PIL import Image
import os

folder_dir="/home/timofey/computer_vision/sirius_data"

    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_br_segment = AutoModelForImageSegmentation.from_pretrained("briaai/RMBG-1.4",trust_remote_code=True)
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True,device=device)
model_br_segment.to(device)
torch.cuda.empty_cache()
print(device)

torch.cuda.empty_cache()

   

bria_folder = "/home/timofey/computer_vision/bria_proccessed_images"
os.makedirs(bria_folder,exist_ok=True)
masks_folder= "/home/timofey/computer_vision/bria_masks"
os.makedirs(masks_folder,exist_ok=True)
for image in os.listdir(folder_dir):
    image_path = os.path.join(folder_dir, image)
    try:
        pillow_mask = pipe(image_path, return_mask=True)
        pillow_image = pipe(image_path).convert("RGB")

        save_path = os.path.join(bria_folder, image)
        save_masks_path=os.path.join(masks_folder,image)
        # Сохраняем изображение, если это объект PIL
        if isinstance(pillow_image, Image.Image):
            pillow_image.save(save_path)
            pillow_mask.save(save_masks_path)
            
        else:
            print(f"Unexpected type for pillow_image: {type(pillow_image)}")

    except Exception as e:
        print(f"Error processing {image}: {e}")
        continue






