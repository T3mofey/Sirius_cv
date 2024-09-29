#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:17:16 2024

@author: timofey
"""

from torchvision import transforms
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch
import torchvision.transforms as T
from transformers import AutoModelForImageSegmentation

# Set device for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import os
from natsort import natsorted

def replace_background(image_path,mask_path,background_path):
    # reading image
  image=cv2.imread(image_path)
  image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # reading background and resizing it
  background=Image.open(background_path).convert("RGB")
  background=background.resize((image.shape[1],image.shape[0]))
  background=np.array(background)
    # reading mask and inverting it for the background
  mask=Image.open(mask_path).convert("L")
  mask=np.array(mask)
  if np.max(mask) > 1:
       mask = np.where(mask > 128, 255, 0).astype(np.uint8)
  mask_inv=cv2.bitwise_not(mask)
  object_foreground=cv2.bitwise_and(image,image,mask=mask)
  object_background=cv2.bitwise_and(background,background,mask=mask_inv)
    # adding up back and fore
  result=cv2.add(object_foreground,object_background)
  result_image=Image.fromarray(result)
  return result_image

folder_path = "/home/timofey/computer_vision/resulted_images"
os.makedirs(folder_path,exist_ok=True)
folder_dir="/home/timofey/computer_vision/sirius_data"

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

res_path="/home/timofey/computer_vision/resulted_images"
os.makedirs(res_path,exist_ok=True)

grad_texture="/home/timofey/computer_vision/grad_back"
backgr_dir=grad_texture
mask_dir="/home/timofey/computer_vision/bria_masks"
folder1=natsorted(os.listdir(folder_dir))
folder2=natsorted(os.listdir(backgr_dir))
folder3=natsorted(os.listdir(mask_dir))



for image,back in zip(folder1,folder2):
    pic_base=os.path.splitext(image)[0]
    pic_path=os.path.join(folder_dir,image)
    background_path=os.path.join(backgr_dir,back)
    for mask in folder3:
        mask_base=os.path.splitext(mask)[0]
        mask_path=os.path.join(mask_dir,mask)
        if pic_base==mask_base:     
            try:
                res_image=replace_background(pic_path,mask_path,background_path)
                saved_path=os.path.join(res_path,image)
                res_image.save(saved_path)
            except Exception as e:
                print(f"type of error {e}")
                continue
