#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:17:16 2024

@author: timofey
"""

from PIL import Image
import os
import torch

# Set device for CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import numpy as np
import cv2
from natsort import natsorted
import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 

folder_dir="/home/timofey/computer_vision/sirius_data"

def generate_gradient_texture(width, height, start_color=(200, 200, 200), end_color=(255, 255, 255)):
    """
    Генерация градиентного фона
    width, height — размеры изображения
    start_color — начальный цвет градиента
    end_color — конечный цвет градиента
    """
    base = Image.new('RGB', (width, height), start_color)
    top = Image.new('RGB', (width, height), end_color)

    mask = Image.new('L', (width, height))
    mask_data = []

    for y in range(height):
        alpha = int(255 * (y / height))  # Чем ниже пиксель, тем более виден конечный цвет
        mask_data.extend([alpha] * width)

    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)

    return base

def replace_background(image_path,mask_path,background):
      # reading image
    image=cv2.imread(image_path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
      # reading background and resizing it
    height,width,_=image.shape
    background=background.resize((width,height))
    background=np.array(background)
      # reading mask and inverting it for the background
    mask=Image.open(mask_path).convert("L")
    mask=np.array(mask)
    if np.max(mask) > 1:
     mask = np.where(mask > 128, 255, 0).astype(np.uint8)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask_inv=cv2.bitwise_not(mask)
    object_foreground=cv2.bitwise_and(image,image,mask=mask)
    object_background=cv2.bitwise_and(background,background,mask=mask_inv)
      # adding up back and fore
    result=cv2.add(object_foreground,object_background)
    result_image=Image.fromarray(result)
    return result_image

folder_path = "/home/timofey/computer_vision/resulted_images_nvidia"
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

res_path="/home/timofey/computer_vision/resulted_images_nvidia"
os.makedirs(res_path,exist_ok=True)

grad_texture="/home/timofey/computer_vision/grad_back"
backgr_dir=grad_texture
mask_dir="/home/timofey/computer_vision/nvidia_images"
folder1=natsorted(os.listdir(folder_dir))
folder3=natsorted(os.listdir(mask_dir))

def proccess_images(folder_dir,mask_dir,result_path,background_choice):
    background_colors = {
            'white': ((240, 240, 240), (255, 255, 255)),
            'light_gray': ((200, 200, 200), (220, 220, 220)),
            'gray': ((160, 160, 160), (190, 190, 190)),
            'dark_gray': ((100, 100, 100), (130, 130, 130))
        }
    if background_choice not in background_colors:
        raise ValueError(f" choose the right color from the list {background_colors}")
    start_color, end_color=background_colors[background_choice]
    folder1=natsorted(os.listdir(folder_dir))
    folder3=natsorted(os.listdir(mask_dir))
    valid_extensions=('.jpg','.jpeg','.png')
    for image in folder1:
        if not image.lower().endswith(valid_extensions):
            print(f"Invalid image file: {image}")
            continue
        pic_base=os.path.splitext(image)[0]
        pic_path=os.path.join(folder_dir,image)
        for mask in folder3:
            mask_base=os.path.splitext(mask)[0]
            mask_path=os.path.join(mask_dir,mask)
            if pic_base==mask_base:  
                with Image.open(pic_path) as img:
                    try:
                        width,height=img.size
                    except Exception as e:
                        print(f"COuldn't get image size due to {e}")
                texture=generate_gradient_texture(width, height,start_color=start_color,end_color=end_color)
                try:
                    res_image=replace_background(pic_path,mask_path,texture)
                    saved_path=os.path.join(result_path,image)
                    res_image.save(saved_path)
                except Exception as e:
                    print(f"type of error {e}")
                    continue
                break
proccess_images(folder_dir,mask_dir,res_path,'dark_gray')
