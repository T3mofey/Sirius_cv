#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 13:19:02 2024

@author: timofey
"""

from PIL import Image,ImageDraw
import os
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

grad_texture="/home/timofey/computer_vision/grad_back"
os.makedirs(grad_texture,exist_ok=True)
folder_path = "/home/timofey/computer_vision/grad_back"

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
    gradient_image = generate_gradient_texture(256, 256, start_color=(200, 200, 200), end_color=(255, 255, 255))
    save_path=os.path.join(grad_texture,image)
    gradient_image.save(save_path)
  except Exception as e:
    continue
