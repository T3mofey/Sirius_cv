#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:11:03 2024

@author: timofey
"""

import torch
import numpy as np
import cv2
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

opencv_segment = "/home/timofey/computer_vision/opencv_images"
os.makedirs(opencv_segment, exist_ok=True)
folder_dir = "/home/timofey/computer_vision/sirius_data"

# Очистка папки с результатами
if os.path.exists(opencv_segment):
    for file_name in os.listdir(opencv_segment):
        file_path=os.path.join(opencv_segment, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

for pic in os.listdir(folder_dir):
    pic_path=os.path.join(folder_dir, pic)
    try:
        # Чтение изображения
        sample_image=cv2.imread(pic_path)
        img=cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    
        # Преобразуем изображение в градации серого
        gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Бинаризация изображения 
        _, thresh=cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
        
        # Поиск контуров
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Создание маски
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours,-1,(255),thickness=cv2.FILLED)
        
        # Преобразование маски в двуцветное изображение (белый объект на черном фоне)
        binary_mask=np.where(mask > 0, 255, 0).astype(np.uint8)
        dst_image=Image.fromarray(binary_mask, mode="L")
        
        # Сохранение маски
        save_path=os.path.join(opencv_segment, pic)
        dst_image.save(save_path)
    except Exception as e:
        print(f"Type of error {e}")
        continue