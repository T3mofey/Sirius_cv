#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:59:11 2024

@author: timofey
"""

import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T
import os
import cv2

# Устройство (CUDA или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Загрузка предобученной модели DeepLabV3
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model.eval().to(device)

# Преобразование изображения для модели
def preprocess_image(image):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0).to(device)

# Преобразование маски в черно-белую (двухцветную) маску
def decode_segmentation_mask(output, threshold=0.5):
    # Получаем маску сегментации
    output = output['out'][0].argmax(0).cpu().numpy()

    # Предполагаем, что класс "фон" имеет индекс 0
    # Все пиксели, не являющиеся фоном, будут считаться объектами
    mask = np.where(output != 0, 255, 0).astype(np.uint8)  # Все, что не фон, ставим белым (255), фон - черным (0)
    
    return mask


# Основная функция удаления фона
def remove_background(image_path):
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")
    
    # Подготовка изображения для модели
    input_tensor = preprocess_image(image)

    # Прогон через модель
    with torch.no_grad():
        output = model(input_tensor)

    # Получение сегментационной маски
    mask = decode_segmentation_mask(output)
    
    # Преобразование изображения в numpy для обработки с OpenCV
    image_np = np.array(image)
    
    # Применение маски для выделения объекта
    result = cv2.bitwise_and(image_np, image_np, mask=mask)

    result_image=Image.fromarray(result)
    return result_image

deeplab_segment="/home/timofey/computer_vision/deeplab_images"
os.makedirs(deeplab_segment,exist_ok=True)
folder_dir="/home/timofey/computer_vision/sirius_data"
for image in os.listdir(folder_dir):
    image_path=os.path.join(folder_dir,image)
    try:
        res=remove_background(image_path=image_path)
        save_path=os.path.join(deeplab_segment,image)
        res.save(save_path)
    except Exception as e:
        print(f"Type of error {e}")
        continue

