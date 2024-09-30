#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:46:30 2024

@author: timofey
"""

import numpy as np
import os
from PIL import Image
import torch 

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

mask_from_opencv_dir = "/home/timofey/computer_vision/opencv_images"
mask_from_matt_dir="/home/timofey/computer_vision/matt_images"
mask_from_nvidia_dir="/home/timofey/computer_vision/nvidia_images"
mask_from_bria_dir= "/home/timofey/computer_vision/bria_masks"

main_folder=sorted(os.listdir(mask_from_bria_dir))
folder2=sorted(os.listdir(mask_from_opencv_dir))
folder3=sorted(os.listdir(mask_from_nvidia_dir))
folder4=sorted(os.listdir(mask_from_matt_dir))

def compute_iou(mask1, mask2): #Функция для подсчета Intersection over UNion score
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union != 0 else 0
    return iou

def compute_dice(mask1, mask2): #Функция для подсчета DIce-score(F1-score)
    intersection = np.logical_and(mask1, mask2).sum()
    dice = (2 * intersection) / (mask1.sum() + mask2.sum())
    return dice
def load_image_as_binary_mask(image_path, target_size=(328, 246)):
    # Открываем изображение и конвертируем его в чёрно-белый формат
    image = Image.open(image_path).convert("L")
    # Изменяем размер изображения
    image_resized = image.resize(target_size)
    # Преобразуем изображение в NumPy массив
    mask = np.array(image_resized)
    # Преобразуем в бинарную маску (0 - фон, 1 - объект)
    binary_mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    return binary_mask
#Создаем массивы в которых будем хранить результаты 
arr_iou_opencv = []
arr_iou_nvidia = []
arr_iou_matt = []
arr_dice_opencv = []
arr_dice_nvidia = []
arr_dice_matt = []

for image_main,image2,image3,image4 in zip(main_folder,folder2,folder3,folder4):
    main_image_path=os.path.join(mask_from_bria_dir,image_main)
    image_opencv_path=os.path.join(mask_from_opencv_dir,image2)
    image_nvidia_path=os.path.join(mask_from_nvidia_dir,image3)
    image_matt_path=os.path.join(mask_from_matt_dir,image4)
    try:
        #Загружаем маски разных моделей
        mask_from_bria = load_image_as_binary_mask(main_image_path)
        mask_from_opencv = load_image_as_binary_mask(image_opencv_path)
        mask_from_nvidia = load_image_as_binary_mask(image_nvidia_path)
        mask_from_matt = load_image_as_binary_mask(image_matt_path)
        #Сохраняем результаты в массивы 
        arr_iou_opencv.append(compute_iou(mask_from_bria, mask_from_opencv))
        arr_dice_opencv.append(compute_dice(mask_from_bria, mask_from_opencv))
        
        arr_iou_nvidia.append(compute_iou(mask_from_bria, mask_from_nvidia))
        arr_dice_nvidia.append(compute_dice(mask_from_bria, mask_from_nvidia))
        
        arr_iou_matt.append(compute_iou(mask_from_bria, mask_from_matt))
        arr_dice_matt.append(compute_dice(mask_from_bria, mask_from_matt))
        
    except Exception as e:
        print(f"Type of error {e}")
        continue
# Делаем все массивы типа numpy.array
arr_iou_opencv = np.array(arr_iou_opencv)
arr_dice_opencv = np.array(arr_dice_opencv)
arr_iou_nvidia = np.array(arr_iou_nvidia)
arr_dice_nvidia = np.array(arr_dice_nvidia)
arr_iou_matt = np.array(arr_iou_matt)
arr_dice_matt = np.array(arr_dice_matt)
# Выводим средние по результатам 
print(f"Mean IoU score for opencv: {np.mean(arr_iou_opencv)} ")
print(f"Mean Dice score for opencv: {np.mean(arr_dice_opencv)}")
print(f"Mean IoU score for nvidia: {np.mean(arr_iou_nvidia)} ")
print(f"Mean Dice score for nvidia: {np.mean(arr_dice_nvidia)}")
print(f"Mean IoU score for matt: {np.mean(arr_iou_matt)} ")
print(f"Mean Dice score for matt: {np.mean(arr_dice_matt)}")

