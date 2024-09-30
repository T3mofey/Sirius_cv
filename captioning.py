#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:01:20 2024

@author: timofey
"""

import torch
import os
from PIL import Image,ImageDraw,ImageFont
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def add_caption_below_image(image, caption):
    # Размеры исходного изображения
    img_width, img_height = image.size

    # Задаем параметры текста и прямоугольника
    font = ImageFont.load_default()  # Используем стандартный шрифт
    padding = 10  # Отступы
    text_height = 40  # Высота прямоугольника под текст

    # Создаем новое изображение (старое + место под текст)
    new_image_height = img_height + text_height + padding * 2
    new_image = Image.new("RGB", (img_width, new_image_height), (255, 255, 255))  # Белый фон
    new_image.paste(image, (0, 0))  # Вставляем старое изображение

    # Создаем объект для рисования и добавляем текст
    draw = ImageDraw.Draw(new_image)
    text_position = (padding, img_height + padding)  # Позиция текста
    draw.text(text_position, caption, font=font, fill="black")  # Черный текст

    return new_image

arr=[]
image_dir="/home/timofey/computer_vision/resulted_images"
output_path="/home/timofey/computer_vision/images_with_captions"
os.makedirs(output_path,exist_ok=True)
folder_path = "/home/timofey/computer_vision/images_with_captions"
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


text = "Product is"

for image in os.listdir(image_dir):
    image_path=os.path.join(image_dir,image)
    raw_image=Image.open(image_path).convert("RGB")
    inputs = processor(raw_image,text=text,return_tensors="pt").to(device)
    out= model.generate(**inputs,num_beams=3, top_p=0.5, temperature=0.7,max_length=80)
    caption=processor.decode(out[0], skip_special_tokens=True)
    print(image)
    output=add_caption_below_image(raw_image, caption=caption)
    save_path=os.path.join(output_path,image)
    output.save(save_path)
    

   

