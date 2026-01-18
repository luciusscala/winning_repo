from PIL import Image
import numpy as np
import tensorflow as tf
from datasets import load_dataset

def resize(row):
    img = row['file_path']

    if img.mode != "RGB":
        print("--- forcing RGB ---")
        img = img.convert("RGB")
        
    width, height = img.size

    new_h = 256
    new_w = int(width * (new_h / height))  
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    final_size = 256
    padded = Image.new(
        mode=resized.mode,
        size=(final_size, final_size),
        color=(255, 255, 255) if resized.mode == 'RGB' else 255
    )

    x_offset = (final_size - resized.width) // 2
    y_offset = 0 
    padded.paste(resized, (x_offset, y_offset))

    #convert padded + resized image to numpy array
    img_array = np.array(padded, dtype=np.float64)
    # print(img_array.shape)

    return {'tensors': img_array}

def format_images():
    ds = load_dataset("imageomics/sentinel-beetles")
    ds = ds.map(resize, desc="Processing")
    # ds.save_to_disk("./dataset")

    return ds





