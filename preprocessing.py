# app/preprocessing.py

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_Lab2RGB)

def apply_laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpened = np.uint8(np.clip(image - laplacian, 0, 255))
    return sharpened

# ✅ Your original function modified to work with uploaded files
def preprocess_image_from_bytes(image_bytes):
    try:
        # Read bytes and convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = img_to_array(image)
        img_array = apply_clahe(img_array.astype('uint8'))
        img_array = apply_laplacian_filter(img_array)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        return img_array
    except UnidentifiedImageError:
        print("Skipping file - Unidentified image format.")
        return None
