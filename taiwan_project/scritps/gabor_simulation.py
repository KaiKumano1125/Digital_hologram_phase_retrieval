import numpy as np 
import cv2
from scipy.fft import fft2, ifft2, fftshift, ifftshift

#read gray scale image and normalize to [0,1]
def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return normalized_img

def save_Intensity(intensity_array, filename):
