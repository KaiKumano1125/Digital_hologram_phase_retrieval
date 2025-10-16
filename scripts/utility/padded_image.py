import numpy as np
import cv2
import os

def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return normalized_img

def save_image(array, filename):
    output_img = (array * 255).astype(np.uint8)
    cv2.imwrite(filename, output_img)
    print(f"Image saved to {filename}")

def pad_image(image, pad_factor):
    height, width = image.shape   
    pad_height = height * pad_factor
    pad_width = width * pad_factor

    padded_array = np.ones((pad_height, pad_width), dtype = image.dtype)

    start_y = (pad_height - height) // 2
    start_x = (pad_width - width) // 2

    padded_array[start_y:start_y+height, start_x:start_x+width] = image
    return padded_array

def main():
    input_filename = "C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\scritps\\input\\Object.bmp"
    pad_factor = 2

    try:
        original_image = read_image(input_filename)
    except FileNotFoundError as e:
        print(e)
        return 
    
    padded_image = pad_image(original_image, pad_factor)

    output_filename = "C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\scritps\\output_gabor\\padded_image"
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    save_image(padded_image, os.path.join(output_filename, "Object_padded.bmp"))

if __name__ == "__main__":
    main()
