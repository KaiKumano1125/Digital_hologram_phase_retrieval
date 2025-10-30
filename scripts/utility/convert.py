#convert bmp to png
import cv2

bmp_path = "C:\\Users\\Kai Kumano\\workspace\\Phase_retrieval_algorithm\\output\\output_gabor\\cell\\In-line Hologram.bmp"
png_path = "C:\\Users\\Kai Kumano\\workspace\\Phase_retrieval_algorithm\\output\\output_gabor\\cell\\In-line Hologram.png"

# Read the BMP image
image = cv2.imread(bmp_path)

# Save the image as PNG
cv2.imwrite(png_path, image)
