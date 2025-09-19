import numpy as np 
import cv2
from scipy.fft import fft2, ifft2, fftshift

#read gray scale image and normalize to [0,1]
def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return normalized_img

#Normalize floar array and saves it as an 8 bit grayscale PNG image.
def save_intensity(intensity_array, filename):
    max_val = np.max(intensity_array)
    if max_val > 0:
        normalized_intensity = (intensity_array / max_val) * 255.0
    else:
        normalized_intensity = np.zeros_like(intensity_array)
    output_img = normalized_intensity.astype(np.uint8)
    cv2.imwrite(filename, output_img)
    print(f"Image saved to {filename}")

#generate spherical reference wave
def generate_reference_wave(width, height, wavelength, z):
    k = 2 * np.pi / wavelength
    cx , cy = width // 2.0 , height // 2.0
    x = np.arange(width) - cx
    y = np.arange(height) - cy
    x, y = np.meshgrid(x,y)

    r_sq = x**2 + y**2 + z**2
    r = np.sqrt(r_sq)

    wave = np.exp(1j * k * r) / r
    return wave

# angular spectrum method
def angular_spectrum_propagation(input_wave, width, height, wavelength, z, dx, dy):
    angular_spectrum = fft2(input_wave)
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(width, d=dx)
    fy = np.fft.fftfreq(height, d=dy)
    fx, fy = np.meshgrid(fx, fy)

    H = np.exp(1j * z * k * np.sqrt(1.0 - (wavelength * fx)**2 - (wavelength * fy)**2))

    propagated_spectrum = angular_spectrum * H

    propagated_wave = ifft2(propagated_spectrum)

    return propagated_wave

def main():
    #simulation parameters
    wavelength = 500e-9  # 500 nm
    z1 = 0.04 #distance from light source to object 
    z2 = 0.2 #distance from object to hologram plane
    dx = 5.0e-7
    dy = 5.0e-7
    pad_factor = 2

    #input file 
    object_filename ="input/Object.bmp"

    try:
        original_image = read_image(object_filename)
    except FileNotFoundError as e :
        print(e)
        return
    
    original_height, original_width = original_image.shape
    padded_height, padded_width = original_height * pad_factor, original_width * pad_factor

    # Create a zero-padded object wave
    padded_object_wave = np.zeros((padded_height, padded_width), dtype=np.complex128)
    start_y, start_x = (padded_height - original_height) // 2, (padded_width - original_width) // 2
    padded_object_wave[start_y:start_y+original_height, start_x:start_x+original_width] = original_image

    # Generate and propagate the spherical reference wave
    spherical_wave = generate_reference_wave(padded_width, padded_height, wavelength, z1)
    reference_wave_at_hologram = angular_spectrum_propagation(spherical_wave, padded_width, padded_height, wavelength, z1 + z2, dx, dy)

    # Propagate the zero-padded object wave
    propagated_object_wave = angular_spectrum_propagation(padded_object_wave, padded_width, padded_height, wavelength, z2, dx, dy)

    # Combine the waves to form the total wave field at the hologram plane
    total_wave = reference_wave_at_hologram + propagated_object_wave

    # --- Compute and save the hologram's intensity ---
    hologram_intensity = np.abs(total_wave)**2
    
    # Crop the hologram to the original size
    cropped_hologram_intensity = hologram_intensity[start_y:start_y+original_height, start_x:start_x+original_width]

    output_filename = f"output_gabor/hologram_intensity_Z1={z1}_dx={dx}.png"
    save_intensity(cropped_hologram_intensity, output_filename)

    # --- Save the hologram's spectrum for visualization ---
    hologram_spectrum_complex = fft2(total_wave)
    hologram_spectrum = np.abs(fftshift(hologram_spectrum_complex))**2
    hologram_spectrum = np.log(1 + hologram_spectrum)
    
    cropped_hologram_spectrum = hologram_spectrum[start_y:start_y+original_height, start_x:start_x+original_width]

    # output_spectrum_filename = f"C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\taiwan_project\\output\\gabor_simulation.py/hologram_spectrum_Z1={z1}_dx={dx}.png"
    # save_intensity(cropped_hologram_spectrum, output_spectrum_filename)
    save_intensity(cropped_hologram_spectrum, f"output_gabor/hologram_spectrum_Z1={z1}_dx={dx}.png")
if __name__ == "__main__":
    main()
    



