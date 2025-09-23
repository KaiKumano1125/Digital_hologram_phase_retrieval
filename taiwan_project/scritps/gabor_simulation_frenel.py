import numpy as np
import cv2
from scipy.fft import fft2, ifft2, fftshift
import os

def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return normalized_img

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

def create_fresnel_impulse_response(width, height, wavelength, z, dx, dy):
    k = 2 * np.pi / wavelength
    cx, cy = width / 2.0, height / 2.0
    
    x_coords = (np.arange(width) - cx) * dx
    y_coords = (np.arange(height) - cy) * dy
    x, y = np.meshgrid(x_coords, y_coords)
    r_sq = x**2 + y**2
    phase = k / (2 * z) * r_sq
    
    h = np.exp(1j * phase)
    
    return fftshift(h)

def fresnel_convolution_propagation(input_wave, original_width, original_height, wavelength, z, dx, dy):
    padded_width, padded_height = original_width * 2, original_height * 2
    padded_input_wave = np.zeros((padded_height, padded_width), dtype=np.complex128)
    start_y, start_x = original_height // 2, original_width // 2
    padded_input_wave[start_y:start_y+original_height, start_x:start_x+original_width] = input_wave
    
    h = create_fresnel_impulse_response(padded_width, padded_height, wavelength, z, dx, dy)
    
    #Perform the three FFTs
    # F[a]
    fft_input_wave = fft2(padded_input_wave)
    # F[h]
    fft_impulse_response = fft2(h)
    # 3. Multiply the results in the frequency domain
    propagated_spectrum = fft_input_wave * fft_impulse_response
    # 4. Perform the final inverse FFT
    propagated_wave = ifft2(propagated_spectrum)
    # Crop the result to the original size
    cropped_wave = propagated_wave[start_y:start_y+original_height, start_x:start_x+original_width]
    return cropped_wave

def main():
    # Simulation parameters
    wavelength = 500e-9  # 500 nm
    z1 = 0.133            # distance from light source to object
    z2 = 0.4             # distance from object to hologram plane
    dx = 5.0e-7          # pixel size
    dy = 5.0e-7

    # Input file
    object_filename = "input/Man.bmp"

    try:
        original_image = read_image(object_filename)
    except FileNotFoundError as e:
        print(e)
        return
    original_height, original_width = original_image.shape
    # Create the initial object wave (as a complex array)
    object_wave = original_image.astype(np.complex128)
    # Generate and propagate the spherical reference wave
    spherical_wave = generate_reference_wave(original_width, original_height, wavelength, z1)
    reference_wave_at_hologram = fresnel_convolution_propagation(spherical_wave, original_width, original_height, wavelength, z1 + z2, dx, dy)
    # Propagate the object wave
    propagated_object_wave = fresnel_convolution_propagation(object_wave, original_width, original_height, wavelength, z2, dx, dy)
    # Combine the waves to form the total wave field at the hologram plane
    total_wave = reference_wave_at_hologram + propagated_object_wave
    # Compute and save the hologram's intensity
    hologram_intensity = np.abs(total_wave)**2
    output_filename = f"output_gabor/frenel_diffraction/without_bl/z2=0.2m/z1=0.133m/hologram_intensity_Z1={z1}_dx={dx}_Man_fr.png"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    save_intensity(hologram_intensity, output_filename)

if __name__ == "__main__":
    main()
