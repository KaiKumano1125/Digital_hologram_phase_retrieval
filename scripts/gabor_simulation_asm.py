import os
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

def generate_two_point_sources(width, height, wavelength, z, offset=100):
    k = 2 * np.pi / wavelength
    cx , cy = width // 2.0 , height // 2.0
    x = np.arange(width) - cx
    y = np.arange(height) - cy
    x, y = np.meshgrid(x,y)

    r1_sq = (x - offset)**2 + y**2 + z**2
    r2_sq = (x + offset)**2 + y**2 + z**2
    r1 = np.sqrt(r1_sq)
    r2 = np.sqrt(r2_sq)

    wave1 = np.exp(1j * k * r1) / r1
    wave2 = np.exp(1j * k * r2) / r2

    wave = wave1 + wave2
    return wave

# angular spectrum method
def angular_spectrum_propagation(input_wave, width, height, wavelength, z, dx, dy, band_limit=False):
    angular_spectrum = fft2(input_wave)
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(width, d=dx)
    fy = np.fft.fftfreq(height, d=dy)
    fx, fy = np.meshgrid(fx, fy)

    kz_sq = 1.0 - (wavelength * fx)**2 - (wavelength * fy)**2
    prop_mask = kz_sq >= 0

    H = np.zeros_like(angular_spectrum, dtype=np.complex128)
    H[prop_mask] = np.exp(1j * z * k * np.sqrt(kz_sq[prop_mask]))

    #apply band-limiting if enable
    if band_limit:
        u_lim = 1 / (wavelength) * np.sqrt((2 * dx * z)**2 + 1)
        v_lim = 1 / (wavelength) * np.sqrt((2 * dy * z)**2 + 1)

        filter_mask = np.ones_like(H)
        filter_mask[np.abs(fx) > u_lim] = 0
        filter_mask[np.abs(fy) > v_lim] = 0

        H *= filter_mask

    propagated_spectrum = angular_spectrum * H
    propagated_wave = ifft2(propagated_spectrum)

    return propagated_wave

def create_phase_map(width, height, max_phase_rad):
    x = np.arange(width)
    y = np.arange(height)

    phase_map = np.outer(np.ones(height), x / width) * max_phase_rad
    return phase_map


def main():
    #simulation parameters
    wavelength = 500e-9  # 500 nm
    z1 = 0.005            #distance from light source to object
    z2 = 0.0002           #distance from object to hologram plane
    dx = 5.0e-7
    dy = 5.0e-7
    pad_factor = 2
    band_limit = True

    # input file
    # object_filename = "input/cell/phase_cells_v2.png"
    # if object_filename is None:
    #     raise FileNotFoundError(f"Image file '{object_filename}' not found.")

    # try:
    #     original_image = read_image(object_filename)
    # except FileNotFoundError as e :
    #     print(e)
    #     return
    
    amp_image = read_image("../input/cell/amp_cells_v2.png")
    phase_image = read_image("../input/cell/phase_cell.png")

    original_height, original_width = amp_image.shape
    padded_height, padded_width = original_height * pad_factor, original_width * pad_factor

    # Create a zero-padded object wave
    padded_object_wave = np.zeros((padded_height, padded_width), dtype=np.complex128)


    start_y, start_x = (padded_height - original_height) // 2, (padded_width - original_width) // 2
    
    
    phase_radians = (phase_image * 2 * np.pi)  # assuming input normalized to [0,1]
    object_wave = amp_image * np.exp(1j * phase_radians)

    padded_object_wave[start_y:start_y+original_height, start_x:start_x+original_width] = object_wave

    # Generate and propagate the spherical reference wave
    spherical_wave = generate_two_point_sources(padded_width, padded_height, wavelength, z1, offset=150)
    reference_wave_at_hologram = angular_spectrum_propagation(spherical_wave, padded_width, padded_height, wavelength, z1 + z2, dx, dy, band_limit=band_limit)
    #save reference wave intensity
    I_R = np.abs(reference_wave_at_hologram)**2
    I_R_norm = I_R / np.max(I_R)
    cropped_I_R_norm = I_R_norm[start_y:start_y+original_height, start_x:start_x+original_width]
    save_intensity(cropped_I_R_norm, "../output/output_gabor/cell_original/reference_wave_intensity_3.png")

    # Propagate the zero-padded object wave
    propagated_object_wave = angular_spectrum_propagation(padded_object_wave, padded_width, padded_height, wavelength, z2, dx, dy, band_limit=band_limit)

    # Combine the waves to form the total wave field at the hologram plane
    total_wave = reference_wave_at_hologram + propagated_object_wave

    # Compute and save the hologram's intensity
    hologram_intensity = np.abs(total_wave)**2
    
    # Crop the hologram to the original size
    cropped_hologram_intensity = hologram_intensity[start_y:start_y+original_height, start_x:start_x+original_width]

    # gt_amp = np.abs(object_wave)
    # gt_phase = np.angle(object_wave)

    # # gt_amp = gt_amp[start_y:start_y+original_height, start_x:start_x+original_width]
    # # gt_phase = gt_phase[start_y:start_y+original_height, start_x:start_x+original_width]

    # def norm01(x):
    #     return (x - x.min()) / (x.max() - x.min() + 1e-12)
    
    # gt_amp = norm01(gt_amp)
    # gt_phase = norm01(gt_phase) 

    # save_intensity(gt_amp, f"output_gabor/ASM/with_bl/z2=0.002m/z1=0.05m/gt_amp.png")
    # save_intensity(gt_phase, f"output_gabor/ASM/with_bl/z2=0.002m/z1=0.05m/gt_phase.png")

    output_filename = f"../output/output_gabor/cell_original/hologram_intensity_Z1={z1}_dx={dx}_cell_3.png"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    save_intensity(cropped_hologram_intensity, output_filename)

    # # --- Compute and save the hologram's phase ---
    # hologram_phase = np.angle(total_wave)
    
    # # Normalize phase from [-pi, pi] to [0, 1]
    # normalized_hologram_phase = (hologram_phase - (-np.pi)) / (np.pi - (-np.pi))
    
    # # Crop the phase hologram to the original size
    # cropped_hologram_phase = normalized_hologram_phase[start_y:start_y+original_height, start_x:start_x+original_width]
    
    # output_phase_filename = f"output_gabor/hologram_phase_Z1={z1}_dx={dx}.png"
    # save_intensity(cropped_hologram_phase, output_phase_filename)

    # Save the hologram's spectrum for visualization
    hologram_spectrum_complex = fft2(total_wave)
    hologram_spectrum = np.abs(fftshift(hologram_spectrum_complex))**2
    hologram_spectrum = np.log(1 + hologram_spectrum)
    
    # cropped_hologram_spectrum = hologram_spectrum[start_y:start_y+original_height, start_x:start_x+original_width]

    # output_spectrum_filename = f"C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\taiwan_project\\output\\gabor_simulation.py/hologram_spectrum_Z1={z1}_dx={dx}.png"
    # save_intensity(cropped_hologram_spectrum, output_spectrum_filename)
    # save_intensity(cropped_hologram_spectrum, f"output_gabor/hologram_spectrum_Z1={z1}_dx={dx}.png")
if __name__ == "__main__":
    main()
    



