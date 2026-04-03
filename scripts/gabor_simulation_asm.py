import os
import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from utility.optics import read_image, save_intensity, generate_two_point_sources


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


def main():
    #simulation parameters
    wavelength = 500e-9  # 500 nm
    z1 = 0.005            #distance from light source to object
    z2 = 0.0002           #distance from object to hologram plane
    dx = 5.0e-7
    dy = 5.0e-7
    pad_factor = 2
    band_limit = True

    amp_image = read_image("../input/cell/amp_cells_v2.png")
    phase_image = read_image("../input/cell/phase_cell.png")

    original_height, original_width = amp_image.shape
    padded_height, padded_width = original_height * pad_factor, original_width * pad_factor

    padded_object_wave = np.zeros((padded_height, padded_width), dtype=np.complex128)

    start_y, start_x = (padded_height - original_height) // 2, (padded_width - original_width) // 2

    phase_radians = (phase_image * 2 * np.pi)  # assuming input normalized to [0,1]
    object_wave = amp_image * np.exp(1j * phase_radians)

    padded_object_wave[start_y:start_y+original_height, start_x:start_x+original_width] = object_wave

    # Generate and propagate the spherical reference wave
    #if you want to use two point sources as reference wave, uncomment the following line and comment the next line
    # spherical_wave = generate_reference_wave(padded_width, padded_height, wavelength, z1)
    spherical_wave = generate_two_point_sources(padded_width, padded_height, wavelength, z1, offset=150)
    reference_wave_at_hologram = angular_spectrum_propagation(spherical_wave, padded_width, padded_height, wavelength, z1 + z2, dx, dy, band_limit=band_limit)

    I_R = np.abs(reference_wave_at_hologram)**2
    I_R_norm = I_R / np.max(I_R)
    cropped_I_R_norm = I_R_norm[start_y:start_y+original_height, start_x:start_x+original_width]
    save_intensity(cropped_I_R_norm, "../output/output_gabor/cell_original/reference_wave_intensity_1.png")

    propagated_object_wave = angular_spectrum_propagation(padded_object_wave, padded_width, padded_height, wavelength, z2, dx, dy, band_limit=band_limit)

    total_wave = reference_wave_at_hologram + propagated_object_wave

    hologram_intensity = np.abs(total_wave)**2

    cropped_hologram_intensity = hologram_intensity[start_y:start_y+original_height, start_x:start_x+original_width]

    output_filename = f"../output/output_gabor/cell_original/hologram_intensity_Z1={z1}_dx={dx}_cell_1.png"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    save_intensity(cropped_hologram_intensity, output_filename)

    hologram_spectrum_complex = fft2(total_wave)
    hologram_spectrum = np.abs(fftshift(hologram_spectrum_complex))**2
    hologram_spectrum = np.log(1 + hologram_spectrum)

if __name__ == "__main__":
    main()
