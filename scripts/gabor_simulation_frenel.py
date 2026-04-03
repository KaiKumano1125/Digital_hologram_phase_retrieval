import numpy as np
from scipy.fft import fft2, ifft2, fftshift
import os
from utility.optics import read_image, save_intensity, generate_reference_wave, create_phase_map


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

def fresnel_convolution_propagation(input_wave, original_width, original_height, wavelength, z, dx, dy, band_limit=False):
    padded_width, padded_height = original_width * 2, original_height * 2
    padded_input_wave = np.zeros((padded_height, padded_width), dtype=np.complex128)
    start_y, start_x = original_height // 2, original_width // 2
    padded_input_wave[start_y:start_y+original_height, start_x:start_x+original_width] = input_wave

    h = create_fresnel_impulse_response(padded_width, padded_height, wavelength, z, dx, dy)

    fft_input_wave = fft2(padded_input_wave)
    fft_impulse_response = fft2(h)

    if band_limit:
        u_lim = 1 / (wavelength * np.sqrt((2 * dx * z)**2 + 1))
        v_lim = 1 / (wavelength * np.sqrt((2 * dy * z)**2 + 1))

        fx = np.fft.fftfreq(padded_width, d=dx)
        fy = np.fft.fftfreq(padded_height, d=dy)
        fx, fy = np.meshgrid(fx, fy)

        filter_mask = np.ones_like(fft_impulse_response)
        filter_mask[np.abs(fx) > u_lim] = 0.0
        filter_mask[np.abs(fy) > v_lim] = 0.0

        fft_impulse_response *= filter_mask

    propagated_spectrum = fft_input_wave * fft_impulse_response
    propagated_wave = ifft2(propagated_spectrum)
    cropped_wave = propagated_wave[start_y:start_y+original_height, start_x:start_x+original_width]
    return cropped_wave


def main():
    wavelength = 500e-9  # 500 nm
    z1 = 0.05            # distance from light source to object
    z2 = 0.002           # distance from object to hologram plane
    dx = 5.0e-7          # pixel size
    dy = 5.0e-7
    band_limit = True
    max_phase_rad = np.pi / 2

    object_filename = "input/Man.bmp"

    try:
        original_amplitude = read_image(object_filename)
    except FileNotFoundError as e:
        print(e)
        return
    original_height, original_width = original_amplitude.shape

    phase_map = create_phase_map(original_width, original_height, max_phase_rad)

    object_wave = original_amplitude * np.exp(1j * phase_map)

    spherical_wave = generate_reference_wave(original_width, original_height, wavelength, z1)
    reference_wave_at_hologram = fresnel_convolution_propagation(spherical_wave, original_width, original_height, wavelength, z1 + z2, dx, dy, band_limit=band_limit)

    propagated_object_wave = fresnel_convolution_propagation(object_wave, original_width, original_height, wavelength, z2, dx, dy, band_limit=band_limit)

    total_wave = reference_wave_at_hologram + propagated_object_wave

    hologram_intensity = np.abs(total_wave)**2
    output_filename = f"output_gabor/frenel_diffraction/with_bl/z2=0.002m/z1=0.05m/complex/hologram_intensity_Z1={z1}_dx={dx}_object_padded_fr.png"
    if not os.path.exists(os.path.dirname(output_filename)):
        os.makedirs(os.path.dirname(output_filename))
    save_intensity(hologram_intensity, output_filename)

if __name__ == "__main__":
    main()
