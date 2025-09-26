# This code executes the gradient descent algorithm based on the Fresnel diffraction.

import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift
import os
import argparse
import cv2
from torch.utils.tensorboard import SummaryWriter

# The device to run the simulation on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")


def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return torch.tensor(normalized_img, dtype=torch.float64, device=device)


def save_Intensity(intensity_tensor, filename):
    intensity_array = intensity_tensor.detach().cpu().numpy()
    max_val = np.max(intensity_array)
    if max_val > 0:
        normalized_intensity = (intensity_array / max_val) * 255.0
    else:
        normalized_intensity = np.zeros_like(intensity_array)
    output_img = normalized_intensity.astype(np.uint8)
    cv2.imwrite(filename, output_img)
    print(f"Image saved to {filename}")


def create_fresnel_impulse_response(width, height, wavelength, z, dx, dy):
    k = 2 * np.pi / wavelength
    cx, cy = width / 2.0, height / 2.0
    
    x_coords = (torch.arange(width, device=device) - cx) * dx
    y_coords = (torch.arange(height, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    r_sq = x**2 + y**2
    phase = k / (2 * z) * r_sq
    
    h = torch.exp(1j * phase)
    return fftshift(h)


def fresnel_convolution_prop(input_wave, impulse_response_fft, crop_width, crop_height):
    padded_height, padded_width = input_wave.shape
    
    # FFT propagation
    fft_input_wave = fft2(input_wave)
    propagated_spectrum = fft_input_wave * impulse_response_fft
    propagated_wave = ifft2(propagated_spectrum)

    # Center crop
    start_y = padded_height // 2 - crop_height // 2
    start_x = padded_width // 2 - crop_width // 2
    cropped_wave = propagated_wave[start_y:start_y+crop_height, start_x:start_x+crop_width]

    return cropped_wave


def generate_spherical_reference_wave_tensor(width, height, wavelength, z):
    k = 2 * np.pi / wavelength
    cx, cy = width // 2, height // 2
    x = torch.arange(width, device=device) - cx
    y = torch.arange(height, device=device) - cy
    x, y = torch.meshgrid(x, y, indexing='ij')

    r_sq = x**2 + y**2 + z**2
    r = torch.sqrt(r_sq)

    # Avoid division by zero
    r = torch.where(r == 0, torch.tensor(1e-9, dtype=torch.float64, device=device), r)

    wave = torch.exp(1j * k * r) / r
    return wave.to(torch.complex128)


def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Fresnel phase retrieval simulation using gradient descent.")
    parser.add_argument('--wavelength', type=float, default=500e-9, help='Wavelength in meters (e.g., 500e-9 for 500nm).')
    parser.add_argument('--z1', type=float, default=0.5, help='Distance from light source to object in meters.')
    parser.add_argument('--z2', type=float, default=0.2, help='Distance from object to hologram plane in meters.')
    parser.add_argument('--dx', type=float, default=8.0e-6, help='Pixel size in x direction in meters.')
    parser.add_argument('--dy', type=float, default=8.0e-6, help='Pixel size in y direction in meters.')
    parser.add_argument('--pad_factor', type=int, default=2, help='Zero-padding factor.')
    
    args = parser.parse_args()

    # Simulation parameters
    wavelength = args.wavelength
    z1 = args.z1
    z2 = args.z2
    dx = args.dx
    dy = args.dy
    pad_factor = args.pad_factor
    
    # Load the target hologram intensity (ground truth)
    target_intensity_path = "C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\scritps\\output_gabor\\target_gt\\hologram_intensity_Z1=0.04_dx=8e-06_fr.png"
    target_intensity = read_image(target_intensity_path)
    
    h, w = target_intensity.shape
    padded_height, padded_width = h * pad_factor, w * pad_factor
    
    # TensorBoard Setup
    writer = SummaryWriter(log_dir=f'runs/fresnel_retrieval_iterations=500_lr=1e-3_Adam_z1={z1}_z2={z2}_dx={dx}_dy={dy}_wavelength={wavelength}')
    print(f"TensorBoard writer created at: {writer.log_dir}")
    
    # Pre-compute the FFT of the impulse response
    impulse_response_h = create_fresnel_impulse_response(padded_width, padded_height, wavelength, z2, dx, dy)
    fft_impulse_response = fft2(impulse_response_h)
    
    # Unknown object amplitude "s"
    # s = torch.rand((h, w), dtype=torch.float64, requires_grad=True, device=device)
    # Unknown object amplitude "s"
    s = torch.ones((h, w), dtype=torch.float64, requires_grad=True, device=device)
    save_Intensity(s, "initial_s.png")

    # Known object's phase
    k = 2 * np.pi / wavelength
    cx, cy = w / 2.0, h / 2.0
    x_coords = (torch.arange(w, device=device) - cx) * dx
    y_coords = (torch.arange(h, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    r_sq = x**2 + y**2
    known_phase = torch.exp(1j * k / (2 * z1) * r_sq).to(torch.complex128)
    save_Intensity(torch.angle(known_phase), "known_phase.png")
    # s_plane_phase = torch.angle(known_phase)
    # # Normalize the phase for visualization
    # normalized_phase = (s_plane_phase + np.pi) / (2 * np.pi)

    # # Convert to a format that can be saved as an image
    # phase_array = normalized_phase.detach().cpu().numpy()

    # # Save the phase information as an image
    # output_filename = "s_plane_phase.png"
    # cv2.imwrite(output_filename, (phase_array * 255).astype(np.uint8))
    # print(f"Phase information of the s-plane saved to {output_filename}")

    # Optimizer(Adam)
    optimizer = torch.optim.Adam([s], lr=1e-3)

    num_iterations = 500
    for i in range(num_iterations):
        object_wave_complex = s.to(torch.complex128) * known_phase

        # Pad to larger grid
        padded_object_wave = torch.zeros((padded_height, padded_width), dtype=torch.complex128, device=device)
        start_y = padded_height // 2 - h // 2
        start_x = padded_width // 2 - w // 2
        padded_object_wave[start_y:start_y+h, start_x:start_x+w] = object_wave_complex

        # Propagation
        propagated_object_wave = fresnel_convolution_prop(padded_object_wave, fft_impulse_response, w, h)

        spherical_wave = generate_spherical_reference_wave_tensor(padded_width, padded_height, wavelength, z1 + z2)
        reference_wave_at_hologram = fresnel_convolution_prop(spherical_wave, fft_impulse_response, w, h)
        
        total_wave = propagated_object_wave + reference_wave_at_hologram

        # Compute phase at hologram
        phase_at_hologram = torch.angle(total_wave)
        phase_tensor = (phase_at_hologram / (2 * np.pi) + 0.5).unsqueeze(0)  # shape: (1, H, W)
        writer.add_image('Phase at Hologram Plane', phase_tensor.to(torch.float32), i, dataformats='CHW')
        simulated_intensity = torch.abs(total_wave)**2

        # Logging
        sim_norm_for_logging = simulated_intensity / (simulated_intensity.max() + 1e-9)
        sim_tensor_for_logging = sim_norm_for_logging.unsqueeze(0).to(torch.float32)   # shape: (1, H, W) == CHW
        writer.add_image('Simulated Hologram Intensity', sim_tensor_for_logging, i, dataformats='CHW')

        # Loss (MSE)
        sim_norm = simulated_intensity / (simulated_intensity.max() + 1e-9)
        tgt_norm = target_intensity / (target_intensity.max() + 1e-9)
        loss = torch.nn.functional.mse_loss(sim_norm, tgt_norm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('Loss/MSE', loss.item(), i)

        if (i + 1) % 50 == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6e}")
            s_normalized = s.detach().cpu().numpy()
            max_val = np.max(s_normalized)
            if max_val > 0:
                s_normalized = s_normalized / max_val
            s_tensor = torch.tensor(s_normalized, dtype=torch.float32).unsqueeze(0)
            writer.add_image('Reconstructed Amplitude', s_tensor, i)

    # Save result
    final_s_plane_wave = s.to(torch.complex128) * known_phase
    final_s_plane_phase = torch.angle(final_s_plane_wave)
    # Normalize the phase for visualization
    normalized_phase = (final_s_plane_phase + np.pi) / (2 * np.pi)
    # Convert to a format that can be saved as an image
    phase_array = normalized_phase.detach().cpu().numpy()
    
    output_phase_filename = "output_reconstruction/final_s_plane_phase.png"
    cv2.imwrite(output_phase_filename, (phase_array * 255).astype(np.uint8))
    print(f"Final s-plane phase information saved to {output_phase_filename}")

    if not os.path.exists("output_reconstruction"):
        os.makedirs("output_reconstruction")
    save_Intensity(s, "output_reconstruction/reconstructed_s.png")
    save_Intensity(simulated_intensity, "output_reconstruction/simulated_hologram_intensity.png")
    
    print("Reconstruction complete.")
    writer.close()

if __name__ == "__main__":
    main()
