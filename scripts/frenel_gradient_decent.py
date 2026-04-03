# This code executes the gradient descent algorithm based on the Fresnel diffraction.
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift
import os
import argparse
import time
from torch.utils.tensorboard import SummaryWriter
from utility.optics_torch import (
    device, read_image, save_intensity, total_variation_loss_function,
    generate_spherical_reference_wave_tensor
)

print(f"using device: {device}")


def create_fresnel_impulse_response(width, height, wavelength, z, dx, dy):
    k = 2 * np.pi / wavelength
    cx, cy = width / 2.0, height / 2.0

    x_coords = (torch.arange(width, device=device) - cx) * dx
    y_coords = (torch.arange(height, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    r_sq = x**2 + y**2
    phase = k / (2 * z) * r_sq

    h = torch.exp(1j * phase)
    return fftshift(h)


def fresnel_convolution_prop(input_wave, impulse_response_fft, crop_width, crop_height):
    padded_height, padded_width = input_wave.shape

    fft_input_wave = fft2(input_wave)
    propagated_spectrum = fft_input_wave * impulse_response_fft
    propagated_wave = ifft2(propagated_spectrum)

    start_y = padded_height // 2 - crop_height // 2
    start_x = padded_width // 2 - crop_width // 2
    cropped_wave = propagated_wave[start_y:start_y+crop_height, start_x:start_x+crop_width]

    return cropped_wave


def main():
    parser = argparse.ArgumentParser(description="Fresnel phase retrieval simulation using gradient descent.")
    parser.add_argument('--wavelength', type=float, default=500e-9)
    parser.add_argument('--z1', type=float, default=0.05)
    parser.add_argument('--z2', type=float, default=0.002)
    parser.add_argument('--dx', type=float, default=5.0e-7)
    parser.add_argument('--dy', type=float, default=5.0e-7)
    parser.add_argument('--pad_factor', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=10000)
    parser.add_argument('--output_dir', type=str, default='output_reconstruction')
    parser.add_argument('--tv_weight', type=float, default=1e-8)

    args = parser.parse_args()

    wavelength = args.wavelength
    z1 = args.z1
    z2 = args.z2
    dx = args.dx
    dy = args.dy
    pad_factor = args.pad_factor
    max_iter = args.max_iter
    tv_weight = args.tv_weight

    base_output_dir = args.output_dir
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    target_intensity_path = "C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\scritps\\output_gabor\\target_gt\\hologram_intensity_Z1=0.05_dx=5e-07_complex_fr.png"
    target_intensity = read_image(target_intensity_path)

    h, w = target_intensity.shape
    padded_height, padded_width = h * pad_factor, w * pad_factor

    writer = SummaryWriter(log_dir=f'runs/phase_input_z1={z1}_tvloss_weight={tv_weight}_maxiter={max_iter}')
    print(f"TensorBoard writer created at: {writer.log_dir}")

    impulse_response_h = create_fresnel_impulse_response(padded_width, padded_height, wavelength, z2, dx, dy)
    fft_impulse_response = fft2(impulse_response_h)

    amp_param = torch.ones((h, w), dtype=torch.float64, device=device, requires_grad=True)
    phase_param = torch.rand((h, w), dtype=torch.float64, device=device, requires_grad=True)

    save_intensity(amp_param, os.path.join(base_output_dir, "initial_s.png"))

    k = 2 * np.pi / wavelength
    cx, cy = w / 2.0, h / 2.0
    x_coords = (torch.arange(w, device=device) - cx) * dx
    y_coords = (torch.arange(h, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    r_sq = x**2 + y**2
    known_phase = torch.exp(1j * k / (2 * z1) * r_sq).to(torch.complex128)
    save_intensity(torch.angle(known_phase), os.path.join(base_output_dir, "known_phase.png"))

    optimizer = torch.optim.Adam([amp_param, phase_param], lr=1e-3)

    start_time = time.time()
    total_time = 0.0

    try:
        for i in range(max_iter):
            iter_start_time = time.time()

            object_wave_complex = (amp_param * torch.exp(1j * phase_param)).to(torch.complex128)

            padded_object_wave = torch.zeros((padded_height, padded_width), dtype=torch.complex128, device=device)
            start_y = padded_height // 2 - h // 2
            start_x = padded_width // 2 - w // 2
            padded_object_wave[start_y:start_y+h, start_x:start_x+w] = object_wave_complex

            propagated_object_wave = fresnel_convolution_prop(padded_object_wave, fft_impulse_response, w, h)
            spherical_wave = generate_spherical_reference_wave_tensor(padded_width, padded_height, wavelength, z1 + z2)
            reference_wave_at_hologram = fresnel_convolution_prop(spherical_wave, fft_impulse_response, w, h)
            total_wave = propagated_object_wave + reference_wave_at_hologram

            simulated_intensity = torch.abs(total_wave)**2

            sim_norm = simulated_intensity / (simulated_intensity.max() + 1e-9)
            tgt_norm = target_intensity / (target_intensity.max() + 1e-9)

            mse_loss = torch.nn.functional.mse_loss(sim_norm, tgt_norm)

            tv_amp = total_variation_loss_function(amp_param)
            tv_phase = total_variation_loss_function(phase_param)
            loss = mse_loss + tv_weight * (tv_amp + tv_phase)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            total_time += iter_duration

            if (i + 1) % 50 == 0:
                print(f"Iteration {i+1}, MSE_Loss: {mse_loss.item():.6e}, TV_Loss: {tv_amp.item():.6e}, Total_Loss: {loss.item():.6e}")
                writer.add_scalar('Loss/MSE', mse_loss.item(), i)
                writer.add_scalar('Loss/TV', tv_amp.item(), i)
                writer.add_scalar('Performance/Time_Iteration', iter_duration, i)

                if device.type == 'cuda':
                    gpu_mem_allocate = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    writer.add_scalar('Performance/GPU_Memory_Allocated_MB', gpu_mem_allocate, i)
                    cached_mem = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    writer.add_scalar('Performance/GPU_Memory_Cached_MB', cached_mem, i)
                print(f"Time for iteration {i+1}: {iter_duration:.4f} sec, Total time: {total_time:.2f} sec")

                save_intensity(amp_param, os.path.join(base_output_dir, "reconstructed_amplitude_progress.png"))
                amp_normalized = amp_param.detach().cpu().numpy()
                writer.add_image('Reconstructed Amplitude(a)', torch.tensor(amp_normalized, dtype=torch.float32).unsqueeze(0), i)

                save_intensity(phase_param, os.path.join(base_output_dir, "reconstructed_phase_progress.png"))
                phase_normalized = (phase_param.detach().cpu().numpy() + np.pi) / (2 * np.pi)
                writer.add_image('Reconstructed Phase(phi)', torch.tensor(phase_normalized, dtype=torch.float32).unsqueeze(0), i)

                complex_field_tensor = (amp_param.detach() * torch.exp(1j * phase_param.detach()))
                complex_field_np = complex_field_tensor.cpu().numpy()
                np.save(os.path.join(base_output_dir, "reconstructed_complex_field.npy"), complex_field_np)

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest results...")

    total_time = time.time() - start_time
    print(f"Total time for {max_iter} iterations: {total_time:.2f} sec, Average time per iteration: {total_time / max_iter:.4f} sec")

    final_complex = amp_param * torch.exp(1j * phase_param)

    amp_final = final_complex.abs()
    save_intensity(amp_final, os.path.join(base_output_dir, "final_s_plane_amplitude.png"))

    phase_final = torch.angle(final_complex)
    save_intensity(phase_final, os.path.join(base_output_dir, "final_s_plane_phase.png"))

    final_complex_np = final_complex.detach().cpu().numpy()
    np.save(os.path.join(base_output_dir, "final_s_plane_complex.npy"), final_complex_np)

    print(f"Final s-plane phase information saved")
    print("Reconstruction complete.")
    writer.close()

if __name__ == "__main__":
    main()
