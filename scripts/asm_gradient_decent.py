# This code executes the gradient descent algorithm based on the Angular spectrum method.
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift
import os
import argparse
import cv2
import time
from torch.utils.tensorboard import SummaryWriter

# The device to run the simulation on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

#Total variation loss function
def total_variation_loss_function(img):
    tv_h = torch.sum(torch.abs(img[:-1, :] - img[1:, :]))
    tv_w = torch.sum(torch.abs(img[:, :-1] - img[:, 1:]))
    return tv_h + tv_w

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
    # print(f"Image saved to {filename}")

def angular_function(width, height, wavelength, z, dx, dy):
    k = 2 * np.pi / wavelength
    fy = torch.fft.fftfreq(height, d=dy, device=device)
    fx = torch.fft.fftfreq(width, d=dx, device=device)
    
    fy,fx = torch.meshgrid(fy, fx, indexing='ij')

    prop_mask = ((wavelength * fx)**2 + (wavelength * fy)**2 <= 1.0)

    kz = torch.sqrt((1.0/wavelength)**2 - fx**2 - fy**2 + 0j)

    H = torch.exp(1j * 2*np.pi*kz * z)
    H *= prop_mask
    return H.to(torch.complex128)

def angular_spectrum_prop(input_wave, transfer_function, cropped_width, cropped_height):
    fft_input = fft2(input_wave)
    propagated_wave = ifft2(fft_input * transfer_function)

    start_y = input_wave.shape[0] // 2 - cropped_height // 2
    start_x = input_wave.shape[1] // 2 - cropped_width // 2
    return propagated_wave[start_y:start_y+cropped_height, start_x:start_x+cropped_width]


def generate_spherical_reference_wave_tensor(width, height, wavelength, z):
    k = 2 * np.pi / wavelength
    cx, cy = width // 2, height // 2
    x = (torch.arange(width, device=device) - cx) 
    y = (torch.arange(height, device=device) - cy) 
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
    parser.add_argument('--wavelength', type=float, default=500e-9, help='Wavelength in meters.')
    parser.add_argument('--z1', type=float, default=0.05, help='Distance from light source to object in meters.')
    parser.add_argument('--z2', type=float, default=0.002, help='Distance from object to hologram plane in meters.')
    parser.add_argument('--dx', type=float, default=5.0e-7, help='Pixel size in x direction in meters.')
    parser.add_argument('--dy', type=float, default=5.0e-7, help='Pixel size in y direction in meters.')
    parser.add_argument('--pad_factor', type=int, default=2, help='Zero-padding factor.')
    parser.add_argument('--max_iter', type=int, default=20000, help='Maximum number of iterations.')
    parser.add_argument('--outdir', type=str, default='output_reconstruction', help='Directory to save output images.')
    parser.add_argument('--tv_weight', type=float, default=1e-8, help='Weight for total variation loss.')

    args = parser.parse_args()

    # Simulation parameters
    wavelength = args.wavelength
    z1 = args.z1
    z2 = args.z2
    dx = args.dx
    dy = args.dy
    pad_factor = args.pad_factor
    max_iter = args.max_iter
    tv_weight = args.tv_weight

    base_output_dir = args.outdir
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    
    # Load the target hologram intensity (ground truth)
    target_intensity_path = "../output/output_gabor/target_gt/asm/hologram_intensity_Z1=0.05_dx=5e-07_man.png"
    target_intensity = read_image(target_intensity_path)
    
    h, w = target_intensity.shape
    padded_height, padded_width = h * pad_factor, w * pad_factor
    
    # TensorBoard Setup
    writer = SummaryWriter(log_dir=f'../runs/asmV2_z1={z1}_tvloss_weight={tv_weight}_maxiter={max_iter}')
    print(f"TensorBoard writer created at: {writer.log_dir}")
    
    # Pre-compute the angular spectrum transfer function
    transfer_function_z2 = angular_function(padded_width, padded_height, wavelength, z2, dx, dy)
    transfer_function_z1_z2 = angular_function(padded_width, padded_height, wavelength, z1 + z2, dx, dy)
    
    # Pre-compute the FFT of the impulse response
    # impulse_response_h = create_fresnel_impulse_response(padded_width, padded_height, wavelength, z2, dx, dy)
    # fft_impulse_response = fft2(impulse_response_h)
    
    # Unknown object amplitude "s"
    # s = torch.ones((h, w), dtype=torch.float64, requires_grad=True, device=device)
    amp_param = torch.ones((h,w), dtype=torch.float64, device=device, requires_grad=True)
    phase_param = torch.rand((h,w), dtype=torch.float64, device=device, requires_grad=True) 

    save_Intensity(amp_param, os.path.join(base_output_dir, "initial_s.png"))

    # Known object's phase
    k = 2 * np.pi / wavelength
    cx, cy = w / 2.0, h / 2.0
    x_coords = (torch.arange(w, device=device) - cx) * dx
    y_coords = (torch.arange(h, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    r_sq = x**2 + y**2
    known_phase = torch.exp(1j * k / (2 * z1) * r_sq).to(torch.complex128)
    save_Intensity(torch.angle(known_phase), os.path.join(base_output_dir, "known_phase.png"))

    # Optimizer(Adam)
    optimizer = torch.optim.Adam([amp_param, phase_param], lr=1e-3)

    start_time = time.time()
    total_time = 0.0


    try:
        for i in range(max_iter):
            iter_start_time = time.time()

            object_wave_complex = (amp_param * torch.exp(1j * phase_param)).to(torch.complex128)
            # object_wave_complex = s.to(torch.complex128) * known_phase

            # Pad to larger grid
            padded_object_wave = torch.zeros((padded_height, padded_width), dtype=torch.complex128, device=device)
            start_y = padded_height // 2 - h // 2
            start_x = padded_width // 2 - w // 2
            padded_object_wave[start_y:start_y+h, start_x:start_x+w] = object_wave_complex

            # Propagation
            propagated_object_wave = angular_spectrum_prop(padded_object_wave, transfer_function_z2, w, h)
            spherical_wave = generate_spherical_reference_wave_tensor(padded_width, padded_height, wavelength, z1 + z2)
            reference_wave_at_hologram = angular_spectrum_prop(spherical_wave, transfer_function_z1_z2, w, h)
            total_wave = propagated_object_wave + reference_wave_at_hologram

            simulated_intensity = torch.abs(total_wave)**2

            # Loss (MSE)
            sim_norm = simulated_intensity / (simulated_intensity.max() + 1e-9)
            tgt_norm = target_intensity / (target_intensity.max() + 1e-9)

            mse_loss = torch.nn.functional.mse_loss(sim_norm, tgt_norm)

            tv_amp = total_variation_loss_function(amp_param)
            tv_phase = total_variation_loss_function(phase_param)
            loss = mse_loss + tv_weight * (tv_amp + tv_phase)
            # loss = torch.nn.functional.mse_loss(sim_norm, tgt_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_end_time = time.time()
            iter_duration = iter_end_time - iter_start_time
            total_time += iter_duration

            # Logging and saving only every 50 iterations
            if (i + 1) % 50 == 0:
                # print(f"Iteration {i+1}, Loss: {loss.item():.6e}")
                # writer.add_scalar('Loss', loss.item(), i)
                # writer.add_scalar('Performance/Time_Iteration', iter_duration, i)
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

                # Save reconstructed amplitude (for TensorBoard)
                save_Intensity(amp_param, os.path.join(base_output_dir, "reconstructed_amplitude_progress.png"))
                amp_normalized = amp_param.detach().cpu().numpy()
                writer.add_image('Reconstructed Amplitude(a)', torch.tensor(amp_normalized, dtype=torch.float32).unsqueeze(0), i)
                # Save reconstructed phase (for TensorBoard)
                save_Intensity(phase_param, os.path.join(base_output_dir, "reconstructed_phase_progress.png"))
                phase_normalized = (phase_param.detach().cpu().numpy() + np.pi) / (2 * np.pi)
                writer.add_image('Reconstructed Phase(phi)', torch.tensor(phase_normalized, dtype=torch.float32).unsqueeze(0), i)

                complex_field_tensor = (amp_param.detach() * torch.exp(1j * phase_param.detach()))
                complex_field_np = complex_field_tensor.cpu().numpy()
                np.save(os.path.join(base_output_dir, "reconstructed_complex_field.npy"), complex_field_np)

                # max_val = np.max(s_normalized)
                # if max_val > 0:
                #     s_normalized = s_normalized / max_val
                # s_tensor = torch.tensor(s_normalized, dtype=torch.float32).unsqueeze(0)
                # writer.add_image('Reconstructed Amplitude(s)', s_tensor, i)

                # Save checkpoint outputs
                # if not os.path.exists("output_reconstruction/Adam_randoms_z1=0.5"):
                #     os.makedirs("output_reconstruction/Adam_randoms_z1=0.5")
                # save_Intensity(s, os.path.join(base_output_dir, "reconstructed_s_progress.png"))
                # save_Intensity(simulated_intensity, os.path.join(base_output_dir, "simulated_hologram_intensity_progress.png"))
                # --- END CHECKPOINT SAVE ---

    except KeyboardInterrupt:
        print("Training interrupted. Saving latest results...")

    total_time = time.time() - start_time
    print(f"Total time for {max_iter} iterations: {total_time:.2f} sec, Average time per iteration: {total_time / max_iter:.4f} sec")
    

    # Final save after loop
    final_complex = amp_param * torch.exp(1j * phase_param)

    amp_final = final_complex.abs()
    save_Intensity(amp_final, os.path.join(base_output_dir, "final_s_plane_amplitude.png"))

    phase_final = torch.angle(final_complex)
    save_Intensity(phase_final, os.path.join(base_output_dir, "final_s_plane_phase.png"))
    # final_s_plane_phase = torch.angle(final_s_plane_wave)
    # normalized_phase = (final_s_plane_phase + np.pi) / (2 * np.pi)
    # phase_array = normalized_phase.detach().cpu().numpy()
    
    # output_phase_filename = os.path.join(base_output_dir, "final_s_plane_phase.png")
    # cv2.imwrite(output_phase_filename, (phase_array * 255).astype(np.uint8))

    final_complex_np = final_complex.detach().cpu().numpy()
    np.save(os.path.join(base_output_dir, "final_s_plane_complex.npy"), final_complex_np)
    
    print(f"Final s-plane phase information saved")
    
    # 2. Save reconstructed amplitude and simulated hologram (.png)
    # save_Intensity(s, os.path.join(base_output_dir, "reconstructed_s.png"))
    save_Intensity(simulated_intensity, os.path.join(base_output_dir, "simulated_hologram_intensity.png"))

    print("Reconstruction complete.")
    writer.close()

if __name__ == "__main__":
    main()
