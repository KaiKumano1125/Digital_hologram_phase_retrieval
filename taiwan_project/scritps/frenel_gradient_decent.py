# This code executes the gradient decent algorithm based on the Frenel diffraction.

from csv import writer
import numpy as np
import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift
import os
import argparse
from PIL import Image
import cv2
#from torch.utils.tensorboard import SummaryWriter

# The device to run the simulation on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"using device: {device}")

def read_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{filename}' not found.")
    normalized_img = img.astype(np.float64) / 255.0
    return torch.tensor(normalized_img, dtype = torch.float64, device = device)

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

def create_frenel_impulse_response(width, height, wavelength, z, dx, dy):
    k = 2 * np.pi / wavelength
    cx, cy = width / 2.0, height / 2.0
    
    x_coords = (torch.arange(width, device=device) - cx) * dx
    y_coords = (torch.arange(height, device=device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='xy')
    r_sq = x**2 + y**2
    phase = k / (2 * z) * r_sq
    
    h = torch.exp(1j * phase)
    
    return fftshift(h)

def frenel_convolution_prop(input_wave, impulse_response_fft, original_width, original_height):

    padded_height, padded_width = input_wave.shape
    start_y, start_x = original_height // 2, original_width // 2
    
    #F[a]
    fft_input_wave = fft2(input_wave)

    #Multiply the results in the frequency domain
    propagated_spectrum = fft_input_wave * impulse_response_fft

    #Perform the final inverse FFT
    propagated_wave = ifft2(propagated_spectrum)

    #Crop the result to the original size
    cropped_wave = propagated_wave[start_y:start_y+original_height, start_x:start_x+original_width]

    return cropped_wave

def generate_spherical_reference_wave_tensor(width, height, wavelength, z):
    """
    Generates a spherical reference wave using PyTorch tensors.
    """
    k = 2 * np.pi / wavelength
    cx , cy = width // 2.0 , height // 2.0
    x = torch.arange(width, device=device) - cx
    y = torch.arange(height, device=device) - cy
    x, y = torch.meshgrid(x, y, indexing='ij')
    
    r_sq = x**2 + y**2 + z**2
    r = torch.sqrt(r_sq)

    wave = torch.exp(1j * k * r) / r
    return wave.to(torch.complex128)


def main():
    #setup argument parser
    parser = argparse.ArgumentParser(description="Fresnel phase retrieval simulation using gradient descent.")
    parser.add_argument('--wavelength', type=float, default=500e-9, help='Wavelength in meters (e.g., 500e-9 for 500nm).')
    parser.add_argument('--z1', type=float, default=0.04, help='Distance from light source to object in meters.')
    parser.add_argument('--z2', type=float, default=0.2, help='Distance from object to hologram plane in meters.')
    parser.add_argument('--dx', type=float, default=8.0e-6, help='Pixel size in x direction in meters.')
    parser.add_argument('--dy', type=float, default=8.0e-6, help='Pixel size in y direction in meters.')
    parser.add_argument('--pad_factor', type=int, default=2, help='Zero-padding factor.')

    args = parser.parse_args()

    #simulation parameters
    wavelength = args.wavelength
    z1 = args.z1            #distance from light source to object 
    z2 = args.z2             #distance from object to hologram plane
    dx = args.dx
    dy = args.dy
    pad_factor = args.pad_factor
    original_dims = (1024, 1024)
    padded_height, padded_width = original_dims[0] * pad_factor, original_dims[1] * pad_factor

    #Load the target hologram intensity (This is ground truth I)
    target_intensity_path = "C:\\Users\\Kai Kumano\\workspace\\Taiwan_phase_retrieval_algorithm\\taiwan_project\\scritps\\output_gabor\\target_gt\\hologram_intensity_Z1=0.04_dx=8e-06_fr.png"
    target_intensity = read_image(target_intensity_path)

    #Pre-compute thre FFT of the impulse response 
    impulse_response_h = create_frenel_impulse_response(padded_width, padded_height, wavelength, z2, dx, dy)
    fft_impulse_response = fft2(impulse_response_h)

    ###Reconstruction process###
    #1 Define the unknown object amplitude "s"
    s = torch.rand(original_dims, dtype = torch.float64, requires_grad = True, device = device)

    #2 Define the known: the object's phase component (exp)
    k = 2 * np.pi / wavelength
    cx , cy = original_dims[1] // 2.0 , original_dims[0] // 2.0
    x_coords = (torch.arange(original_dims[1], device = device) - cx) * dx
    y_coords = (torch.arange(original_dims[0], device = device) - cy) * dy
    x, y = torch.meshgrid(x_coords, y_coords, indexing='ij')
    r_sq = x**2 + y**2
    known_phase = torch.exp(1j * k / (2 * z1) * r_sq).to(torch.complex128)

    #Optimizer
    # optimizer = torch.optim.Adam([s], lr = 0.01)
    optimizer = torch.optim.SGD([s], lr = 1e-2)

    num_iterations = 100
    for i in range(num_iterations):
        # Propagate the object wave and spherical wave
        object_wave = s.to(torch.complex128) * known_phase
        padded_object_wave = torch.zeros((padded_height, padded_width), dtype = torch.complex128, device = device)
        start_y, start_x = original_dims[0] // 2, original_dims[1] // 2
        padded_object_wave[start_y:start_y+original_dims[0], start_x:start_x+original_dims[1]] = object_wave

        propagated_object_wave = frenel_convolution_prop(padded_object_wave, fft_impulse_response, padded_width, padded_height)

        spherical_wave = generate_spherical_reference_wave_tensor(padded_width, padded_height, wavelength, z1 + z2)
        reference_wave_hologram = frenel_convolution_prop(spherical_wave, fft_impulse_response, padded_width, padded_height)

        #Superposition of the two waves
        total_wave = propagated_object_wave + reference_wave_hologram

        #Calculate the intensity of the propagated wave
        simulated_intensity = torch.abs(total_wave)**2

        #Crop the simulated intensity
        cropped_simulated_intensity = simulated_intensity[start_y:start_y+original_dims[0], start_x:start_x+original_dims[1]]

        #Calculate the loss(MSE)
        loss = torch.nn.functional.mse_loss(cropped_simulated_intensity, target_intensity)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Tensorboard logging
        writer.add_scalar('Loss', loss.item(), i)

        if (i+1) % 100 == 0 or i == 0:
            print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item()}")
            s_normalized = s.detach().cpu().numpy()
            s_normalized = s_normalized / np.max(s_normalized)
            writer.add_image('Reconstructed Amplitude', torch.tensor(s_normalized, dtype=torch.float32).unsqueeze(0), i)
        
    # save the final reconstructed amplitude 
    if not os.path.exists("output_reconstruction"):
        os.makedirs("output_reconstruction")
    save_Intensity(s, "output_reconstruction/reconstructed_amplitude.png")

    print ("Reconstruction completed.")

if __name__ == "__main__":
    main()





