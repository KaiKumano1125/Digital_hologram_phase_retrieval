#generate RBC and WBC resembling model
import numpy as np
# import matplotlib.pyplot as plt
import os

def generate_rbc(shape, radius, thickness, amp_val=0.8, phase_val=np.pi/2):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    amplitude = np.ones(shape)
    phase = np.zeros(shape)

    ring_mask = (r >= radius - thickness) & (r <= radius + thickness)
    amplitude[ring_mask] = amp_val
    phase[ring_mask] = phase_val
    return amplitude, phase

def generate_wbc(shape, radius, amp_val=0.5, phase_val=np.pi/4):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    amplitude = np.ones(shape)
    phase = np.zeros(shape)

    blob_mask = r <= radius
    amplitude[blob_mask] = amp_val
    phase[blob_mask] = phase_val
    return amplitude, phase

shape = (1024, 1024)

amp_rbc, phase_rbc = generate_rbc(shape, radius=100, thickness=20)
amp_wbc, phase_wbc = generate_wbc(shape, radius=60)

# Combine them (optional)
amp_combined = np.maximum(amp_rbc, amp_wbc)
phase_combined = np.maximum(phase_rbc, phase_wbc)

# Normalize and save
import cv2

def normalize(x):
    x = x - x.min()
    return (x / (x.max() + 1e-9) * 255).astype(np.uint8)

os.makedirs("../input/cell", exist_ok=True)
cv2.imwrite("../input/cell/amp_cells.png", normalize(amp_combined))
cv2.imwrite("../input/cell/phase_cells.png", normalize(phase_combined))

print("Cell amplitude and phase images saved.")