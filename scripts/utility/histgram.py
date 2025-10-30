import numpy as np
import matplotlib.pyplot as plt

# Load the phase data (assumed in radians)
phase = np.load("C:/Users/Kai Kumano/workspace/Phase_retrieval_algorithm/scripts/input/cell/phase_cell.npy")

# Plot with grayscale normalization (-π to π)
plt.figure(figsize=(6, 6))
plt.imshow(phase, cmap='gray', vmin=-np.pi, vmax=np.pi)
plt.axis('off')  # Hide axes
plt.tight_layout(pad=0)
plt.savefig("phase_cell_gray_normalized.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
