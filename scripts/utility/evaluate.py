import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity # type: ignore

# ---------- Function to compute metrics ----------
def evaluate_metrics(gt_path, out1_path, out2_path):
    # Load images as grayscale float64 [0,1]
    gt  = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    out1 = cv2.imread(out1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    out2 = cv2.imread(out2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    # Ensure same size
    h, w = gt.shape
    out1 = cv2.resize(out1, (w, h))
    out2 = cv2.resize(out2, (w, h))

    # Compute PSNR
    psnr1 = peak_signal_noise_ratio(gt, out1, data_range=1.0)
    psnr2 = peak_signal_noise_ratio(gt, out2, data_range=1.0)

    # Compute SSIM
    ssim1 = structural_similarity(gt, out1, data_range=1.0)
    ssim2 = structural_similarity(gt, out2, data_range=1.0)

    print("==========================================")
    print(f"GT vs Output1: PSNR = {psnr1:.2f} dB, SSIM = {ssim1:.4f}")
    print(f"GT vs Output2: PSNR = {psnr2:.2f} dB, SSIM = {ssim2:.4f}")
    print("==========================================")

    return psnr1, ssim1, psnr2, ssim2

# ---------- Example usage ----------
if __name__ == "__main__":
    # gt_path  = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\input\Man.bmp"
    # out1_path = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\output\output_reconstruction\reference_constraint\final_s_plane_int.png"
    # out2_path = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\output\output_reconstruction\without_reference_constraint\final_s_plane_int_without.png"

    gt_path  = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\output\output_gabor\target_gt\cell\phase_cells.png"
    ref_path = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\output\output_reconstruction\cell\final_s_plane_phs.png"
    without_path = r"C:\Users\Kai Kumano\workspace\Phase_retrieval_algorithm\output\output_reconstruction\cell\final_s_plane_phs_without.png"


    evaluate_metrics(gt_path, ref_path, without_path)
