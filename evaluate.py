"""
evaluate.py — CLI entry point for PSNR / SSIM evaluation.

Usage:
    python evaluate.py <gt_path> <output1_path> <output2_path>

Example:
    python evaluate.py \\
        output/target_gt/phase_cells.png \\
        output/reconstruction/final_s_plane_phs.png \\
        output/reconstruction_no_ref/final_s_plane_phs.png
"""

import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def evaluate(gt_path: str, out1_path: str, out2_path: str) -> tuple:
    """Compute PSNR and SSIM for two reconstructions against a ground truth.

    Args:
        gt_path: Path to the ground-truth grayscale image.
        out1_path: Path to the first reconstruction image.
        out2_path: Path to the second reconstruction image.

    Returns:
        Tuple (psnr1, ssim1, psnr2, ssim2).
    """
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    out1 = cv2.imread(out1_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
    out2 = cv2.imread(out2_path, cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0

    h, w = gt.shape
    out1 = cv2.resize(out1, (w, h))
    out2 = cv2.resize(out2, (w, h))

    psnr1 = peak_signal_noise_ratio(gt, out1, data_range=1.0)
    psnr2 = peak_signal_noise_ratio(gt, out2, data_range=1.0)
    ssim1 = structural_similarity(gt, out1, data_range=1.0)
    ssim2 = structural_similarity(gt, out2, data_range=1.0)

    print("==========================================")
    print(f"GT vs Output1: PSNR = {psnr1:.2f} dB,  SSIM = {ssim1:.4f}")
    print(f"GT vs Output2: PSNR = {psnr2:.2f} dB,  SSIM = {ssim2:.4f}")
    print("==========================================")

    return psnr1, ssim1, psnr2, ssim2


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM of two reconstructions.")
    parser.add_argument("gt", help="Ground-truth image path.")
    parser.add_argument("out1", help="First reconstruction image path.")
    parser.add_argument("out2", help="Second reconstruction image path.")
    args = parser.parse_args()

    evaluate(args.gt, args.out1, args.out2)


if __name__ == "__main__":
    main()
