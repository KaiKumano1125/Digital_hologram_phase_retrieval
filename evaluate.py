"""
evaluate.py -- CLI entry point for PSNR / SSIM evaluation.

Usage:
    python evaluate.py --config config/config.yaml
    python evaluate.py --config config/config.yaml --component phase
    python evaluate.py <gt_path> <output1_path> [<output2_path> ...]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from config.config_loader import load_config


def _read_grayscale(path: str) -> np.ndarray:
    """Load a grayscale image as float64 in [0, 1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image file '{path}' not found.")
    return img.astype(np.float64) / 255.0


def evaluate(gt_path: str, output_paths: list[str]) -> list[tuple[str, float, float]]:
    """Compute PSNR and SSIM for one or more outputs against a ground truth."""
    gt = _read_grayscale(gt_path)
    h, w = gt.shape

    results: list[tuple[str, float, float]] = []
    for output_path in output_paths:
        out = _read_grayscale(output_path)
        out = cv2.resize(out, (w, h))

        psnr = peak_signal_noise_ratio(gt, out, data_range=1.0)
        ssim = structural_similarity(gt, out, data_range=1.0)
        results.append((output_path, psnr, ssim))

    print("==========================================")
    print(f"Ground truth: {gt_path}")
    for output_path, psnr, ssim in results:
        print(f"{output_path}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
    print("==========================================")

    return results


def evaluate_reconstruction(
    config_path: str,
    component: str = "both",
    output_dir: str | None = None,
) -> None:
    """Evaluate reconstruction outputs against the configured input images."""
    cfg = load_config(config_path)
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else Path(cfg.outdir)

    eval_jobs = []
    if component in {"amplitude", "both"}:
        eval_jobs.append(
            (
                "Amplitude",
                cfg.amp_path,
                str(resolved_output_dir / "final_s_plane_amp.png"),
            )
        )
    if component in {"phase", "both"}:
        eval_jobs.append(
            (
                "Phase",
                cfg.phase_path,
                str(resolved_output_dir / "final_s_plane_phs.png"),
            )
        )

    for label, gt_path, recon_path in eval_jobs:
        print(f"[{label}]")
        evaluate(gt_path, [recon_path])


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PSNR/SSIM for reconstruction outputs.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Legacy mode: ground-truth path followed by one or more output paths.",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file. Uses cfg.amp_path/cfg.phase_path and cfg.outdir automatically.",
    )
    parser.add_argument(
        "--component",
        choices=("amplitude", "phase", "both"),
        default="both",
        help="Which reconstruction result to evaluate in --config mode.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the reconstruction output directory in --config mode.",
    )
    args = parser.parse_args()

    if args.config:
        evaluate_reconstruction(args.config, component=args.component, output_dir=args.output_dir)
        return

    if len(args.paths) < 2:
        parser.error("Provide --config, or pass <gt_path> followed by one or more output paths.")

    gt_path, *output_paths = args.paths
    evaluate(gt_path, output_paths)


if __name__ == "__main__":
    main()
