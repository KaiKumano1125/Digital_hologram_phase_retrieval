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
from skimage.exposure import match_histograms
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

    return results


def evaluate_with_offset(gt_path: str, output_paths: list[str], offset: float = 0.0) -> list[tuple[str, float, float]]:
    """Like evaluate() but subtracts a constant offset from outputs before comparison.

    Use offset=0.5 when the reconstruction saves phase as (angle+pi)/(2pi)
    but the GT saves phase as angle/(2pi) — corrects the pi/2pi baseline shift.
    """
    gt = _read_grayscale(gt_path)
    h, w = gt.shape

    results: list[tuple[str, float, float]] = []
    for output_path in output_paths:
        out = _read_grayscale(output_path)
        out = cv2.resize(out, (w, h))
        out = np.clip(out - offset, 0.0, 1.0)

        psnr = peak_signal_noise_ratio(gt, out, data_range=1.0)
        ssim = structural_similarity(gt, out, data_range=1.0)
        results.append((output_path, psnr, ssim))

    print("==========================================")
    print(f"Ground truth: {gt_path}")
    for output_path, psnr, ssim in results:
        print(f"{output_path}: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
    print("==========================================")

    return results


def _evaluate_phase_npy(npy_path: str, gt_path: str) -> None:
    """Evaluate phase using histogram matching against GT.

    Loads the raw complex field (.npy), extracts phase, histogram-matches it
    to the GT distribution, then computes PSNR/SSIM. This removes global
    offset/scale differences so the score reflects spatial structure only.
    """
    complex_field = np.load(npy_path)
    phase_rad = np.angle(complex_field)                        # [-pi, pi]
    recon = np.mod(phase_rad, 2 * np.pi) / (2 * np.pi)        # [0, 1]

    gt = _read_grayscale(gt_path)
    h, w = gt.shape
    if recon.shape != (h, w):
        recon = cv2.resize(recon.astype(np.float32), (w, h)).astype(np.float64)

    matched = match_histograms(recon, gt)
    psnr = peak_signal_noise_ratio(gt, matched, data_range=1.0)
    ssim = structural_similarity(gt, matched, data_range=1.0)
    print("==========================================")
    print(f"Ground truth: {gt_path}")
    print(f"{npy_path} [histogram-matched]: PSNR = {psnr:.2f} dB, SSIM = {ssim:.4f}")
    print("==========================================")


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
        if label == "Phase":
            # Use raw .npy to avoid peak-normalisation artefact in the PNG
            npy_path = str(resolved_output_dir / "final_s_plane_complex.npy")
            _evaluate_phase_npy(npy_path, gt_path)
        else:
            evaluate_with_offset(gt_path, [recon_path], offset=0.0)


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
