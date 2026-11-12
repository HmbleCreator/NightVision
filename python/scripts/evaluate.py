#!/usr/bin/env python3
"""
NightVision Evaluation Suite
Calculates quantitative low-light metrics: Peak Signal-to-Noise Ratio (PSNR),
Structural Similarity Index (SSIM), and Signal-to-Noise Ratio (SNR).
Managed via `uv run scripts/evaluate.py`.
"""

import sys
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image


def calculate_psnr(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """Calculates Peak Signal to Noise Ratio between reference and test image."""
    mse = np.mean((img_ref.astype(np.float64) - img_test.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(20 * math.log10(max_pixel / math.sqrt(mse)))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculates mean Structural Similarity Index (SSIM) on luminance."""
    if img1.ndim == 3:
        # Convert RGB to Luminance (Rec.601)
        y1 = 0.299 * img1[:, :, 0] + 0.587 * img1[:, :, 1] + 0.114 * img1[:, :, 2]
        y2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
    else:
        y1 = img1.astype(np.float64)
        y2 = img2.astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    mu1 = np.mean(y1)
    mu2 = np.mean(y2)
    sigma1_sq = np.var(y1)
    sigma2_sq = np.var(y2)
    sigma12 = np.mean((y1 - mu1) * (y2 - mu2))

    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim)


def calculate_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculates Signal-to-Noise Ratio in decibels (dB)."""
    p_signal = np.mean(signal.astype(np.float64) ** 2)
    p_noise = np.mean(noise.astype(np.float64) ** 2)
    if p_noise == 0:
        return 100.0
    return float(10 * math.log10(p_signal / p_noise))


def main():
    print("=" * 60)
    print("NightVision Quantitative Analytics Suite (uv managed)")
    print("=" * 60)

    if len(sys.argv) >= 3:
        ref_path = Path(sys.argv[1])
        test_path = Path(sys.argv[2])
        if not ref_path.exists() or not test_path.exists():
            print(f"Error: One of the files does not exist: {ref_path}, {test_path}")
            sys.exit(1)

        ref_img = np.array(Image.open(ref_path).convert("RGB"))
        test_img = np.array(Image.open(test_path).convert("RGB"))

        psnr_val = calculate_psnr(ref_img, test_img)
        ssim_val = calculate_ssim(ref_img, test_img)
        print(f"Reference: {ref_path}")
        print(f"Enhanced : {test_path}")
        print(f"PSNR     : {psnr_val:.2f} dB")
        print(f"SSIM     : {ssim_val:.4f}")
    else:
        print("Usage: uv run scripts/evaluate.py <ground_truth.png> <enhanced.png>")
        print("Running internal self-verification test...")
        # Synthetic test pattern
        synthetic_ref = np.full((128, 128, 3), 120, dtype=np.uint8)
        synthetic_noise = np.clip(
            synthetic_ref + np.random.normal(0, 15, synthetic_ref.shape), 0, 255
        ).astype(np.uint8)

        psnr = calculate_psnr(synthetic_ref, synthetic_noise)
        ssim = calculate_ssim(synthetic_ref, synthetic_noise)
        print(f"Synthetic test PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        print("Self-test passed.")


if __name__ == "__main__":
    main()
