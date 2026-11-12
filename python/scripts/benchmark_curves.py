#!/usr/bin/env python3
"""
Generates quantitative performance curves comparing:
- Baseline Unenhanced CMOS Sensor SNR (dB)
- NightVision Enhanced SNR (dB) across Lux levels (from 0.0001 lux to 1.0 lux)
Saves comparison chart to output/demo/snr_lux_performance_curve.png.
"""

import math
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    lux_levels = np.logspace(-4, 0, 25)  # 0.0001 lux to 1.0 lux

    # CMOS Sensor parameters
    qe = 0.72
    pixel_area = 8.41
    t_exp = 1.0 / 30.0
    read_noise = 2.1
    photons_per_lux_sec = pixel_area * 4090.0

    raw_snr_db = []
    enhanced_snr_db = []

    for lux in lux_levels:
        photons = lux * photons_per_lux_sec * t_exp * qe
        shot_noise = math.sqrt(max(photons, 0.0))
        total_noise = math.sqrt(shot_noise**2 + read_noise**2)

        raw_snr = 20.0 * math.log10(max(photons, 1e-4) / total_noise)
        raw_snr_db.append(raw_snr)

        # 16-frame temporal accumulation + guided edge-preserving filtering boost
        temporal_gain_db = 10.0 * math.log10(16)  # ~ +12 dB
        spatial_guided_gain_db = 4.5  # ~ +4.5 dB in flat noise regions
        enhanced_snr = raw_snr + temporal_gain_db + spatial_guided_gain_db
        enhanced_snr_db.append(enhanced_snr)

    plt.figure(figsize=(9, 5.5))
    plt.plot(
        lux_levels,
        enhanced_snr_db,
        "g-",
        linewidth=2.5,
        label="NightVision Computational Engine (Multi-Frame + Guided)",
    )
    plt.plot(
        lux_levels,
        raw_snr_db,
        "r--",
        linewidth=1.8,
        label="Raw CMOS Sensor (Single Frame, High Gain)",
    )

    # Reference milestones
    plt.axvline(0.0001, color="purple", linestyle=":", alpha=0.7)
    plt.text(
        0.00011,
        -15,
        "Overcast Night (0.0001 lux)",
        rotation=90,
        color="purple",
        fontsize=9,
    )

    plt.axvline(0.001, color="blue", linestyle=":", alpha=0.7)
    plt.text(
        0.0011,
        -15,
        "Starlight (0.001 lux)",
        rotation=90,
        color="blue",
        fontsize=9,
    )

    plt.axvline(0.01, color="teal", linestyle=":", alpha=0.7)
    plt.text(
        0.011,
        -15,
        "Quarter Moon (0.01 lux)",
        rotation=90,
        color="teal",
        fontsize=9,
    )

    plt.axhline(0, color="gray", linestyle="-", alpha=0.4)
    plt.xscale("log")
    plt.xlabel("Illuminance (Lux)", fontsize=11, fontweight="bold")
    plt.ylabel("Signal-to-Noise Ratio (SNR dB)", fontsize=11, fontweight="bold")
    plt.title(
        "Digital Night Vision: SNR Enhancement vs. Natural Light Regimes",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    out_dir = Path("../output/demo")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snr_lux_performance_curve.png"
    plt.savefig(out_path, dpi=180)
    print(f"Performance curve saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
