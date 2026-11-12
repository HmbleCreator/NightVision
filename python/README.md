# NightVision Python Analytics & Evaluation Suite

This package provides quantitative benchmarking, metric calculations (PSNR, SSIM, SNR, CNR), and dataset evaluation tools for the Rust NightVision computational engine.

Managed cleanly with **`uv`**.

### Usage with `uv`

Run scripts directly without manual venv activation:
```bash
# Run benchmark evaluation
uv run scripts/evaluate.py

# Generate comparison plots (Lux vs PSNR/SSIM)
uv run scripts/plot_snr_curves.py
```
