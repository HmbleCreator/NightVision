# 🌙 NightVision: Deterministic Computational ISP & Night Vision Engine

A physics-grounded, zero-allocation computational Image Signal Processor (ISP) and digital night vision engine engineered in **Rust** and **C99/C++17**, spanning high-throughput desktop multi-core processors and resource-constrained microcontrollers (`#![no_std]`).

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org)
[![C99/C++17](https://img.shields.io/badge/C%2FC%2B%2B-C99%20%7C%20C%2B%2B17-blue.svg)](include/nightvision.h)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-green.svg)](LICENSE)
[![Benchmarks](https://img.shields.io/badge/docs-Architecture%20%26%20Benchmarks-purple.svg)](docs/ARCHITECTURE_AND_BENCHMARKS.md)

---

## 🎯 Dual-Engine Architecture

`NightVision` is architected around two synchronized backends sharing deterministic noise physics and bidirectional tone models:

1. **`nightvision-core` (Desktop & Edge SoC Engine)**:
   - Multi-threaded planar $Y, C_b, C_r$ zero-allocation ISP processor (`PlanarIspProcessor`).
   - Physics-gated $3\times3$ guided filter ($r=1$) and continuous $1.25\times$ pre-tone detail injection.
   - Sustains **360.5 to 438.7 FPS** ($2.28\text{ ms}$ to $2.77\text{ ms}$) on 600×400 low-light frames.

2. **`nightvision-mcu` (Embedded Fixed-Point Streaming Engine)**:
   - Zero-allocation 3-line circular sliding buffer (`LineBuffer3`), consuming **1.8 KB to 2.3 KB SRAM** across typical sensor resolutions.
   - `#![no_std]` streaming pipeline utilizing single-cycle bit-shifted arithmetic (`>> 2`, `>> 3`) and precomputed 8-bit Flash tables.
   - Sustains **166.0 to 232.9 FPS** ($4.29\text{ ms}$ to $6.02\text{ ms}$) on 600×400 low-light frames with zero runtime heap allocations.

3. **C99/C++17 Firmware API & Tables (`include/nightvision.h`, `include/nightvision_tables.h`)**:
   - Zero-SRAM Flash lookup tables (**3.50 KB Flash / `.rodata`**, 0 bytes SRAM) for STM32H7, ESP32-S3, RP2040, and RISC-V targets.
   - Single-cycle unsharp mask macro: `#define NIGHTVISION_UNSHARP_MASK_1_25X(orig, blur)` with safe intermediate integer casting and saturation clamping.

For complete mathematical derivations, memory footprint calculations, and empirical benchmark matrices across extreme low light and photodiode saturation, see the technical whitepaper:
👉 **[docs/ARCHITECTURE_AND_BENCHMARKS.md](docs/ARCHITECTURE_AND_BENCHMARKS.md)**

---

## 🏛️ System Architecture

```
NightVision/
├── Cargo.toml                         # Cargo workspace configuration
├── CMakeLists.txt                     # C/C++ interface & CMake export
├── include/                           # C/C++ embedded public headers
│   ├── nightvision.h                  # C-API engine definitions & line processors
│   └── nightvision_tables.h           # Static .rodata Flash lookup tables (3.50 KB)
├── docs/
│   └── ARCHITECTURE_AND_BENCHMARKS.md # Unified technical whitepaper & benchmark suite
├── assets/
│   └── benchmarks/                    # 5-panel panoramic comparison strips
├── crates/
│   ├── nightvision-core/              # Zero-allocation planar ISP and physics engine
│   ├── nightvision-mcu/               # #![no_std] fixed-point 3-line streaming engine
│   ├── nightvision-cli/               # CLI batch processor, benchmark harness, and header exporter
│   └── nightvision-gui/               # Real-time native desktop viewer (egui / eframe)
```

---

## 🔬 Core Algorithms

### 1. Physics-Based CMOS Noise Simulation (`sensor.rs`)
Models physical photon starvation across natural lighting environments:
* **Overcast Night** ($0.0001\,\text{lux}$): Extreme photon starvation ($< 0.1\,e^-/\text{pixel}$).
* **Clear Starlight** ($0.001\,\text{lux}$): Heavy Poisson shot noise + Gaussian read noise.
* **Quarter Moon** ($0.01\,\text{lux}$): Deep shadow noise regime.
* **Twilight** ($1.0\,\text{lux}$): Low-light transition.

$$\text{Total Electrons} = \text{Poisson}(\eta \cdot \bar{N}_{\text{photons}}) + \text{Poisson}(I_{\text{dark}} \cdot t_{\text{exp}}) + \mathcal{N}(0, \sigma_{\text{read}}^2)$$

### 2. Motion-Compensated Temporal SNR Accumulation (`temporal.rs`)
Photon flux is accumulated across consecutive frames to boost Signal-to-Noise Ratio:
$$\Delta \text{SNR} = 10 \log_{10}(K) \quad (+12\,\text{dB for } K=16)$$
To prevent motion ghosting, a per-pixel motion metric $D_t(x, y) = |I_t - \bar{I}_{t-1}|$ dynamically shifts pixel blending from high accumulation ($\alpha_{\text{static}} = 0.92$) down to instant response ($\alpha_{\text{motion}} = 0.08$) in moving regions, protected by morphological dilation.

### 3. Edge-Preserving Guided Denoising (`spatial.rs`)
Implements an $O(1)$ box-filtered **Guided Image Filter** (He et al.) that smooths flat sensor noise regions while locking onto high-frequency gradients (silhouettes, text, edges). YCbCr separation applies aggressive chroma despeckling while preserving luma resolution.

### 4. Retinex & CLAHE Local Contrast (`enhancement.rs`)
* **Multi-Scale Retinex with Color Restoration (MSRCR)** decomposes illuminance and reflectance across Gaussian scales ($\sigma \in \{15, 80, 200\}$).
* **CLAHE** partitions luminance into contextual tiles, clipping histograms at strict thresholds to prevent noise explosion in pitch-black tiles, with bilinear interpolation eliminating tile seams.
* **Digital Auto-Gating**: Detects high-luminance flares and applies instant local tone compression ($I / (1 + 0.4I)$) protecting the observer's night-adapted eyes.

### 5. Tactical Phosphor Modes (`modes.rs`)
* **Full-Color DNV**: Full natural RGB color with white balance color constancy.
* **Tactical White Phosphor (P45)**: Cool silver-white phosphor preferred by modern special operations for maximum visual acuity.
* **Tactical Green Phosphor (P43)**: Classic military Gen 3 green peak ($545\,\text{nm}$) with phosphor persistence.
* **High-Contrast Recon**: Edge-boosted amber phosphor for reading text and identifying targets.
* **Thermal Fusion**: Ironbow pseudo-thermal colormap fused with optical outlines.

---

## 🚀 Quickstart Guide

### Prerequisites
* **Rust**: `1.80+` (`cargo` and `rustc`)
* **uv**: Modern fast Python package manager

### 1. Run the End-to-End Demonstration
Simulates raw CMOS capture in starlight and quarter-moon, runs temporal accumulation, renders all tactical modes, and generates side-by-side comparison images:

```bash
cargo run --release -p nightvision-cli -- demo --resolution 640x480 --frames 12
```

The output images will be stored in `output/demo/`:
* `00_reference_ground_truth.png`
* `01_starlight_raw_cmos_single.png` (raw CMOS static noise)
* `02_starlight_enhanced_full_color.png` (enhanced full-color night vision)
* `03_starlight_white_phosphor.png` (tactical P45 white phosphor)
* `03_starlight_green_phosphor.png` (tactical P43 green phosphor)
* `04_starlight_side_by_side_comparison.png` (multi-view composite)

### 2. Run Pipeline Benchmarks
Measures processing latency and sustained frame rates across VGA, HD 720p, and Full HD 1080p:

```bash
cargo run --release -p nightvision-cli -- benchmark --iterations 20
```

### 3. Launch the Interactive GUI Viewer
Launches the native desktop dashboard with real-time sliders for Lux, Exposure, Temporal depth, Denoise radius, and live mode switching:

```bash
cargo run --release -p nightvision-gui
```

### 4. Python Analytics & Metrics (Managed via `uv`)
Calculate quantitative PSNR and SSIM between ground truth and enhanced frames:

```bash
cd python
uv run scripts/evaluate.py ../output/demo/00_reference_ground_truth.png ../output/demo/02_quarter_moon_enhanced_full_color.png
```

Generate the publication-ready SNR vs. Lux enhancement curve:

```bash
uv run scripts/benchmark_curves.py
```

---

## 📊 Quantitative Verification

Under Quarter-Moon illuminance ($0.01\,\text{lux}$):
* **Raw CMOS Sensor Output**: $\text{PSNR} = 10.68\,\text{dB} \mid \text{SSIM} = 0.5767$
* **NightVision Enhanced Output**: $\text{PSNR} = 12.41\,\text{dB} \mid \text{SSIM} = 0.7499$
* **Signal-to-Noise Ratio Gain**: **$+12\,\text{dB}$ to $+16\,\text{dB}$** in static and low-light regions.

