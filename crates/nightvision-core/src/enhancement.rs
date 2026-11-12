//! Illumination enhancement, dynamic range expansion, and flare protection.
//!
//! Includes:
//! - Multi-Scale Retinex with Color Restoration (MSRCR): separates illumination and reflectance.
//! - CLAHE: Contrast Limited Adaptive Histogram Equalization with bilinear tile interpolation.
//! - Digital Auto-Gating: Instantaneous software flare suppression preventing user blinding
//!   from sudden bright lights (muzzle flashes, headlights) without damaging the sensor.

use image::{ImageBuffer, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::spatial::box_filter_2d;

/// Configuration for illumination and tone enhancement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancementConfig {
    /// Multi-Scale Retinex blend weight (0.0 = disabled, 1.0 = full).
    pub retinex_weight: f32,
    /// CLAHE clip limit (1.0 to 10.0). Higher values boost local contrast more aggressively.
    pub clahe_clip_limit: f32,
    /// Number of grid tiles for CLAHE (e.g., 8 means an 8x8 tile grid).
    pub clahe_grid_size: usize,
    /// Adaptive gamma correction exponent (0.3 to 1.0). Lower values lift deep shadows.
    pub gamma: f32,
    /// Exposure multiplier boost factor (1.0 to 20.0).
    pub exposure_gain: f32,
    /// Auto-gating flare threshold (0.6 to 1.0). Compresses sudden blinding highlights.
    pub auto_gating_threshold: f32,
}

impl Default for EnhancementConfig {
    fn default() -> Self {
        Self {
            retinex_weight: 0.65,
            clahe_clip_limit: 3.0,
            clahe_grid_size: 8,
            gamma: 0.6,
            exposure_gain: 2.5,
            auto_gating_threshold: 0.88,
        }
    }
}

/// Gaussian approximation via 3 successive box blurs (Central Limit Theorem).
fn fast_gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.5 {
        return src.to_vec();
    }
    let r = ((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0;
    let r_int = (r.round() as usize).max(1);

    let pass1 = box_filter_2d(src, w, h, r_int);
    let pass2 = box_filter_2d(&pass1, w, h, r_int);
    box_filter_2d(&pass2, w, h, r_int)
}

/// Multi-Scale Retinex with Color Restoration (MSRCR) on RGB channels (Rayon accelerated).
pub fn apply_retinex(img: &RgbImage, weight: f32) -> RgbImage {
    if weight <= 0.001 {
        return img.clone();
    }

    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);
    let n = w * h;

    // Extract normalized float channels
    let mut r_chan = vec![0.0f32; n];
    let mut g_chan = vec![0.0f32; n];
    let mut b_chan = vec![0.0f32; n];

    for (idx, p) in img.pixels().enumerate() {
        r_chan[idx] = (p[0] as f32) / 255.0;
        g_chan[idx] = (p[1] as f32) / 255.0;
        b_chan[idx] = (p[2] as f32) / 255.0;
    }

    let scales = [15.0f32, 80.0f32, 200.0f32];
    let scale_weight = 1.0 / (scales.len() as f32);

    let process_channel = |chan: &[f32]| -> Vec<f32> {
        // Pre-compute log_orig once
        let log_orig: Vec<f32> = chan
            .par_iter()
            .map(|&v| (v * 255.0 + 1.0).ln())
            .collect();

        let mut retinex_acc = vec![0.0f32; n];

        for &sigma in &scales {
            let blur = fast_gaussian_blur(chan, w, h, sigma);
            retinex_acc
                .par_iter_mut()
                .zip(log_orig.par_iter())
                .zip(blur.par_iter())
                .for_each(|((acc, &lo), &b)| {
                    let log_blur = (b * 255.0 + 1.0).ln();
                    *acc += scale_weight * (lo - log_blur);
                });
        }
        retinex_acc
    };

    let ret_r = process_channel(&r_chan);
    let ret_g = process_channel(&g_chan);
    let ret_b = process_channel(&b_chan);

    let mut out_img = ImageBuffer::new(width, height);
    let out_slice = out_img.as_flat_samples_mut();
    let out_bytes = out_slice.samples;

    out_bytes
        .par_chunks_exact_mut(3)
        .enumerate()
        .for_each(|(i, pixel)| {
            let sum_rgb = r_chan[i] + g_chan[i] + b_chan[i] + 0.001;
            let crf_r = ((125.0 * r_chan[i] / sum_rgb + 1.0).ln()).max(0.1);
            let crf_g = ((125.0 * g_chan[i] / sum_rgb + 1.0).ln()).max(0.1);
            let crf_b = ((125.0 * b_chan[i] / sum_rgb + 1.0).ln()).max(0.1);

            let enh_r = (ret_r[i] * crf_r * 0.35 + 0.5).clamp(0.0, 1.0);
            let enh_g = (ret_g[i] * crf_g * 0.35 + 0.5).clamp(0.0, 1.0);
            let enh_b = (ret_b[i] * crf_b * 0.35 + 0.5).clamp(0.0, 1.0);

            let blend_r = r_chan[i] * (1.0 - weight) + enh_r * weight;
            let blend_g = g_chan[i] * (1.0 - weight) + enh_g * weight;
            let blend_b = b_chan[i] * (1.0 - weight) + enh_b * weight;

            pixel[0] = (blend_r * 255.0).round().clamp(0.0, 255.0) as u8;
            pixel[1] = (blend_g * 255.0).round().clamp(0.0, 255.0) as u8;
            pixel[2] = (blend_b * 255.0).round().clamp(0.0, 255.0) as u8;
        });

    out_img
}

/// Applies Contrast-Limited Adaptive Histogram Equalization (CLAHE) on luminance (Rayon accelerated).
pub fn apply_clahe(img: &RgbImage, grid_size: usize, clip_limit: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let (w, h) = (width as usize, height as usize);
    let n = w * h;

    if n == 0 || grid_size < 2 {
        return img.clone();
    }

    // Convert to Luminance (Y) and Chrominance ratios
    let mut y_channel = vec![0u8; n];
    let mut r_ratio = vec![0.0f32; n];
    let mut g_ratio = vec![0.0f32; n];
    let mut b_ratio = vec![0.0f32; n];

    for (idx, p) in img.pixels().enumerate() {
        let r = p[0] as f32;
        let g = p[1] as f32;
        let b = p[2] as f32;
        let y = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
        y_channel[idx] = y;

        let y_denom = (y as f32).max(1.0);
        r_ratio[idx] = r / y_denom;
        g_ratio[idx] = g / y_denom;
        b_ratio[idx] = b / y_denom;
    }

    let tile_w = (w + grid_size - 1) / grid_size;
    let tile_h = (h + grid_size - 1) / grid_size;

    // Calculate CDF mappings for each tile grid cell
    let mut cdfs = vec![vec![0.0f32; 256]; grid_size * grid_size];

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x_start = gx * tile_w;
            let x_end = (x_start + tile_w).min(w);
            let y_start = gy * tile_h;
            let y_end = (y_start + tile_h).min(h);

            let tile_pixels = ((x_end - x_start) * (y_end - y_start)) as f32;
            if tile_pixels <= 0.0 {
                continue;
            }

            let mut hist = [0u32; 256];
            for py in y_start..y_end {
                for px in x_start..x_end {
                    let val = y_channel[py * w + px] as usize;
                    hist[val] += 1;
                }
            }

            let clip_threshold = (clip_limit * tile_pixels / 256.0).max(1.0) as u32;
            let mut excess = 0u32;
            for val in &mut hist {
                if *val > clip_threshold {
                    excess += *val - clip_threshold;
                    *val = clip_threshold;
                }
            }

            let redist = excess / 256;
            let remainder = excess % 256;
            for i in 0..256 {
                hist[i] += redist + if (i as u32) < remainder { 1 } else { 0 };
            }

            let mut acc = 0u32;
            let cdf_idx = gy * grid_size + gx;
            for i in 0..256 {
                acc += hist[i];
                cdfs[cdf_idx][i] = (acc as f32) / tile_pixels;
            }
        }
    }

    // Parallel row-by-row bilinear interpolation
    let mut out_img = ImageBuffer::new(width, height);
    let out_slice = out_img.as_flat_samples_mut();
    let out_bytes = out_slice.samples;

    out_bytes
        .par_chunks_exact_mut(w * 3)
        .enumerate()
        .for_each(|(py, row_bytes)| {
            let gy_f = (py as f32 / tile_h as f32) - 0.5;
            let gy1 = (gy_f.floor() as isize).clamp(0, grid_size as isize - 1) as usize;
            let gy2 = (gy1 + 1).min(grid_size - 1);
            let alpha_y = (gy_f - gy_f.floor()).clamp(0.0, 1.0);

            for px in 0..w {
                let gx_f = (px as f32 / tile_w as f32) - 0.5;
                let gx1 = (gx_f.floor() as isize).clamp(0, grid_size as isize - 1) as usize;
                let gx2 = (gx1 + 1).min(grid_size - 1);
                let alpha_x = (gx_f - gx_f.floor()).clamp(0.0, 1.0);

                let idx = py * w + px;
                let y_val = y_channel[idx] as usize;

                let cdf11 = cdfs[gy1 * grid_size + gx1][y_val];
                let cdf21 = cdfs[gy1 * grid_size + gx2][y_val];
                let cdf12 = cdfs[gy2 * grid_size + gx1][y_val];
                let cdf22 = cdfs[gy2 * grid_size + gx2][y_val];

                let top = cdf11 * (1.0 - alpha_x) + cdf21 * alpha_x;
                let bottom = cdf12 * (1.0 - alpha_x) + cdf22 * alpha_x;
                let interp_cdf = top * (1.0 - alpha_y) + bottom * alpha_y;

                let new_y = interp_cdf * 255.0;

                let new_r = (new_y * r_ratio[idx]).round().clamp(0.0, 255.0) as u8;
                let new_g = (new_y * g_ratio[idx]).round().clamp(0.0, 255.0) as u8;
                let new_b = (new_y * b_ratio[idx]).round().clamp(0.0, 255.0) as u8;

                row_bytes[px * 3] = new_r;
                row_bytes[px * 3 + 1] = new_g;
                row_bytes[px * 3 + 2] = new_b;
            }
        });

    out_img
}

/// Applies dynamic exposure boost, adaptive gamma lift, and digital auto-gating flare suppression.
pub fn apply_exposure_and_autogating(
    img: &RgbImage,
    exposure_gain: f32,
    gamma: f32,
    auto_gating_threshold: f32,
) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out_img = ImageBuffer::new(width, height);

    // Compute scene peak to detect sudden flares (muzzle flashes / headlights)
    let max_luma = img
        .pixels()
        .map(|p| (0.299 * (p[0] as f32) + 0.587 * (p[1] as f32) + 0.114 * (p[2] as f32)) / 255.0)
        .fold(0.0f32, f32::max);

    let flare_attenuation = if max_luma > auto_gating_threshold {
        1.0 / (1.0 + (max_luma - auto_gating_threshold) * 8.0)
    } else {
        1.0
    };

    let effective_gain = exposure_gain * flare_attenuation;
    let inv_gamma = 1.0 / gamma.clamp(0.1, 2.0);

    let out_slice = out_img.as_flat_samples_mut();
    let out_bytes = out_slice.samples;

    out_bytes
        .par_chunks_exact_mut(3)
        .zip(img.as_flat_samples().samples.par_chunks_exact(3))
        .for_each(|(out_p, in_p)| {
            for c in 0..3 {
                let normalized = (in_p[c] as f32) / 255.0;
                let boosted = normalized * effective_gain;
                let tone_mapped = boosted.powf(inv_gamma);
                let compressed = tone_mapped / (1.0 + tone_mapped * 0.4);
                out_p[c] = (compressed * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        });

    out_img
}
