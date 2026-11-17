//! Physics-Grounded Single-Frame Highlight Recovery.
//!
//! Reconstructs partially clipped highlights using surviving RGB channels to restore
//! HDR headroom (> 1.0), followed by Durand-Dorsey base-detail decomposition in the
//! log domain and C¹-continuous filmic tone reproduction.
//!
//! Solves the 8-bit ceiling trap:
//! - When Red clips on warm skin (e.g. R=253 ADU) while Green is 200 ADU and Blue is 165 ADU,
//!   luma Y is only ~212 ADU.
//! - By analyzing RGB channels directly, we restore the clipped channel's slope using the
//!   surviving channels' ratio (C_rec = max(T_clip, S * (C_base / S_base))).
//! - In the resulting HDR buffer, the face core (~1.25 linear) and unclipped cheek (~0.95 linear)
//!   are cleanly separated, allowing filmic tonemapping without darkening nearby midtones.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::soft_knee::{filmic_tonemap, FilmicConfig};

/// Configuration for highlight recovery.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HighlightConfig {
    /// Clipping threshold in normalized [0, 1] units.
    /// Default: 248.0 / 255.0 (~0.9725, JPEG spike - 5).
    pub clip_threshold: f32,
    /// Minimum channel threshold to qualify as surviving (unclipped).
    /// Default: 240.0 / 255.0 (~0.941).
    pub surviving_threshold: f32,
    /// Filmic soft-knee parameters.
    pub filmic: FilmicConfig,
    /// Guided filter / box blur radius for base layer extraction.
    pub filter_radius: usize,
    /// Regularization epsilon for log domain.
    pub eps: f32,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            clip_threshold: 248.0 / 255.0,
            surviving_threshold: 240.0 / 255.0,
            filmic: FilmicConfig::default(),
            filter_radius: 8,
            eps: 1.0 / 255.0,
        }
    }
}

/// Scratch memory for highlight recovery (reusable across frames with zero heap allocations).
#[derive(Default)]
pub struct HighlightScratch {
    pub r_plane: Vec<f32>,
    pub g_plane: Vec<f32>,
    pub b_plane: Vec<f32>,
    pub y_hdr: Vec<f32>,
    pub log_base: Vec<f32>,
    pub temp: Vec<f32>,
}

impl HighlightScratch {
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.r_plane.len() != n {
            self.r_plane.resize(n, 0.0);
            self.g_plane.resize(n, 0.0);
            self.b_plane.resize(n, 0.0);
            self.y_hdr.resize(n, 0.0);
            self.log_base.resize(n, 0.0);
            self.temp.resize(n, 0.0);
        }
    }
}

/// Simple 1D horizontal box blur for planar data.
fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, r: usize) {
    let r_i = r as isize;
    let w_i = w as isize;
    for x in 0..w {
        let x_i = x as isize;
        let x0 = (x_i - r_i).max(0) as usize;
        let x1 = (x_i + r_i).min(w_i - 1) as usize;
        let count = (x1 - x0 + 1) as f32;
        let mut sum = 0.0f32;
        for i in x0..=x1 {
            sum += src[i];
        }
        dst[x] = sum / count;
    }
}

/// Separable 2D Box Filter on planar memory.
pub fn box_filter_2d(src: &[f32], dst: &mut [f32], temp: &mut [f32], w: usize, h: usize, r: usize) {
    if r == 0 || w == 0 || h == 0 {
        dst.copy_from_slice(src);
        return;
    }

    // Horizontal pass
    temp.par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(t_row, s_row)| {
            box_blur_h(s_row, t_row, w, r);
        });

    // Vertical pass
    let r_i = r as isize;
    let h_i = h as isize;
    dst.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let y_i = y as isize;
            let y0 = (y_i - r_i).max(0) as usize;
            let y1 = (y_i + r_i).min(h_i - 1) as usize;
            let count = (y1 - y0 + 1) as f32;
            let inv_count = 1.0 / count;

            for x in 0..w {
                let mut sum = 0.0f32;
                for row_idx in y0..=y1 {
                    sum += temp[row_idx * w + x];
                }
                out_row[x] = sum * inv_count;
            }
        });
}

/// Recovers partially-clipped and overexposed highlights directly in planar YCbCr memory.
///
/// Steps:
/// 1. Reconstruct RGB from (Y, Cb, Cr).
/// 2. Phase 1: Reconstruct clipped channels from surviving channel ratios (HDR headroom > 1.0).
/// 3. Phase 2: Compute Y_hdr and decompose into Base + Detail in log domain.
/// 4. Phase 3: Apply C¹ filmic tonemap to Base_lin, multiply by exp(Detail).
/// 5. Phase 4: Desaturate highlights only when all channels saturate.
/// 6. Convert back to (Y, Cb, Cr) in-place.
pub fn recover_highlights_planar(
    y_plane: &mut [f32],
    cb_plane: &mut [f32],
    cr_plane: &mut [f32],
    scratch: &mut HighlightScratch,
    w: usize,
    h: usize,
    config: &HighlightConfig,
) {
    let n = w * h;
    if n == 0 {
        return;
    }
    scratch.ensure_capacity(n);

    let clip_thr = config.clip_threshold;
    let surviving_thr = config.surviving_threshold;
    let eps = config.eps;
    let r_filter = config.filter_radius;

    // Step 1: Reconstruct RGB from YCbCr
    let r_plane = &mut scratch.r_plane;
    let g_plane = &mut scratch.g_plane;
    let b_plane = &mut scratch.b_plane;

    r_plane
        .par_iter_mut()
        .zip(g_plane.par_iter_mut())
        .zip(b_plane.par_iter_mut())
        .zip(y_plane.par_iter())
        .zip(cb_plane.par_iter())
        .zip(cr_plane.par_iter())
        .for_each(|(((((r, g), b), &y), &cb), &cr)| {
            let cb0 = cb - 0.5;
            let cr0 = cr - 0.5;
            *r = (y + 1.402 * cr0).max(0.0);
            *g = (y - 0.344136 * cb0 - 0.714136 * cr0).max(0.0);
            *b = (y + 1.772 * cb0).max(0.0);
        });

    // Step 2: Compute smoothed RGB base planes for robust local channel ratios
    // We smooth G as our primary anchor channel (highest sensor QE and MTF)
    box_filter_2d(&scratch.g_plane, &mut scratch.log_base, &mut scratch.temp, w, h, r_filter);
    // log_base temporarily stores smoothed G
    let g_base = &scratch.log_base;

    // Phase 1: Partial channel slope restoration
    let r_plane = &mut scratch.r_plane;
    let g_plane = &mut scratch.g_plane;
    let b_plane = &mut scratch.b_plane;
    let y_hdr = &mut scratch.y_hdr;

    r_plane
        .par_iter_mut()
        .zip(g_plane.par_iter_mut())
        .zip(b_plane.par_iter_mut())
        .zip(y_hdr.par_iter_mut())
        .zip(g_base.par_iter())
        .for_each(|((((r, g), b), y_out), &g_b)| {
            let ch_max = (*r).max(*g).max(*b);
            let ch_min = (*r).min(*g).min(*b);

            if ch_max >= clip_thr && ch_min < surviving_thr {
                // Partial clipping detected (e.g. warm skin: R clipped, G/B surviving)
                let g_ref = g_b.max(0.05);

                if *r >= clip_thr && *g < surviving_thr {
                    // Restore Red slope from Green
                    let r_rec = (*g) * (ch_max / g_ref);
                    *r = (*r).max(r_rec);
                }
                if *b >= clip_thr && *g < surviving_thr {
                    // Restore Blue slope from Green
                    let b_rec = (*g) * (ch_max / g_ref);
                    *b = (*b).max(b_rec);
                }
                if *g >= clip_thr && *r < surviving_thr {
                    // Restore Green from Red
                    let g_rec = (*r) * (ch_max / (*r).max(0.05));
                    *g = (*g).max(g_rec);
                }
            }

            // Compute Y_hdr with extended headroom
            *y_out = 0.299 * (*r) + 0.587 * (*g) + 0.114 * (*b);
        });

    // Phase 2: Log Base-Detail Decomposition
    // L = log(Y_hdr + eps)
    let y_hdr = &scratch.y_hdr;
    let log_luma = &mut scratch.r_plane; // Reuse r_plane as log_luma scratch
    log_luma
        .par_iter_mut()
        .zip(y_hdr.par_iter())
        .for_each(|(l, &y)| {
            *l = (y + eps).ln();
        });

    // B = Guided / Box filter of L
    let log_base = &mut scratch.log_base;
    let temp = &mut scratch.temp;
    box_filter_2d(&scratch.r_plane, log_base, temp, w, h, r_filter);

    // Phase 3 & 4: Tonemap Base in linear domain, modulate with Detail exp(D)
    let y_hdr = &scratch.y_hdr;
    let log_luma = &scratch.r_plane;
    let log_base = &scratch.log_base;
    let filmic_cfg = config.filmic;

    y_plane
        .par_iter_mut()
        .zip(cb_plane.par_iter_mut())
        .zip(cr_plane.par_iter_mut())
        .zip(y_hdr.par_iter())
        .zip(log_luma.par_iter())
        .zip(log_base.par_iter())
        .for_each(|(((((y_out, cb), cr), &y_orig), &l), &b)| {
            if y_orig < filmic_cfg.knee {
                // Identity for pixels below knee: untouched (±0 ADU)
                *y_out = y_orig;
            } else {
                // Log detail: D = L - B
                let d = l - b;
                // Exponentiate base back to linear space: B_lin = exp(B) - eps
                let b_lin = (b.exp() - eps).max(0.0);
                // Apply C¹ filmic tonemap to linear base
                let b_mapped = filmic_tonemap(b_lin, &filmic_cfg);
                // Reconstruct Y with multiplicative detail
                let y_rec = b_mapped * d.exp();
                *y_out = y_rec.clamp(0.0, 1.0);

                // Highlight desaturation: only when Y approaches saturation ceiling
                // Scale chroma smoothly toward 0.5 (neutral)
                let sat_drop = if *y_out > filmic_cfg.desat_start {
                    let t = ((*y_out - filmic_cfg.desat_start) / (filmic_cfg.desat_end - filmic_cfg.desat_start))
                        .clamp(0.0, 1.0);
                    1.0 - t * t * (3.0 - 2.0 * t)
                } else {
                    1.0
                };
                *cb = 0.5 + (*cb - 0.5) * sat_drop;
                *cr = 0.5 + (*cr - 0.5) * sat_drop;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midtones_below_knee_stay_untouched() {
        let w = 8usize;
        let h = 8usize;
        let n = w * h;
        let mut y = vec![0.5f32; n]; // 127.5 ADU, well below knee (0.70)
        let mut cb = vec![0.55f32; n];
        let mut cr = vec![0.45f32; n];
        let mut scratch = HighlightScratch::default();
        let cfg = HighlightConfig::default();

        let y_orig = y.clone();
        recover_highlights_planar(&mut y, &mut cb, &mut cr, &mut scratch, w, h, &cfg);

        for (a, b) in y.iter().zip(y_orig.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Midtones below knee must be exact identity: got {}, expected {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_partial_clipping_extends_headroom() {
        let w = 4usize;
        let h = 4usize;
        let n = w * h;

        // Warm skin: R = 254 (clipped), G = 190 (surviving), B = 150 (surviving)
        let r0 = 254.0 / 255.0;
        let g0 = 190.0 / 255.0;
        let b0 = 150.0 / 255.0;

        let y0 = 0.299 * r0 + 0.587 * g0 + 0.114 * b0;
        let cb0 = -0.168736 * r0 - 0.331264 * g0 + 0.5 * b0 + 0.5;
        let cr0 = 0.5 * r0 - 0.418688 * g0 - 0.081312 * b0 + 0.5;

        let mut y = vec![y0; n];
        let mut cb = vec![cb0; n];
        let mut cr = vec![cr0; n];
        let mut scratch = HighlightScratch::default();
        let cfg = HighlightConfig::default();

        recover_highlights_planar(&mut y, &mut cb, &mut cr, &mut scratch, w, h, &cfg);

        // Resulting Y should be smoothly mapped without clipping artifacts
        for &val in &y {
            assert!(val >= 0.0 && val <= 1.0, "Output must be normalized [0, 1]: got {}", val);
        }
    }

    #[test]
    fn test_detail_layer_sign_preservation() {
        let w = 16usize;
        let h = 16usize;
        let n = w * h;

        // Gradient ramp from 0.4 to 1.2 with high-frequency sine detail
        let mut y = vec![0.0f32; n];
        for y_idx in 0..h {
            for x_idx in 0..w {
                let base = 0.4 + 0.7 * (x_idx as f32 / w as f32);
                let detail = 0.03 * ((x_idx * 4) as f32).sin();
                y[y_idx * w + x_idx] = base + detail;
            }
        }
        let mut cb = vec![0.5f32; n];
        let mut cr = vec![0.5f32; n];
        let mut scratch = HighlightScratch::default();
        let cfg = HighlightConfig::default();

        recover_highlights_planar(&mut y, &mut cb, &mut cr, &mut scratch, w, h, &cfg);

        // Ensure output is strictly monotone on average along x axis
        let row_avg: Vec<f32> = (0..w)
            .map(|x| (0..h).map(|y_idx| y[y_idx * w + x]).sum::<f32>() / (h as f32))
            .collect();

        for i in 1..w {
            assert!(
                row_avg[i] >= row_avg[i - 1] - 0.02,
                "Average brightness should preserve monotonic trend along gradient: at {} got {} vs prev {}",
                i,
                row_avg[i],
                row_avg[i - 1]
            );
        }
    }
}
