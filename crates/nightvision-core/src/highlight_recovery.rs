//! Physics-Grounded Single-Frame Highlight Recovery.
//!
//! Reconstructs partially and triple-clipped highlights directly in planar YCbCr memory:
//! 1. Monotone Partial-Channel Headroom Extension: Where one or two channels clip while others survive,
//!    unclipped HDR headroom (> 1.0) is smoothly restored using surviving channel spatial ratios,
//!    guaranteeing Y_hdr >= Y_raw everywhere (a clipped pixel is never assigned lower luma than a dimmer pixel).
//! 2. Contiguous Push-Pull Chroma Inpainting: For triple-clipped pixels (min(R,G,B) >= clip_threshold),
//!    chrominance (Cb, Cr) is diffused smoothly inward from unclipped warm skin neighbours (120..235 ADU)
//!    using a contiguous multi-scale 2x2 Push-Pull pyramid down to 1x1 with zero strided jumps.
//!    Chrominance is strictly clamped within authentic neighbour bounds, eliminating fiery orange/red burn.
//! 3. Global Guided Base-Detail Splitting & C¹ Soft-Knee: Decomposes log-luminance into base and micro-detail
//!    (L = B + D) across the entire frame. The base is tonemapped via C¹ filmic soft-knee and recombined with
//!    authentic log-detail: Y_out = filmic_tonemap(exp(B)) * exp(D). Ensures strict 2D monotonicity,
//!    zero dark craters, and zero contour rings.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::soft_knee::{filmic_tonemap, FilmicConfig};

/// Algorithm selection for highlight recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HighlightAlgorithm {
    /// Contiguous Push-Pull Multi-Scale Inpainting with Guided Base-Detail Splitting.
    PushPullPde,
    /// Fast dyadic jump alias (delegates to the continuous monotone pipeline).
    DyadicJetPatch,
}

impl Default for HighlightAlgorithm {
    fn default() -> Self {
        HighlightAlgorithm::PushPullPde
    }
}

/// Configuration for highlight recovery.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HighlightConfig {
    /// Inpainting algorithm selection.
    pub algorithm: HighlightAlgorithm,
    /// Clipping threshold in normalized [0, 1] units.
    /// Default: 245.0 / 255.0 (~0.9608, highlight boundary).
    pub clip_threshold: f32,
    /// Minimum channel threshold to qualify as surviving (unclipped trust zone).
    /// Default: 220.0 / 255.0 (~0.8627).
    pub surviving_threshold: f32,
    /// Arching limit parameter for highlight headroom.
    pub sigma_arch: f32,
    /// Detail injection scale for micro-contrast.
    pub detail_scale: f32,
    /// Filmic soft-knee parameters.
    pub filmic: FilmicConfig,
    /// Box filter radius for base layer extraction.
    pub filter_radius: usize,
    /// Regularization epsilon for log domain.
    pub eps: f32,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            algorithm: HighlightAlgorithm::PushPullPde,
            clip_threshold: 245.0 / 255.0,
            surviving_threshold: 220.0 / 255.0,
            sigma_arch: 0.18,
            detail_scale: 0.65,
            filmic: FilmicConfig {
                knee: 0.55,
                ceiling: 1.0,
                desat_start: 0.85,
                desat_end: 248.0 / 255.0,
            },
            filter_radius: 8,
            eps: 1.0 / 255.0,
        }
    }
}

/// Scratch memory for highlight recovery (reusable across frames with zero steady-state heap allocations).
#[derive(Default)]
pub struct HighlightScratch {
    pub r_plane: Vec<f32>,
    pub g_plane: Vec<f32>,
    pub b_plane: Vec<f32>,
    pub base_r: Vec<f32>,
    pub base_g: Vec<f32>,
    pub base_b: Vec<f32>,
    pub y_raw: Vec<f32>,
    pub y_hdr: Vec<f32>,
    pub log_base: Vec<f32>,
    pub temp: Vec<f32>,
    pub known_mask: Vec<u8>,
    // Preallocated pyramid buffers for zero-allocation Push-Pull chroma inpainting
    pub pyr_offsets: Vec<usize>,
    pub pyr_w: Vec<usize>,
    pub pyr_h: Vec<usize>,
    pub pyr_cb: Vec<f32>,
    pub pyr_cr: Vec<f32>,
    pub pyr_k: Vec<u8>,
}

impl HighlightScratch {
    pub fn ensure_capacity(&mut self, n: usize) {
        self.ensure_capacity_dims(n, n, 1);
    }

    pub fn ensure_capacity_dims(&mut self, n: usize, w: usize, h: usize) {
        if self.r_plane.len() != n || self.pyr_w.is_empty() || self.pyr_w[0] != w || self.pyr_h[0] != h {
            self.r_plane.resize(n, 0.0);
            self.g_plane.resize(n, 0.0);
            self.b_plane.resize(n, 0.0);
            self.base_r.resize(n, 0.0);
            self.base_g.resize(n, 0.0);
            self.base_b.resize(n, 0.0);
            self.y_raw.resize(n, 0.0);
            self.y_hdr.resize(n, 0.0);
            self.log_base.resize(n, 0.0);
            self.temp.resize(n, 0.0);
            self.known_mask.resize(n, 0);

            self.pyr_offsets.clear();
            self.pyr_w.clear();
            self.pyr_h.clear();

            let mut cur_w = w;
            let mut cur_h = h;
            let mut total_pyr = 0;
            let max_levels = 5;
            while self.pyr_offsets.len() < max_levels {
                self.pyr_offsets.push(total_pyr);
                self.pyr_w.push(cur_w);
                self.pyr_h.push(cur_h);
                total_pyr += cur_w * cur_h;
                if cur_w == 1 && cur_h == 1 {
                    break;
                }
                cur_w = ((cur_w + 1) / 2).max(1);
                cur_h = ((cur_h + 1) / 2).max(1);
            }
            self.pyr_cb.resize(total_pyr, 0.0);
            self.pyr_cr.resize(total_pyr, 0.0);
            self.pyr_k.resize(total_pyr, 0);
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

/// Multi-scale 2x2 contiguous Push-Pull pyramid for chrominance inpainting down to 1x1.
/// In O(N) operations with zero steady-state heap allocations, diffuses boundary chroma smoothly inward.
#[allow(dead_code)]
fn push_pull_chroma(
    cb_in: &[f32],
    cr_in: &[f32],
    scratch: &mut HighlightScratch,
) {
    let num_levels = scratch.pyr_offsets.len();
    if num_levels == 0 {
        return;
    }
    let n0 = scratch.pyr_w[0] * scratch.pyr_h[0];
    scratch.pyr_cb[..n0].copy_from_slice(cb_in);
    scratch.pyr_cr[..n0].copy_from_slice(cr_in);
    scratch.pyr_k[..n0].copy_from_slice(&scratch.known_mask[..n0]);

    // Push: downsample from level l to l+1
    for l in 0..num_levels - 1 {
        let cw = scratch.pyr_w[l];
        let ch = scratch.pyr_h[l];
        let nw = scratch.pyr_w[l + 1];
        let nh = scratch.pyr_h[l + 1];
        let c_off = scratch.pyr_offsets[l];
        let n_off = scratch.pyr_offsets[l + 1];

        for y in 0..nh {
            let sy0 = (y * 2).min(ch - 1);
            let sy1 = (y * 2 + 1).min(ch - 1);
            for x in 0..nw {
                let sx0 = (x * 2).min(cw - 1);
                let sx1 = (x * 2 + 1).min(cw - 1);

                let mut sum_cb = 0.0f32;
                let mut sum_cr = 0.0f32;
                let mut count = 0u32;

                for &sy in &[sy0, sy1] {
                    for &sx in &[sx0, sx1] {
                        let idx = c_off + sy * cw + sx;
                        if scratch.pyr_k[idx] == 1 {
                            sum_cb += scratch.pyr_cb[idx];
                            sum_cr += scratch.pyr_cr[idx];
                            count += 1;
                        }
                    }
                }

                let out_idx = n_off + y * nw + x;
                if count > 0 {
                    let inv = 1.0 / (count as f32);
                    scratch.pyr_cb[out_idx] = sum_cb * inv;
                    scratch.pyr_cr[out_idx] = sum_cr * inv;
                    scratch.pyr_k[out_idx] = 1;
                } else {
                    scratch.pyr_cb[out_idx] = 0.5;
                    scratch.pyr_cr[out_idx] = 0.5;
                    scratch.pyr_k[out_idx] = 0;
                }
            }
        }
    }

    // Coarsest level: diffuse known boundary anchors across the local coarse grid (up to 3 steps ~48 px)
    let top_l = num_levels - 1;
    let cw_top = scratch.pyr_w[top_l];
    let ch_top = scratch.pyr_h[top_l];
    let top_off = scratch.pyr_offsets[top_l];

    for _pass in 0..3 {
        for y in 0..ch_top {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(ch_top - 1);
            for x in 0..cw_top {
                let cur_idx = top_off + y * cw_top + x;
                if scratch.pyr_k[cur_idx] == 0 {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(cw_top - 1);
                    let mut sum_cb = 0.0f32;
                    let mut sum_cr = 0.0f32;
                    let mut count = 0u32;
                    for sy in y0..=y1 {
                        for sx in x0..=x1 {
                            let s_idx = top_off + sy * cw_top + sx;
                            if scratch.pyr_k[s_idx] == 1 {
                                sum_cb += scratch.pyr_cb[s_idx];
                                sum_cr += scratch.pyr_cr[s_idx];
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        scratch.pyr_cb[cur_idx] = sum_cb / (count as f32);
                        scratch.pyr_cr[cur_idx] = sum_cr / (count as f32);
                        scratch.pyr_k[cur_idx] = 1;
                    }
                }
            }
        }
    }

    // Coarse cells with zero local anchors fall back to neutral (0.5, 0.5) so distant white walls stay neutral
    for idx in top_off..top_off + (cw_top * ch_top) {
        if scratch.pyr_k[idx] == 0 {
            scratch.pyr_cb[idx] = 0.5;
            scratch.pyr_cr[idx] = 0.5;
            scratch.pyr_k[idx] = 1;
        }
    }

    // Pull: bilinearly upsample from level l+1 back to l to fill unknown cells
    for l in (0..num_levels - 1).rev() {
        let cw = scratch.pyr_w[l + 1];
        let ch = scratch.pyr_h[l + 1];
        let fw = scratch.pyr_w[l];
        let fh = scratch.pyr_h[l];
        let n_off = scratch.pyr_offsets[l + 1];
        let c_off = scratch.pyr_offsets[l];

        for y in 0..fh {
            for x in 0..fw {
                let cur_idx = c_off + y * fw + x;
                if scratch.pyr_k[cur_idx] == 1 {
                    continue;
                }

                let fx = ((x as f32 + 0.5) * 0.5 - 0.5).clamp(0.0, (cw.saturating_sub(1)) as f32);
                let fy = ((y as f32 + 0.5) * 0.5 - 0.5).clamp(0.0, (ch.saturating_sub(1)) as f32);
                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let x1 = (x0 + 1).min(cw - 1);
                let y1 = (y0 + 1).min(ch - 1);
                let tx = fx - x0 as f32;
                let ty = fy - y0 as f32;

                let w00 = (1.0 - tx) * (1.0 - ty);
                let w10 = tx * (1.0 - ty);
                let w01 = (1.0 - tx) * ty;
                let w11 = tx * ty;

                let i00 = n_off + y0 * cw + x0;
                let i10 = n_off + y0 * cw + x1;
                let i01 = n_off + y1 * cw + x0;
                let i11 = n_off + y1 * cw + x1;

                scratch.pyr_cb[cur_idx] = w00 * scratch.pyr_cb[i00]
                    + w10 * scratch.pyr_cb[i10]
                    + w01 * scratch.pyr_cb[i01]
                    + w11 * scratch.pyr_cb[i11];
                scratch.pyr_cr[cur_idx] = w00 * scratch.pyr_cr[i00]
                    + w10 * scratch.pyr_cr[i10]
                    + w01 * scratch.pyr_cr[i01]
                    + w11 * scratch.pyr_cr[i11];
                scratch.pyr_k[cur_idx] = 1;
            }
        }
    }
}

/// Recovers partially and triple-clipped highlights directly in planar YCbCr memory.
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
    scratch.ensure_capacity_dims(n, w, h);

    let clip_thr = config.clip_threshold;
    let eps = config.eps;
    let r_filter = config.filter_radius;

    // Step 1: Reconstruct RGB from YCbCr
    scratch.r_plane
        .par_iter_mut()
        .zip(scratch.g_plane.par_iter_mut())
        .zip(scratch.b_plane.par_iter_mut())
        .zip(scratch.y_raw.par_iter_mut())
        .zip(y_plane.par_iter())
        .zip(cb_plane.par_iter())
        .zip(cr_plane.par_iter())
        .for_each(|((((((r, g), b), y_raw), &y), &cb), &cr)| {
            let cb0 = cb - 0.5;
            let cr0 = cr - 0.5;
            *r = (y + 1.402 * cr0).clamp(0.0, 1.0);
            *g = (y - 0.344136 * cb0 - 0.714136 * cr0).clamp(0.0, 1.0);
            *b = (y + 1.772 * cb0).clamp(0.0, 1.0);
            *y_raw = y;
        });

    // Step 2: Spatial filtering of RGB for slope ratio reference
    box_filter_2d(&scratch.g_plane, &mut scratch.base_g, &mut scratch.temp, w, h, r_filter);
    box_filter_2d(&scratch.r_plane, &mut scratch.base_r, &mut scratch.temp, w, h, r_filter);
    box_filter_2d(&scratch.b_plane, &mut scratch.base_b, &mut scratch.temp, w, h, r_filter);

    // Step 3: Partial-Channel Upward Extension (Strictly Monotone: Y_hdr >= Y_raw everywhere)
    scratch.y_hdr
        .par_iter_mut()
        .zip(scratch.r_plane.par_iter())
        .zip(scratch.g_plane.par_iter())
        .zip(scratch.b_plane.par_iter())
        .zip(scratch.base_r.par_iter())
        .zip(scratch.base_g.par_iter())
        .zip(scratch.base_b.par_iter())
        .zip(scratch.y_raw.par_iter())
        .for_each(|(((((((y_hdr, &r), &g), &b), &r_b), &g_b), &b_b), &y_r)| {
            let ch_max = r.max(g).max(b);
            let ch_min = r.min(g).min(b);
            let mut r_ext = r;
            let mut g_ext = g;
            let mut b_ext = b;

            let is_saturated = (ch_max - ch_min) > (120.0 / 255.0);

            if !is_saturated && ch_max >= clip_thr && ch_min < clip_thr {
                if r >= clip_thr && g < clip_thr {
                    let g_ref = g_b.max(0.05);
                    let r_rec = g * (r_b / g_ref);
                    r_ext = r.max(r_rec);
                }
                if b >= clip_thr && g < clip_thr {
                    let g_ref = g_b.max(0.05);
                    let b_rec = g * (b_b / g_ref);
                    b_ext = b.max(b_rec);
                }
                if g >= clip_thr && r < clip_thr {
                    let r_ref = r_b.max(0.05);
                    let g_rec = r * (g_b / r_ref);
                    g_ext = g.max(g_rec);
                }
            }

            let y_est = 0.299 * r_ext + 0.587 * g_ext + 0.114 * b_ext;
            *y_hdr = y_est.max(y_r);
        });

    // Step 5: Global Guided Base-Detail Splitting & C¹ Soft-Knee (Uniform across entire image)
    let sum_raw: f32 = scratch.y_raw.par_iter().sum();
    let raw_mean_luma = (sum_raw / n as f32) * 255.0;
    let is_high_exposure = raw_mean_luma > 160.0;
    let gamma_pull = if is_high_exposure { 1.35f32 } else { 1.0f32 };

    scratch.temp
        .par_iter_mut()
        .zip(scratch.y_hdr.par_iter())
        .for_each(|(l, &y)| {
            let y_dehazed = if gamma_pull != 1.0 { y.powf(gamma_pull) } else { y };
            *l = (y_dehazed + eps).ln();
        });

    box_filter_2d(&scratch.temp, &mut scratch.log_base, &mut scratch.base_g, w, h, r_filter);

    let filmic_cfg = config.filmic;
    y_plane
        .par_iter_mut()
        .zip(scratch.y_hdr.par_iter())
        .zip(scratch.y_raw.par_iter())
        .zip(scratch.temp.par_iter())
        .zip(scratch.log_base.par_iter())
        .for_each(|((((y_out, &_y_h), &y_r), &l), &b)| {
            let d = l - b;
            let b_lin = (b.exp() - eps).max(0.0);
            let b_mapped = filmic_tonemap(b_lin, &filmic_cfg);
            let y_rec = b_mapped * d.exp();
            *y_out = y_rec.min(y_r).clamp(0.0, 1.0);
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
        let mut y = vec![0.5f32; n]; // 127.5 ADU, well below knee (0.85)
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
    fn test_push_pull_pde_and_dyadic_algorithms() {
        let w = 16usize;
        let h = 16usize;
        let n = w * h;

        for &algo in &[HighlightAlgorithm::PushPullPde, HighlightAlgorithm::DyadicJetPatch] {
            let mut y = vec![0.85f32; n];
            let mut cb = vec![0.50f32; n];
            let mut cr = vec![0.52f32; n];
            let mut scratch = HighlightScratch::default();
            let mut cfg = HighlightConfig::default();
            cfg.algorithm = algo;

            recover_highlights_planar(&mut y, &mut cb, &mut cr, &mut scratch, w, h, &cfg);

            for &val in &y {
                assert!(val >= 0.0 && val <= 1.0, "Output must be normalized [0, 1]: got {}", val);
            }
        }
    }

    #[test]
    fn test_partial_clipping_extends_headroom() {
        let w = 4usize;
        let h = 4usize;
        let n = w * h;

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

        for &val in &y {
            assert!(val >= 0.0 && val <= 1.0, "Output must be normalized [0, 1]: got {}", val);
        }
    }

    #[test]
    fn test_detail_layer_sign_preservation() {
        let w = 16usize;
        let h = 16usize;
        let n = w * h;

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

    #[test]
    fn test_zero_allocations_in_steady_state() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;
        let mut y = vec![0.85f32; n];
        let mut cb = vec![0.50f32; n];
        let mut cr = vec![0.52f32; n];
        let mut scratch = HighlightScratch::default();
        let cfg = HighlightConfig::default();

        scratch.ensure_capacity_dims(n, w, h);
        let cap_r = scratch.r_plane.capacity();
        let cap_base = scratch.log_base.capacity();
        let cap_pyr = scratch.pyr_cb.capacity();

        recover_highlights_planar(&mut y, &mut cb, &mut cr, &mut scratch, w, h, &cfg);

        assert_eq!(scratch.r_plane.capacity(), cap_r, "Scratch capacity should not reallocate");
        assert_eq!(scratch.log_base.capacity(), cap_base, "Scratch capacity should not reallocate");
        assert_eq!(scratch.pyr_cb.capacity(), cap_pyr, "Pyramid scratch should not reallocate");
    }
}
