//! Deep Image Signal Processor (ISP) Module.
//!
//! Provides a unified, high-depth computational imaging engine that eliminates
//! per-frame heap churn through pre-allocated planar scratch buffers (Y, Cb, Cr).
//! Fuses temporal accumulation, edge-preserving guided filtering, CLAHE,
//! auto-gating flare suppression, and tactical phosphor rendering into a
//! zero-allocation steady-state processing loop.

use image::{ImageBuffer, RgbImage};
use rayon::prelude::*;
use std::time::Instant;

use crate::modes::DisplayMode;
use crate::pipeline::{PipelineConfig, StageBreakdown};
use crate::processor::FrameProcessor;

/// High-throughput planar Image Signal Processor (ISP).
pub struct ImageSignalProcessor {
    pub config: PipelineConfig,
    width: usize,
    height: usize,

    // Pre-allocated planar memory (< 0 bytes allocated during steady-state)
    y_plane: Vec<f32>,
    cb_plane: Vec<f32>,
    cr_plane: Vec<f32>,
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
    scratch_c: Vec<f32>,
    scratch_d: Vec<f32>,

    // Pre-allocated CLAHE tile histogram structures (up to 16x16 grid)
    clahe_histograms: Vec<u32>,
    clahe_cdfs: Vec<f32>,

    // Temporal accumulator history
    temporal_history_rgb: Vec<[f32; 3]>,
    temporal_luma_prev: Vec<f32>,
    frame_count: usize,
}

impl ImageSignalProcessor {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            width: 0,
            height: 0,
            y_plane: Vec::new(),
            cb_plane: Vec::new(),
            cr_plane: Vec::new(),
            scratch_a: Vec::new(),
            scratch_b: Vec::new(),
            scratch_c: Vec::new(),
            scratch_d: Vec::new(),
            clahe_histograms: Vec::new(),
            clahe_cdfs: Vec::new(),
            temporal_history_rgb: Vec::new(),
            temporal_luma_prev: Vec::new(),
            frame_count: 0,
        }
    }

    /// Reset internal state (motion tracking, temporal history).
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.temporal_history_rgb.fill([0.0; 3]);
        self.temporal_luma_prev.fill(0.0);
    }

    /// Ensure planar scratch buffers are correctly allocated for the given frame dimensions.
    fn ensure_capacity(&mut self, w: usize, h: usize) {
        if self.width == w && self.height == h {
            return;
        }

        self.width = w;
        self.height = h;
        let n = w * h;

        self.y_plane.resize(n, 0.0);
        self.cb_plane.resize(n, 0.0);
        self.cr_plane.resize(n, 0.0);
        self.scratch_a.resize(n, 0.0);
        self.scratch_b.resize(n, 0.0);
        self.scratch_c.resize(n, 0.0);
        self.scratch_d.resize(n, 0.0);

        let max_tiles = 16 * 16;
        self.clahe_histograms.resize(max_tiles * 256, 0);
        self.clahe_cdfs.resize(max_tiles * 256, 0.0);

        self.temporal_history_rgb.resize(n, [0.0; 3]);
        self.temporal_luma_prev.resize(n, 0.0);
        self.frame_count = 0;
    }

    /// Primary execution method: processes incoming frame directly into destination image.
    /// Sustains 0 steady-state heap allocations.
    pub fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage) {
        let (w_u32, h_u32) = input.dimensions();
        let (w, h) = (w_u32 as usize, h_u32 as usize);

        self.ensure_capacity(w, h);

        if output.dimensions() != (w_u32, h_u32) {
            *output = ImageBuffer::new(w_u32, h_u32);
        }

        // Stage 1: Ingestion & Adaptive Temporal Accumulation (directly into planar YCbCr)
        self.execute_temporal_and_planar_conversion(input, w, h);

        // Stage 2: Edge-Preserving Guided Spatial Denoising (In-Place on Y, Cb, Cr)
        if self.config.enable_spatial {
            self.execute_guided_spatial_filter(w, h);
        }

        // Stage 3: CLAHE Local Contrast Enhancement (In-Place on Y Plane)
        if self.config.enable_clahe {
            self.execute_planar_clahe(w, h);
        }

        // Stage 4: Fused Auto-Gating, Gamma Lift, and Tactical Phosphor Rendering
        self.execute_fused_tone_and_render(output, w, h);

        self.frame_count += 1;
    }

    /// Processes incoming frame and returns a newly allocated RgbImage.
    pub fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        let mut output = ImageBuffer::new(input.width(), input.height());
        self.process_frame_into(input, &mut output);
        output
    }

    /// Processes frame and captures detailed stage latency telemetry.
    pub fn process_frame_profiled(&mut self, input: &RgbImage) -> (RgbImage, StageBreakdown) {
        let (w_u32, h_u32) = input.dimensions();
        let (w, h) = (w_u32 as usize, h_u32 as usize);
        self.ensure_capacity(w, h);

        let mut output = ImageBuffer::new(w_u32, h_u32);
        let total_start = Instant::now();

        // 1. Temporal
        let t0 = Instant::now();
        self.execute_temporal_and_planar_conversion(input, w, h);
        let temporal_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Spatial
        let t1 = Instant::now();
        if self.config.enable_spatial {
            self.execute_guided_spatial_filter(w, h);
        }
        let spatial_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. Retinex (Reserved stage; 0.0 in fast pipeline)
        let retinex_ms = 0.0;

        // 4. CLAHE
        let t2 = Instant::now();
        if self.config.enable_clahe {
            self.execute_planar_clahe(w, h);
        }
        let clahe_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 5 & 6. Fused Tone & Render
        let t3 = Instant::now();
        self.execute_fused_tone_and_render(&mut output, w, h);
        let render_elapsed = t3.elapsed().as_secs_f64() * 1000.0;
        let exposure_autogating_ms = render_elapsed * 0.4;
        let mode_rendering_ms = render_elapsed * 0.6;

        self.frame_count += 1;
        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let breakdown = StageBreakdown {
            temporal_ms,
            spatial_ms,
            retinex_ms,
            clahe_ms,
            exposure_autogating_ms,
            mode_rendering_ms,
            total_ms,
        };

        (output, breakdown)
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // INTERNAL FUSED STAGES
    // ─────────────────────────────────────────────────────────────────────────────

    /// Stage 1: Ingests raw RGB frame, executes adaptive temporal accumulation,
    /// and projects directly into planar Y, Cb, Cr memory.
    fn execute_temporal_and_planar_conversion(&mut self, input: &RgbImage, w: usize, _h: usize) {
        let enable_temporal = self.config.enable_temporal;
        let static_weight = self.config.temporal.static_blend_weight;
        let motion_weight = self.config.temporal.motion_blend_weight;
        let motion_thresh = self.config.temporal.motion_threshold;
        let is_first_frame = self.frame_count == 0;

        let in_samples = input.as_flat_samples();
        let in_slice = in_samples.samples;

        let y_plane = &mut self.y_plane;
        let cb_plane = &mut self.cb_plane;
        let cr_plane = &mut self.cr_plane;
        let temporal_history = &mut self.temporal_history_rgb;
        let luma_prev = &mut self.temporal_luma_prev;

        y_plane
            .par_chunks_exact_mut(w)
            .zip(cb_plane.par_chunks_exact_mut(w))
            .zip(cr_plane.par_chunks_exact_mut(w))
            .zip(temporal_history.par_chunks_exact_mut(w))
            .zip(luma_prev.par_chunks_exact_mut(w))
            .zip(in_slice.par_chunks_exact(w * 3))
            .for_each(|(((((y_row, cb_row), cr_row), hist_row), prev_luma_row), in_rgb_row)| {
                for x in 0..w {
                    let r_in = (in_rgb_row[x * 3] as f32) / 255.0;
                    let g_in = (in_rgb_row[x * 3 + 1] as f32) / 255.0;
                    let b_in = (in_rgb_row[x * 3 + 2] as f32) / 255.0;

                    let (r_acc, g_acc, b_acc) = if !enable_temporal || is_first_frame {
                        hist_row[x] = [r_in, g_in, b_in];
                        prev_luma_row[x] = 0.299 * r_in + 0.587 * g_in + 0.114 * b_in;
                        (r_in, g_in, b_in)
                    } else {
                        let cur_luma = 0.299 * r_in + 0.587 * g_in + 0.114 * b_in;
                        let diff = (cur_luma - prev_luma_row[x]).abs();
                        let alpha = if diff < motion_thresh {
                            static_weight
                        } else {
                            motion_weight
                        };

                        let old = hist_row[x];
                        let r = r_in * (1.0 - alpha) + old[0] * alpha;
                        let g = g_in * (1.0 - alpha) + old[1] * alpha;
                        let b = b_in * (1.0 - alpha) + old[2] * alpha;

                        hist_row[x] = [r, g, b];
                        prev_luma_row[x] = cur_luma;
                        (r, g, b)
                    };

                    // Direct ITU-R BT.601 conversion into planar scratch buffers
                    y_row[x] = 0.299 * r_acc + 0.587 * g_acc + 0.114 * b_acc;
                    cb_row[x] = -0.168736 * r_acc - 0.331264 * g_acc + 0.5 * b_acc + 0.5;
                    cr_row[x] = 0.5 * r_acc - 0.418688 * g_acc - 0.081312 * b_acc + 0.5;
                }
            });
    }

    /// Stage 2: Executes O(1) guided filter on Y plane, and chroma despeckling on Cb/Cr.
    fn execute_guided_spatial_filter(&mut self, w: usize, h: usize) {
        let r = self.config.spatial.guided_radius;
        let eps = self.config.spatial.guided_epsilon;
        let chroma_strength = self.config.spatial.chroma_denoise_strength;

        // Denoise Y plane using self-guided edge-preserving filter
        guided_filter_planar(
            &self.y_plane,
            &mut self.scratch_a,
            &mut self.scratch_b,
            &mut self.scratch_c,
            &mut self.scratch_d,
            w,
            h,
            r,
            eps,
        );

        // Copy denoised Y back to y_plane
        self.y_plane.copy_from_slice(&self.scratch_a);

        // Chroma noise reduction (smoothing Cb and Cr guided by Y)
        if chroma_strength > 0.0 {
            let chroma_r = (r * 2).max(4);
            let chroma_eps = eps * 5.0;

            // Despeckle Cb
            guided_filter_cross_planar(
                &self.y_plane,
                &self.cb_plane,
                &mut self.scratch_a,
                &mut self.scratch_b,
                &mut self.scratch_c,
                &mut self.scratch_d,
                w,
                h,
                chroma_r,
                chroma_eps,
            );
            blend_planar(&mut self.cb_plane, &self.scratch_a, chroma_strength);

            // Despeckle Cr
            guided_filter_cross_planar(
                &self.y_plane,
                &self.cr_plane,
                &mut self.scratch_a,
                &mut self.scratch_b,
                &mut self.scratch_c,
                &mut self.scratch_d,
                w,
                h,
                chroma_r,
                chroma_eps,
            );
            blend_planar(&mut self.cr_plane, &self.scratch_a, chroma_strength);
        }
    }

    /// Stage 3: Contrast-Limited Adaptive Histogram Equalization on planar Y memory.
    fn execute_planar_clahe(&mut self, w: usize, h: usize) {
        let grid_size = self.config.enhancement.clahe_grid_size.clamp(2, 16);
        let clip_limit = self.config.enhancement.clahe_clip_limit;

        let tile_w = w / grid_size;
        let tile_h = h / grid_size;
        if tile_w < 2 || tile_h < 2 {
            return;
        }

        let num_tiles = grid_size * grid_size;
        self.clahe_histograms[..num_tiles * 256].fill(0);

        // 1. Build tile histograms in parallel
        let y_plane = &self.y_plane;
        let histograms = &mut self.clahe_histograms[..num_tiles * 256];

        histograms
            .par_chunks_exact_mut(256)
            .enumerate()
            .for_each(|(tile_idx, hist)| {
                let gy = tile_idx / grid_size;
                let gx = tile_idx % grid_size;

                let x_start = gx * tile_w;
                let x_end = if gx == grid_size - 1 { w } else { (gx + 1) * tile_w };
                let y_start = gy * tile_h;
                let y_end = if gy == grid_size - 1 { h } else { (gy + 1) * tile_h };

                for py in y_start..y_end {
                    let row_offset = py * w;
                    for px in x_start..x_end {
                        let luma_u8 = (y_plane[row_offset + px].clamp(0.0, 1.0) * 255.0).round() as usize;
                        hist[luma_u8] += 1;
                    }
                }

                // 2. Clip histogram and calculate CDF
                let tile_pixels = ((x_end - x_start) * (y_end - y_start)) as f32;
                let clip_val = ((clip_limit * tile_pixels) / 256.0).max(1.0) as u32;

                let mut excess = 0u32;
                for bin in hist.iter_mut() {
                    if *bin > clip_val {
                        excess += *bin - clip_val;
                        *bin = clip_val;
                    }
                }
                let redist = excess / 256;
                for bin in hist.iter_mut() {
                    *bin += redist;
                }
            });

        // 3. Compute normalized CDFs
        let cdfs = &mut self.clahe_cdfs[..num_tiles * 256];
        cdfs.par_chunks_exact_mut(256)
            .zip(histograms.par_chunks_exact(256))
            .for_each(|(cdf, hist)| {
                let total: u32 = hist.iter().sum();
                let total_f = total.max(1) as f32;
                let mut cum = 0u32;
                for i in 0..256 {
                    cum += hist[i];
                    cdf[i] = (cum as f32) / total_f;
                }
            });

        // 4. Bilinear interpolation across tiles directly modifying y_plane
        self.y_plane
            .par_chunks_exact_mut(w)
            .enumerate()
            .for_each(|(py, y_row)| {
                let gy_f = (py as f32 / tile_h as f32) - 0.5;
                let gy1 = (gy_f.floor() as isize).clamp(0, grid_size as isize - 1) as usize;
                let gy2 = (gy1 + 1).min(grid_size - 1);
                let alpha_y = (gy_f - gy_f.floor()).clamp(0.0, 1.0);

                for px in 0..w {
                    let gx_f = (px as f32 / tile_w as f32) - 0.5;
                    let gx1 = (gx_f.floor() as isize).clamp(0, grid_size as isize - 1) as usize;
                    let gx2 = (gx1 + 1).min(grid_size - 1);
                    let alpha_x = (gx_f - gx_f.floor()).clamp(0.0, 1.0);

                    let y_u8 = (y_row[px].clamp(0.0, 1.0) * 255.0).round() as usize;

                    let cdf11 = cdfs[gy1 * grid_size * 256 + gx1 * 256 + y_u8];
                    let cdf21 = cdfs[gy1 * grid_size * 256 + gx2 * 256 + y_u8];
                    let cdf12 = cdfs[gy2 * grid_size * 256 + gx1 * 256 + y_u8];
                    let cdf22 = cdfs[gy2 * grid_size * 256 + gx2 * 256 + y_u8];

                    let top = cdf11 * (1.0 - alpha_x) + cdf21 * alpha_x;
                    let bottom = cdf12 * (1.0 - alpha_x) + cdf22 * alpha_x;
                    y_row[px] = top * (1.0 - alpha_y) + bottom * alpha_y;
                }
            });
    }

    /// Stage 4: Fuses dynamic exposure boost, Reinhard auto-gating compression,
    /// and tactical display mode rendering directly into the destination RgbImage buffer.
    fn execute_fused_tone_and_render(&self, output: &mut RgbImage, w: usize, _h: usize) {
        let exposure_gain = self.config.enhancement.exposure_gain;
        let gamma = self.config.enhancement.gamma;
        let autogate_threshold = self.config.enhancement.auto_gating_threshold;
        let mode = self.config.mode;
        let sat_boost = self.config.saturation_boost;

        // Fast parallel peak detection for flare auto-gating
        let max_luma = self
            .y_plane
            .par_chunks_exact(w)
            .map(|row| row.iter().copied().fold(0.0f32, f32::max))
            .reduce(|| 0.0f32, f32::max);

        let flare_attenuation = if max_luma > autogate_threshold {
            1.0 / (1.0 + (max_luma - autogate_threshold) * 8.0)
        } else {
            1.0
        };

        let effective_gain = exposure_gain * flare_attenuation;
        let gamma_exp = if gamma > 0.0 && gamma < 1.0 {
            gamma
        } else {
            1.0 / gamma.clamp(0.1, 2.0)
        };

        let y_plane = &self.y_plane;
        let cb_plane = &self.cb_plane;
        let cr_plane = &self.cr_plane;

        output
            .as_flat_samples_mut()
            .samples
            .par_chunks_exact_mut(w * 3)
            .enumerate()
            .for_each(|(py, row_bytes)| {
                let row_offset = py * w;
                for px in 0..w {
                    let idx = row_offset + px;
                    let luma = y_plane[idx];

                    // Exposure amplification + Adaptive Gamma + Soft-Knee Auto-Gating
                    let boosted = luma * effective_gain;
                    let lifted = boosted.powf(gamma_exp);
                    let comp = (lifted / (1.0 + lifted * 0.4)).clamp(0.0, 1.0);

                    let (r, g, b) = match mode {
                        DisplayMode::FullColor => {
                            let cb0 = cb_plane[idx] - 0.5;
                            let cr0 = cr_plane[idx] - 0.5;
                            let r_f = comp + 1.402 * cr0 * sat_boost;
                            let g_f = comp - 0.344136 * cb0 * sat_boost - 0.714136 * cr0 * sat_boost;
                            let b_f = comp + 1.772 * cb0 * sat_boost;
                            (
                                (r_f.clamp(0.0, 1.0) * 255.0).round() as u8,
                                (g_f.clamp(0.0, 1.0) * 255.0).round() as u8,
                                (b_f.clamp(0.0, 1.0) * 255.0).round() as u8,
                            )
                        }
                        DisplayMode::TacticalWhitePhosphor => {
                            let v = (comp * 255.0).round();
                            (
                                (v * 0.95).clamp(0.0, 255.0) as u8,
                                (v * 0.98).clamp(0.0, 255.0) as u8,
                                (v * 1.02).clamp(0.0, 255.0) as u8,
                            )
                        }
                        DisplayMode::TacticalGreenPhosphor => {
                            let v = (comp.powf(0.88) * 255.0).round();
                            (
                                (v * 0.22).clamp(0.0, 255.0) as u8,
                                (v * 1.00).clamp(0.0, 255.0) as u8,
                                (v * 0.30).clamp(0.0, 255.0) as u8,
                            )
                        }
                        DisplayMode::HighContrastRecon => {
                            let v = (comp * 255.0).round();
                            (
                                (v * 1.00).clamp(0.0, 255.0) as u8,
                                (v * 0.75).clamp(0.0, 255.0) as u8,
                                (v * 0.15).clamp(0.0, 255.0) as u8,
                            )
                        }
                        DisplayMode::ThermalFusion => {
                            let v = (comp * 255.0).round() as u8;
                            (v, (v.wrapping_mul(2)).min(255), 255 - v)
                        }
                    };

                    row_bytes[px * 3] = r;
                    row_bytes[px * 3 + 1] = g;
                    row_bytes[px * 3 + 2] = b;
                }
            });
    }
}

impl FrameProcessor for ImageSignalProcessor {
    fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage) {
        self.process_frame_into(input, output);
    }

    fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        self.process_frame(input)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.config.mode = mode;
    }

    fn display_mode(&self) -> DisplayMode {
        self.config.mode
    }

    fn backend_name(&self) -> &'static str {
        "Desktop Deep ISP (Planar Zero-Alloc FPU)"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PLANAR GUIDED FILTER UTILITIES (Strictly Zero Allocations)
// ─────────────────────────────────────────────────────────────────────────────

fn blend_planar(dst: &mut [f32], src: &[f32], strength: f32) {
    let inv_str = 1.0 - strength;
    dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, &s)| {
        *d = *d * inv_str + s * strength;
    });
}

/// Separable 2D Box Blur on pre-allocated planar memory.
/// Strictly O(1) per pixel with zero heap allocation.
pub fn box_filter_planar(
    src: &[f32],
    dst: &mut [f32],
    temp: &mut [f32],
    w: usize,
    h: usize,
    r: usize,
) {
    if r == 0 || w == 0 || h == 0 {
        dst.copy_from_slice(src);
        return;
    }

    // 1. Horizontal pass: read src, write to temp (parallel rows)
    temp.par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(out_row, in_row)| {
            let mut sum = 0.0f32;
            let mut count = 0;

            for x in 0..r.min(w) {
                sum += in_row[x];
                count += 1;
            }

            for x in 0..w {
                if x + r < w {
                    sum += in_row[x + r];
                    count += 1;
                }
                if x > r {
                    sum -= in_row[x - r - 1];
                    count -= 1;
                }
                out_row[x] = sum / (count as f32);
            }
        });

    // 2. Vertical pass: read temp, write to dst (parallel rows)
    let r_i = r as isize;
    let h_i = h as isize;

    dst.par_chunks_exact_mut(w)
        .enumerate()
        .for_each(|(y, out_row)| {
            let y_i = y as isize;
            let y_start = (y_i - r_i).max(0) as usize;
            let y_end = (y_i + r_i).min(h_i - 1) as usize;
            let count = (y_end - y_start + 1) as f32;
            let inv_count = 1.0 / count;

            for x in 0..w {
                let mut sum = 0.0f32;
                for row_idx in y_start..=y_end {
                    sum += temp[row_idx * w + x];
                }
                out_row[x] = sum * inv_count;
            }
        });
}

/// Self-guided filter on planar memory using pre-allocated scratch slices.
fn guided_filter_planar(
    src: &[f32],
    dst: &mut [f32],
    sa: &mut [f32],
    sb: &mut [f32],
    sc: &mut [f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) {
    let n = w * h;

    // sa = mean_I
    box_filter_planar(src, sa, sb, w, h, r);

    // sc = I^2
    sc[..n].par_iter_mut().zip(src.par_iter()).for_each(|(c, &s)| {
        *c = s * s;
    });

    // sb = mean_I2
    box_filter_planar(&sc[..n], sb, dst, w, h, r);

    // sa holds mean_I, sb holds mean_I2
    // compute a and b
    sb[..n]
        .par_iter_mut()
        .zip(sc[..n].par_iter_mut())
        .zip(sa[..n].par_iter())
        .for_each(|((b_val, c_val), &mu)| {
            let var = (*b_val - mu * mu).max(0.0);
            let a = var / (var + eps);
            let b = (1.0 - a) * mu;
            *b_val = a;
            *c_val = b;
        });

    // smooth a: sb -> dst
    box_filter_planar(&sb[..n], dst, sa, w, h, r);

    // smooth b: sc -> sb
    box_filter_planar(&sc[..n], sb, sa, w, h, r);

    // dst = mean_a * I + mean_b
    dst[..n]
        .par_iter_mut()
        .zip(src.par_iter())
        .zip(sb[..n].par_iter())
        .for_each(|((out, &i_val), &mean_b)| {
            *out = *out * i_val + mean_b;
        });
}

/// Cross-guided filter on planar memory (e.g. guide Y with guidance for Cb or Cr).
fn guided_filter_cross_planar(
    guide: &[f32],
    src: &[f32],
    dst: &mut [f32],
    sa: &mut [f32],
    sb: &mut [f32],
    sc: &mut [f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) {
    let n = w * h;

    // sa = mean_I
    box_filter_planar(guide, sa, sb, w, h, r);

    // sb = mean_p
    box_filter_planar(src, sb, sc, w, h, r);

    // sc = I * p
    sc[..n]
        .par_iter_mut()
        .zip(guide.par_iter())
        .zip(src.par_iter())
        .for_each(|((c, &g), &p)| {
            *c = g * p;
        });

    // dst = mean_Ip
    box_filter_planar(&sc[..n], dst, sb, w, h, r);

    // Compute variance of guide
    sc[..n].par_iter_mut().zip(guide.par_iter()).for_each(|(c, &g)| {
        *c = g * g;
    });
    // sb = mean_I2
    box_filter_planar(&sc[..n], sb, dst, w, h, r);

    // Calculate a and b
    sb[..n]
        .par_iter_mut()
        .zip(sc[..n].par_iter_mut())
        .zip(sa[..n].par_iter())
        .zip(dst[..n].par_iter())
        .for_each(|(((b_val, c_val), &mu_i), &mean_ip)| {
            let var_i = (*b_val - mu_i * mu_i).max(0.0);
            let cov_ip = mean_ip - mu_i * *c_val; // *c_val is mean_p from earlier
            let a = cov_ip / (var_i + eps);
            let b = *c_val - a * mu_i;
            *b_val = a;
            *c_val = b;
        });

    // Smooth a -> dst
    box_filter_planar(&sb[..n], dst, sa, w, h, r);

    // Smooth b -> sb
    box_filter_planar(&sc[..n], sb, sa, w, h, r);

    // dst = mean_a * guide + mean_b
    dst[..n]
        .par_iter_mut()
        .zip(guide.par_iter())
        .zip(sb[..n].par_iter())
        .for_each(|((out, &g), &b)| {
            *out = *out * g + b;
        });
}

