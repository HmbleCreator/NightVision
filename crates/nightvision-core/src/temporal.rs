//! Dynamic motion-compensated multi-frame temporal SNR accumulator.
//!
//! In digital night vision, photon starvation is the fundamental limit.
//! Integrating photon flux across consecutive video frames boosts Signal-to-Noise
//! Ratio by sqrt(K) (+12 dB for 16 frames).
//!
//! To prevent motion blur and ghosting when objects or the camera move,
//! this module implements an adaptive recursive temporal filter with per-pixel
//! motion detection and noise floor gating.

use image::{ImageBuffer, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration parameters for temporal accumulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConfig {
    /// Accumulation factor for static regions (0.0 to 0.98).
    /// 0.90 corresponds to ~10 frames effective depth, 0.95 corresponds to ~20 frames depth.
    pub static_blend_weight: f32,
    /// Accumulation factor for moving regions (0.0 to 0.3). Low values prevent ghosting.
    pub motion_blend_weight: f32,
    /// Motion sensitivity threshold (0.01 to 0.25). Lower values trigger motion detection faster.
    pub motion_threshold: f32,
    /// Whether to enable motion mask dilation to protect moving silhouettes.
    pub enable_motion_dilation: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            static_blend_weight: 0.92,
            motion_blend_weight: 0.08,
            motion_threshold: 0.075,
            enable_motion_dilation: true,
        }
    }
}

/// Adaptive Temporal Accumulator maintaining internal temporal history.
#[derive(Debug, Clone)]
pub struct AdaptiveTemporalAccumulator {
    pub config: TemporalConfig,
    width: u32,
    height: u32,
    /// Internal accumulated floating-point buffer [0.0, 1.0] in RGB
    buffer: Vec<[f32; 3]>,
    /// Motion mask from the last processed frame (0.0 = stationary, 1.0 = rapid motion)
    motion_mask: Vec<f32>,
    frame_count: usize,
}

impl AdaptiveTemporalAccumulator {
    pub fn new(config: TemporalConfig) -> Self {
        Self {
            config,
            width: 0,
            height: 0,
            buffer: Vec::new(),
            motion_mask: Vec::new(),
            frame_count: 0,
        }
    }

    /// Reset accumulator history (e.g. on scene cut or sensor mode change)
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.motion_mask.clear();
        self.frame_count = 0;
    }

    /// Process incoming video frame through the adaptive temporal filter.
    pub fn process_frame(&mut self, current_frame: &RgbImage) -> RgbImage {
        let (w, h) = current_frame.dimensions();
        let pixel_count = (w * h) as usize;

        // Initialize or resize if dimensions changed
        if self.width != w || self.height != h || self.buffer.len() != pixel_count {
            self.width = w;
            self.height = h;
            self.buffer = vec![[0.0; 3]; pixel_count];
            self.motion_mask = vec![0.0; pixel_count];
            self.frame_count = 0;
        }

        // On the very first frame, seed the buffer directly
        if self.frame_count == 0 {
            for (idx, p) in current_frame.pixels().enumerate() {
                self.buffer[idx] = [
                    (p[0] as f32) / 255.0,
                    (p[1] as f32) / 255.0,
                    (p[2] as f32) / 255.0,
                ];
            }
            self.frame_count = 1;
            return current_frame.clone();
        }

        let raw_pixels: Vec<[f32; 3]> = current_frame
            .pixels()
            .map(|p| {
                [
                    (p[0] as f32) / 255.0,
                    (p[1] as f32) / 255.0,
                    (p[2] as f32) / 255.0,
                ]
            })
            .collect();

        // 1. Compute raw motion map by comparing current frame luminance with accumulated state
        let mut raw_motion = vec![0.0f32; pixel_count];
        let thresh = self.config.motion_threshold;

        raw_motion
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, motion_val)| {
                let cur = raw_pixels[idx];
                let acc = self.buffer[idx];

                // Rec.601 Luminance
                let y_cur = 0.299 * cur[0] + 0.587 * cur[1] + 0.114 * cur[2];
                let y_acc = 0.299 * acc[0] + 0.587 * acc[1] + 0.114 * acc[2];

                let delta = (y_cur - y_acc).abs();
                // Normalized motion response: 0.0 when below threshold, ramps up to 1.0
                let m = ((delta - thresh * 0.5) / (thresh * 1.5)).clamp(0.0, 1.0);
                *motion_val = m;
            });

        // 2. Optional motion dilation to prevent halo/ghosting along moving object contours
        let final_motion = if self.config.enable_motion_dilation && w > 2 && h > 2 {
            let mut dilated = vec![0.0f32; pixel_count];
            dilated
                .par_chunks_mut(w as usize)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..w as usize {
                        let mut max_m = raw_motion[y * (w as usize) + x];
                        for dy in [-1isize, 0, 1] {
                            let ny = y as isize + dy;
                            if ny >= 0 && ny < h as isize {
                                for dx in [-1isize, 0, 1] {
                                    let nx = x as isize + dx;
                                    if nx >= 0 && nx < w as isize {
                                        let neighbor_m =
                                            raw_motion[(ny as usize) * (w as usize) + (nx as usize)];
                                        if neighbor_m > max_m {
                                            max_m = neighbor_m;
                                        }
                                    }
                                }
                            }
                        }
                        row[x] = max_m;
                    }
                });
            dilated
        } else {
            raw_motion
        };

        self.motion_mask = final_motion;

        // 3. Adaptive temporal blending
        let static_w = self.config.static_blend_weight;
        let motion_w = self.config.motion_blend_weight;

        let mut output_img = ImageBuffer::new(w, h);
        let out_slice = output_img.as_flat_samples_mut();
        let out_bytes = out_slice.samples;

        self.buffer
            .par_iter_mut()
            .zip(raw_pixels.par_iter())
            .zip(self.motion_mask.par_iter())
            .zip(out_bytes.par_chunks_exact_mut(3))
            .for_each(|(((acc, cur), &motion), out_pixel)| {
                // Effective alpha interpolates between static_weight (high smoothing)
                // and motion_weight (instant response)
                let alpha = static_w * (1.0 - motion) + motion_w * motion;

                let r = alpha * acc[0] + (1.0 - alpha) * cur[0];
                let g = alpha * acc[1] + (1.0 - alpha) * cur[1];
                let b = alpha * acc[2] + (1.0 - alpha) * cur[2];

                *acc = [r, g, b];

                out_pixel[0] = (r * 255.0).round().clamp(0.0, 255.0) as u8;
                out_pixel[1] = (g * 255.0).round().clamp(0.0, 255.0) as u8;
                out_pixel[2] = (b * 255.0).round().clamp(0.0, 255.0) as u8;
            });

        self.frame_count += 1;
        output_img
    }

    /// Retrieve the current motion mask as a grayscale visualization (useful for HUD/debugging)
    pub fn get_motion_mask_image(&self) -> Option<ImageBuffer<image::Luma<u8>, Vec<u8>>> {
        if self.width == 0 || self.height == 0 || self.motion_mask.is_empty() {
            return None;
        }
        let mut mask_img = ImageBuffer::new(self.width, self.height);
        for (idx, &m) in self.motion_mask.iter().enumerate() {
            let x = (idx as u32) % self.width;
            let y = (idx as u32) / self.width;
            mask_img.put_pixel(x, y, image::Luma([(m * 255.0) as u8]));
        }
        Some(mask_img)
    }
}
