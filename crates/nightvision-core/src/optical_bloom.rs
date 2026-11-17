//! Contrast-Gated Physical Optical Bloom Injection.
//!
//! Simulates physical lens veiling glare and ocular intraocular scatter around extreme
//! light sources (e.g. headlights, windows, flares) without washing out bright portrait faces.
//!
//! Anti-Washout Contrast Gating:
//! - On backlit scenes (e.g. dark room around an overexposed window, Y_surround < 0.55),
//!   optical scatter is physically expected and perceived as extreme dynamic range. Full bloom injected.
//! - On high-key scenes (e.g. glare-lifted portrait face, Y_surround > 0.65), additive bloom
//!   would obliterate eyebrows, eyes, and beard. Bloom is gracefully gated to 0.0 via smoothstep.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::highlight_recovery::box_filter_2d;

/// Configuration for contrast-gated optical bloom.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BloomConfig {
    /// Luminance threshold above which pixels contribute to bloom (default: 0.92, ~235 ADU).
    pub threshold: f32,
    /// Multi-scale bloom intensity (default: 0.35).
    pub intensity: f32,
    /// Surround luminance threshold where bloom starts to be suppressed (default: 0.55).
    pub surround_low: f32,
    /// Surround luminance threshold where bloom is fully suppressed (default: 0.65).
    pub surround_high: f32,
    /// Radius for surround luminance estimation (default: 16).
    pub surround_radius: usize,
    /// Fine blur radius (default: 4).
    pub radius_fine: usize,
    /// Broad blur radius (default: 12).
    pub radius_broad: usize,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 0.92,
            intensity: 0.35,
            surround_low: 0.55,
            surround_high: 0.65,
            surround_radius: 16,
            radius_fine: 4,
            radius_broad: 12,
        }
    }
}

/// Scratch buffers for bloom processing.
#[derive(Default)]
pub struct BloomScratch {
    pub mask: Vec<f32>,
    pub fine_blur: Vec<f32>,
    pub broad_blur: Vec<f32>,
    pub surround: Vec<f32>,
    pub temp: Vec<f32>,
}

impl BloomScratch {
    pub fn ensure_capacity(&mut self, n: usize) {
        if self.mask.len() != n {
            self.mask.resize(n, 0.0);
            self.fine_blur.resize(n, 0.0);
            self.broad_blur.resize(n, 0.0);
            self.surround.resize(n, 0.0);
            self.temp.resize(n, 0.0);
        }
    }
}

#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Injects contrast-gated optical bloom into planar luminance.
pub fn inject_optical_bloom_planar(
    y_plane: &mut [f32],
    scratch: &mut BloomScratch,
    w: usize,
    h: usize,
    config: &BloomConfig,
) {
    let n = w * h;
    if n == 0 || config.intensity <= 0.0 {
        return;
    }
    scratch.ensure_capacity(n);

    let thresh = config.threshold;
    let inv_range = 1.0 / (1.0 - thresh).max(1e-4);

    // 1. Build soft highlight energy mask
    scratch.mask
        .par_iter_mut()
        .zip(y_plane.par_iter())
        .for_each(|(m, &y)| {
            if y > thresh {
                *m = ((y - thresh) * inv_range).clamp(0.0, 1.0);
            } else {
                *m = 0.0;
            }
        });

    // 2. Measure local surround luminance Y_surround with wide radius
    box_filter_2d(y_plane, &mut scratch.surround, &mut scratch.temp, w, h, config.surround_radius);

    // 3. Multi-scale blur of the highlight mask
    // Fine scale
    box_filter_2d(&scratch.mask, &mut scratch.fine_blur, &mut scratch.temp, w, h, config.radius_fine);
    // Broad scale
    box_filter_2d(&scratch.mask, &mut scratch.broad_blur, &mut scratch.temp, w, h, config.radius_broad);

    // 4. Contrast-Gated additive injection
    let intensity = config.intensity;
    let s_low = config.surround_low;
    let s_high = config.surround_high;

    let fine = &scratch.fine_blur;
    let broad = &scratch.broad_blur;
    let surround = &scratch.surround;

    y_plane
        .par_iter_mut()
        .zip(fine.par_iter())
        .zip(broad.par_iter())
        .zip(surround.par_iter())
        .for_each(|(((y, &f_val), &b_val), &s_val)| {
            // Contrast gate: 1.0 for dark surround (window), 0.0 for bright surround (portrait face)
            let gate = 1.0 - smoothstep(s_low, s_high, s_val);

            if gate > 0.001 {
                let bloom = 0.60 * f_val + 0.40 * b_val;
                *y = (*y + bloom * gate * intensity).clamp(0.0, 1.0);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_injected_in_dark_surround() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;

        // Dark room (0.1 ADU) with a bright center window (1.0 ADU)
        let mut y = vec![0.1f32; n];
        for yy in 12..20 {
            for xx in 12..20 {
                y[yy * w + xx] = 1.0;
            }
        }

        let mut scratch = BloomScratch::default();
        let cfg = BloomConfig::default();

        let y_before = y.clone();
        inject_optical_bloom_planar(&mut y, &mut scratch, w, h, &cfg);

        // Pixels adjacent to the window (e.g. x=10, y=16) must receive bloom
        let test_px = 16 * w + 10;
        assert!(
            y[test_px] > y_before[test_px],
            "Dark surround must receive optical bloom: before {}, after {}",
            y_before[test_px],
            y[test_px]
        );
    }

    #[test]
    fn test_bloom_suppressed_in_bright_surround() {
        let w = 32usize;
        let h = 32usize;
        let n = w * h;

        // High-key scene (bright portrait face: surround = 0.75, highlight = 1.0)
        let mut y = vec![0.75f32; n];
        for yy in 12..20 {
            for xx in 12..20 {
                y[yy * w + xx] = 1.0;
            }
        }

        let mut scratch = BloomScratch::default();
        let cfg = BloomConfig::default();

        let y_before = y.clone();
        inject_optical_bloom_planar(&mut y, &mut scratch, w, h, &cfg);

        // Pixels adjacent to highlight (e.g. x=10, y=16) must NOT be washed out (gate ≈ 0)
        let test_px = 16 * w + 10;
        let delta = (y[test_px] - y_before[test_px]).abs();
        assert!(
            delta < 1e-3,
            "Bright surround must suppress optical bloom (no face washout): delta = {}",
            delta
        );
    }

    #[test]
    fn test_midtone_far_from_mask_unaffected() {
        let w = 64usize;
        let h = 64usize;
        let n = w * h;

        let mut y = vec![0.3f32; n];
        // Small highlight at top-left
        for yy in 0..4 {
            for xx in 0..4 {
                y[yy * w + xx] = 1.0;
            }
        }

        let mut scratch = BloomScratch::default();
        let cfg = BloomConfig::default();

        let y_before = y.clone();
        inject_optical_bloom_planar(&mut y, &mut scratch, w, h, &cfg);

        // Pixel far away (e.g. x=50, y=50) must not shift
        let far_px = 50 * w + 50;
        assert_eq!(y[far_px], y_before[far_px]);
    }
}
