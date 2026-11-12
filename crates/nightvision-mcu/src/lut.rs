//! Precomputed Look-Up Tables (LUTs) for zero-latency MCU evaluation.
//!
//! Replaces expensive floating-point power curves, logarithms, and division
//! with 1-cycle L1-cache array lookups.

use crate::fixed_math::clamp_to_u8;

/// Unified 8-bit Tone Mapping & Auto-Gating Look-Up Table (256 bytes).
/// Pre-computes: Exposure Gain -> Gamma Lift -> Reinhard Auto-Gating Compression.
#[derive(Debug, Clone, Copy)]
pub struct AutoGatingLut {
    pub table: [u8; 256],
}

impl AutoGatingLut {
    /// Generates LUT based on sensor gain, gamma curve, and flare auto-gating threshold.
    pub fn new(gain: f32, gamma: f32, auto_gating_threshold: f32) -> Self {
        Self::new_bidirectional(gain, gamma, auto_gating_threshold, 16.0, 128.0, 0.0)
    }

    /// Generates an adaptive LUT with dynamic shadow-knee boost for ultra-dark scenes (< 8.0 ADU).
    /// Uses an adaptive Hermite expansion knee on the lowest shadow codes (0..32 ADU) to reveal
    /// photon-starved objects (such as in scenes with mean luma < 6.0) without blowing out midtones.
    pub fn new_with_knee(
        gain: f32,
        gamma: f32,
        auto_gating_threshold: f32,
        mean_luma: f32,
    ) -> Self {
        Self::new_bidirectional(gain, gamma, auto_gating_threshold, mean_luma, 128.0, 0.0)
    }

    /// Generates a bidirectional HDR LUT supporting:
    /// 1. Ultra-dark scenes (< 8.0 ADU): Hermite upward shadow-knee expansion on 0..32 ADU.
    /// 2. Normal scenes (8.0 to 170.0 ADU): Balanced linear gain + Reinhard soft-knee auto-gating.
    /// 3. High-exposure / Glare scenes (> 170.0 ADU or p95 > 225.0 ADU or sat_ratio > 0.08):
    ///    Inverse Hermite highlight-compression shoulder on upper codes (> 140 ADU) that
    ///    pulls washed-out midtones down, recovers highlight contrast, and prevents hard white clipping.
    pub fn new_bidirectional(
        gain: f32,
        gamma: f32,
        auto_gating_threshold: f32,
        mean_luma: f32,
        p95_luma: f32,
        sat_ratio: f32,
    ) -> Self {
        let mut table = [0u8; 256];
        let is_ultra_dark = mean_luma < 8.0;
        let is_high_exposure = mean_luma > 170.0 || p95_luma > 225.0 || sat_ratio > 0.08;

        if is_ultra_dark {
            let gamma_exp = if gamma > 0.0 && gamma < 1.0 {
                gamma
            } else {
                1.0 / gamma.clamp(0.1, 2.0)
            };
            let darkness = ((8.0 - mean_luma) / 7.5).clamp(0.0, 1.0);
            let shadow_boost = 1.0 + darkness * 1.6; // Up to 2.6x extra gain on deep shadows

            for i in 0..256 {
                let norm = (i as f32) / 255.0;
                let knee_multiplier = if i < 32 {
                    let t = (i as f32) / 32.0;
                    let smooth_t = t * t * (3.0 - 2.0 * t);
                    shadow_boost * (1.0 - smooth_t) + 1.0 * smooth_t
                } else {
                    1.0
                };

                let boosted = norm * gain * knee_multiplier;
                let gamma_lift = boosted.powf(gamma_exp);
                let compressed = if norm > auto_gating_threshold {
                    let flare_damping = 1.0 / (1.0 + (norm - auto_gating_threshold) * 6.0);
                    (gamma_lift * flare_damping) / (1.0 + gamma_lift * 0.4)
                } else {
                    gamma_lift / (1.0 + gamma_lift * 0.4)
                };
                table[i] = (compressed * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        } else if is_high_exposure {
            // High exposure / Glare regime:
            // 1. Contrast expansion in midtones (gamma > 1.0)
            // 2. Concave power highlight compression on upper codes (strictly anchoring f(255) = 255)
            let shoulder_start = 140.0 / 255.0; // ~140 ADU
            for i in 0..256 {
                let norm = (i as f32) / 255.0;
                let val = if norm < shoulder_start {
                    shoulder_start * (norm / shoulder_start).powf(1.15)
                } else {
                    let t = (norm - shoulder_start) / (1.0 - shoulder_start);
                    shoulder_start + (1.0 - shoulder_start) * t.powf(2.2)
                };
                table[i] = (val * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        } else {
            let gamma_exp = if gamma > 0.0 && gamma < 1.0 {
                gamma
            } else {
                1.0 / gamma.clamp(0.1, 2.0)
            };
            for i in 0..256 {
                let norm = (i as f32) / 255.0;
                let boosted = norm * gain;
                let gamma_lift = boosted.powf(gamma_exp);
                let compressed = if norm > auto_gating_threshold {
                    let flare_damping = 1.0 / (1.0 + (norm - auto_gating_threshold) * 6.0);
                    (gamma_lift * flare_damping) / (1.0 + gamma_lift * 0.4)
                } else {
                    gamma_lift / (1.0 + gamma_lift * 0.4)
                };
                table[i] = (compressed * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }

        Self { table }
    }

    #[inline(always)]
    pub fn lookup(&self, val: u8) -> u8 {
        self.table[val as usize]
    }
}

/// Tactical Display Phosphor Look-Up Table (3 x 256 = 768 bytes).
/// Converts 8-bit enhanced luminance into authentic military phosphor RGB without floating-point math.
#[derive(Debug, Clone, Copy)]
pub struct PhosphorLut {
    pub lut_r: [u8; 256],
    pub lut_g: [u8; 256],
    pub lut_b: [u8; 256],
}

impl PhosphorLut {
    /// Authentic P43 Military Green Phosphor (545 nm emission peak).
    pub fn p43_green() -> Self {
        let mut lut_r = [0u8; 256];
        let mut lut_g = [0u8; 256];
        let mut lut_b = [0u8; 256];

        for i in 0..256 {
            let y = (i as f32) / 255.0;
            let val = (y.powf(0.88) * 255.0).clamp(0.0, 255.0);

            lut_r[i] = (val * 0.22).round() as u8;
            lut_g[i] = (val * 1.00).round() as u8;
            lut_b[i] = (val * 0.30).round() as u8;
        }

        Self { lut_r, lut_g, lut_b }
    }

    /// Tactical White Phosphor (P45) with cool high-contrast response converging cleanly to neutral white at peak.
    pub fn p45_white() -> Self {
        let mut lut_r = [0u8; 256];
        let mut lut_g = [0u8; 256];
        let mut lut_b = [0u8; 256];

        for i in 0..256 {
            let y = (i as f32) / 255.0;
            let val = y * 255.0;

            let highlight_desat = y.powf(4.0);
            let r_scale = 0.96 * (1.0 - highlight_desat) + 1.0 * highlight_desat;
            let g_scale = 0.98 * (1.0 - highlight_desat) + 1.0 * highlight_desat;
            let b_scale = 1.02 * (1.0 - highlight_desat) + 1.0 * highlight_desat;

            lut_r[i] = (val * r_scale).round().clamp(0.0, 255.0) as u8;
            lut_g[i] = (val * g_scale).round().clamp(0.0, 255.0) as u8;
            lut_b[i] = (val * b_scale).round().clamp(0.0, 255.0) as u8;
        }

        Self { lut_r, lut_g, lut_b }
    }

    /// High-contrast tactical reconnaissance amber phosphor.
    pub fn recon_amber() -> Self {
        let mut lut_r = [0u8; 256];
        let mut lut_g = [0u8; 256];
        let mut lut_b = [0u8; 256];

        for i in 0..256 {
            let val = (i as f32) * 1.02;
            lut_r[i] = (val * 1.00).round().clamp(0.0, 255.0) as u8;
            lut_g[i] = (val * 0.82).round().clamp(0.0, 255.0) as u8;
            lut_b[i] = (val * 0.38).round().clamp(0.0, 255.0) as u8;
        }

        Self { lut_r, lut_g, lut_b }
    }

    #[inline(always)]
    pub fn map_rgb(&self, luma: u8) -> [u8; 3] {
        let idx = luma as usize;
        [self.lut_r[idx], self.lut_g[idx], self.lut_b[idx]]
    }
}

/// Deterministic Physics Noise Profile LUT (256 bytes).
/// Pre-computes the physical photon noise threshold tau(Y) for every luminance value:
///   tau(Y) = k * sqrt(alpha * Y + read_noise^2)
/// Allows 1-cycle edge detection without calculating local variances at runtime!
#[derive(Debug, Clone, Copy)]
pub struct DeterministicNoiseLut {
    pub threshold_table: [u8; 256],
}

impl DeterministicNoiseLut {
    pub fn new(conversion_gain: f32, read_noise_electrons: f32, k_factor: f32) -> Self {
        let mut threshold_table = [0u8; 256];
        let read_noise_sq = read_noise_electrons * read_noise_electrons;

        for i in 0..256 {
            let luma = i as f32;
            // Poisson shot noise variance (proportional to signal) + Gaussian read noise variance
            let noise_variance = conversion_gain * luma + read_noise_sq;
            let noise_std = noise_variance.sqrt();
            let threshold = (k_factor * noise_std).round() as i32;
            threshold_table[i] = clamp_to_u8(threshold.max(2));
        }

        Self { threshold_table }
    }

    #[inline(always)]
    pub fn noise_threshold(&self, luma: u8) -> u8 {
        self.threshold_table[luma as usize]
    }
}

