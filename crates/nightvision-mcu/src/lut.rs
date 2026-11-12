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
        let mut table = [0u8; 256];
        let inv_gamma = 1.0 / gamma.clamp(0.2, 2.0);

        for i in 0..256 {
            let norm = (i as f32) / 255.0;

            // 1. Linear gain amplification
            let boosted = norm * gain;

            // 2. Adaptive gamma expansion (lifts deep shadow details)
            let gamma_lift = boosted.powf(inv_gamma);

            // 3. Digital Auto-Gating: Soft-knee Reinhard compression
            // Clamps headlights / muzzle flashes smoothly without blowing out highlights
            let compressed = if norm > auto_gating_threshold {
                let flare_damping = 1.0 / (1.0 + (norm - auto_gating_threshold) * 6.0);
                (gamma_lift * flare_damping) / (1.0 + gamma_lift * 0.4)
            } else {
                gamma_lift / (1.0 + gamma_lift * 0.4)
            };

            table[i] = (compressed * 255.0).round().clamp(0.0, 255.0) as u8;
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

    /// Tactical White Phosphor (P45) with cool high-contrast response.
    pub fn p45_white() -> Self {
        let mut lut_r = [0u8; 256];
        let mut lut_g = [0u8; 256];
        let mut lut_b = [0u8; 256];

        for i in 0..256 {
            let y = (i as f32) / 255.0;
            let s_curve = (y * 1.06 - 0.03).clamp(0.0, 1.0) * 255.0;

            lut_r[i] = (s_curve * 0.95).round() as u8;
            lut_g[i] = (s_curve * 0.98).round() as u8;
            lut_b[i] = (s_curve * 1.02).round().clamp(0.0, 255.0) as u8;
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
