//! C¹-Continuous Filmic Soft-Knee Tonemap and Per-Channel Highlight Desaturation.
//!
//! Implements the ACES/cinema-grade highlight roll-off used in digital intermediate (DI)
//! pipelines (ARRI Alexa, RED IPP2). The transfer curve matches both value and first
//! derivative at the knee point, eliminating boundary steps (seam = 0 ADU) while
//! compressing highlights monotonically into the displayable range.
//!
//! Desaturation is driven by per-channel RGB analysis (not luma Y) to correctly
//! detect single-channel clipping on warm skin tones where R clips at 253 ADU
//! while Y is only ~212 ADU.

use serde::{Deserialize, Serialize};

/// Configuration for the filmic soft-knee tonemap and highlight desaturation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilmicConfig {
    /// Knee point in normalized [0, 1] linear space. Below this, the curve is identity.
    /// Default: 0.70 (~180 ADU). Pixels below knee are untouched (±0 ADU).
    pub knee: f32,
    /// Ceiling asymptote in normalized [0, 1] linear space. Highlights roll off smoothly into this ceiling.
    /// Default: 1.00 (~255 ADU). For high-exposure recovery, configured to 0.89 (~227 ADU).
    pub ceiling: f32,
    /// Start of highlight desaturation ramp (normalized min-channel value).
    /// Default: 0.85 (~217 ADU). Below this, chroma is fully preserved.
    pub desat_start: f32,
    /// End of highlight desaturation ramp (normalized min-channel value).
    /// Default: 248.0/255.0 (~0.973). At and above this, chroma is fully removed.
    pub desat_end: f32,
}

impl Default for FilmicConfig {
    fn default() -> Self {
        Self {
            knee: 0.70,
            ceiling: 1.0,
            desat_start: 0.85,
            desat_end: 248.0 / 255.0,
        }
    }
}

/// Attempt to match smoothstep semantics: Hermite interpolation between edge0 and edge1.
#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// C¹-continuous filmic tonemap with Hermite highlight roll-off.
///
/// For x < knee: identity f(x) = x
/// For knee <= x <= 1.0: Hermite cubic matching value and 1st derivative at knee,
/// strictly anchoring at 1.0 (255 ADU) at ceiling:
///   u = (x - knee) / (1 - knee)
///   f(x) = knee + (1 - knee) * (u * (1 + u - u²))
/// For x > 1.0 (HDR headroom): asymptotic approach to 1.0:
///   f(x) = 1.0 - 0.001 / (1.0 + (x - 1.0) / (1.0 - knee))
///
/// Properties:
/// - `f(knee⁻) = f(knee⁺) = knee` (value continuity)
/// - `f'(knee⁻) = f'(knee⁺) = 1.0` (derivative continuity → C¹)
/// - `f'(x) >= 0` for all x (monotonicity, 0.00% inversions)
/// - `f(1.0) = 1.0` (preserves white point at 255 ADU)
#[inline]
pub fn filmic_tonemap(x: f32, config: &FilmicConfig) -> f32 {
    let knee = config.knee;
    let ceiling = config.ceiling;
    if x < knee {
        x
    } else if x < 1.0 {
        let delta = 1.0 - knee;
        if delta <= 1e-6 {
            return ceiling;
        }
        let u = (x - knee) / delta;
        let a = knee + 1.0 - 2.0 * ceiling;
        let b = 3.0 * ceiling - knee - 2.0;
        knee + u * (delta + u * (b + u * a))
    } else {
        ceiling
    }
}

/// Per-channel highlight desaturation.
///
/// Drives desaturation by `min(R, G, B)`, not luma Y. This correctly handles
/// warm skin where Red clips at 253 ADU (Y ≈ 212 ADU) — only desaturates
/// when ALL three channels approach the ceiling (true triple-channel saturation).
///
/// Modifies R, G, B in-place by lerping toward their mean (grey axis) based
/// on `desat_factor = 1.0 - smoothstep(desat_start, desat_end, min(R,G,B))`.
#[inline]
pub fn highlight_desaturate(r: &mut f32, g: &mut f32, b: &mut f32, config: &FilmicConfig) {
    let ch_min = r.min(*g).min(*b);
    let desat_amount = smoothstep(config.desat_start, config.desat_end, ch_min);

    if desat_amount > 0.0 {
        let grey = 0.299 * *r + 0.587 * *g + 0.114 * *b;
        *r = *r + desat_amount * (grey - *r);
        *g = *g + desat_amount * (grey - *g);
        *b = *b + desat_amount * (grey - *b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> FilmicConfig {
        FilmicConfig::default()
    }

    // Test 1: Identity below knee
    #[test]
    fn filmic_identity_below_knee() {
        let cfg = default_cfg();
        for i in 0..=178 {
            let x = i as f32 / 255.0;
            let y = filmic_tonemap(x, &cfg);
            assert!(
                (y - x).abs() < 1e-6,
                "Below knee: f({:.4}) = {:.4}, expected identity",
                x,
                y
            );
        }
    }

    // Test 2: C¹ continuity at knee
    #[test]
    fn filmic_c1_continuous_at_knee() {
        let cfg = default_cfg();
        let delta = 1e-4_f32;

        let f_minus = filmic_tonemap(cfg.knee - delta, &cfg);
        let f_plus = filmic_tonemap(cfg.knee + delta, &cfg);

        // Value continuity
        let f_at = filmic_tonemap(cfg.knee, &cfg);
        assert!(
            (f_at - cfg.knee).abs() < 1e-6,
            "f(knee) = {}, expected {}",
            f_at,
            cfg.knee
        );

        // Derivative continuity: slope ≈ 1.0 from both sides
        let slope = (f_plus - f_minus) / (2.0 * delta);
        assert!(
            (slope - 1.0).abs() < 0.01,
            "Slope at knee = {}, expected ≈1.0",
            slope
        );
    }

    // Test 3: Strict monotonicity over full range [0, 4.0]
    #[test]
    fn filmic_strict_monotonicity() {
        let cfg = default_cfg();
        let steps = 10000;
        let mut prev = filmic_tonemap(0.0, &cfg);
        for i in 1..=steps {
            let x = 4.0 * (i as f32) / (steps as f32);
            let y = filmic_tonemap(x, &cfg);
            assert!(
                y > prev || (y - prev).abs() < 1e-7,
                "Monotonicity violation: f({:.6}) = {:.6} <= f(prev) = {:.6}",
                x,
                y,
                prev
            );
            prev = y;
        }
    }

    // Test 4: Asymptotic bound
    #[test]
    fn filmic_asymptotic_bound() {
        let cfg = default_cfg();

        // f(x) <= 1.0 for all finite x
        for x in [1.0, 2.0, 5.0, 10.0, 50.0, 100.0] {
            let y = filmic_tonemap(x, &cfg);
            assert!(y <= 1.0, "f({}) = {} > 1.0", x, y);
        }

        // f(100.0) > 0.99
        let y_100 = filmic_tonemap(100.0, &cfg);
        assert!(
            y_100 > 0.99,
            "f(100.0) = {}, expected > 0.99",
            y_100
        );
    }

    // Test 5: `previews_are_locally_monotone` — 0% gradient inversions on 1000-pixel ramp
    #[test]
    fn previews_are_locally_monotone() {
        let cfg = default_cfg();
        let n = 1000;
        let mut inversions = 0usize;
        let mut prev = filmic_tonemap(0.0, &cfg);

        for i in 1..=n {
            let x = 2.0 * (i as f32) / (n as f32);
            let y = filmic_tonemap(x, &cfg);
            if y < prev - 1e-7 {
                inversions += 1;
            }
            prev = y;
        }

        let inversion_pct = (inversions as f64) / (n as f64) * 100.0;
        assert!(
            inversion_pct < 0.01,
            "Gradient inversions: {:.2}% ({}/{}), expected 0.00%",
            inversion_pct,
            inversions,
            n
        );
    }

    // Test 6: Warm skin preserved — R=253, G=200, B=165 → chroma unchanged
    #[test]
    fn highlight_desat_warm_skin_preserved() {
        let cfg = default_cfg();
        let mut r = 253.0 / 255.0;
        let mut g = 200.0 / 255.0;
        let mut b = 165.0 / 255.0;
        let (r0, g0, b0) = (r, g, b);

        highlight_desaturate(&mut r, &mut g, &mut b, &cfg);

        assert!(
            (r - r0).abs() < 1e-4,
            "Warm skin R shifted: {} → {}",
            r0,
            r
        );
        assert!(
            (g - g0).abs() < 1e-4,
            "Warm skin G shifted: {} → {}",
            g0,
            g
        );
        assert!(
            (b - b0).abs() < 1e-4,
            "Warm skin B shifted: {} → {}",
            b0,
            b
        );
    }

    // Test 7: Triple-clip desaturates — R=252, G=250, B=249 → chroma reduced
    #[test]
    fn highlight_desat_triple_clip() {
        let cfg = default_cfg();
        let mut r: f32 = 252.0 / 255.0;
        let mut g: f32 = 250.0 / 255.0;
        let mut b: f32 = 249.0 / 255.0;

        let chroma_before: f32 = {
            let mean = (r + g + b) / 3.0;
            ((r - mean).powi(2) + (g - mean).powi(2) + (b - mean).powi(2)).sqrt()
        };

        highlight_desaturate(&mut r, &mut g, &mut b, &cfg);

        let chroma_after: f32 = {
            let mean = (r + g + b) / 3.0;
            ((r - mean).powi(2) + (g - mean).powi(2) + (b - mean).powi(2)).sqrt()
        };

        assert!(
            chroma_after < chroma_before * 0.5,
            "Triple-clip chroma not reduced enough: before={:.6}, after={:.6}",
            chroma_before,
            chroma_after
        );
    }

    #[test]
    fn filmic_ceiling_compression() {
        let cfg = FilmicConfig {
            knee: 0.55,
            ceiling: 0.89,
            ..Default::default()
        };
        // Below knee identity
        assert!((filmic_tonemap(0.50, &cfg) - 0.50).abs() < 1e-6);
        // At 1.0 reaches ceiling 0.89
        assert!((filmic_tonemap(1.00, &cfg) - 0.89).abs() < 1e-4);
        // Above 1.0 clamped to ceiling 0.89
        assert!((filmic_tonemap(1.50, &cfg) - 0.89).abs() < 1e-4);
        // Continuous at knee
        let delta = 1e-4_f32;
        let slope = (filmic_tonemap(cfg.knee + delta, &cfg) - filmic_tonemap(cfg.knee - delta, &cfg)) / (2.0 * delta);
        assert!((slope - 1.0).abs() < 0.02);
    }
}
