//! Deterministic Physics-Based Edge-Preserving Spatial Filter for MCUs.
//!
//! Replaces runtime variance calculations and floating-point divisions with
//! physical Poisson-Gaussian noise thresholding and bit-shifting math.

use crate::fixed_math::abs_diff_u8;
use crate::lut::DeterministicNoiseLut;

/// Filters a row using 3x3 local neighborhood and deterministic noise thresholding.
///
/// Preserves sharp edges (text, silhouettes) while aggressively smoothing flat sensor static.
pub fn filter_line_deterministic_slice(
    top: &[u8],
    mid: &[u8],
    bot: &[u8],
    out: &mut [u8],
    noise_lut: &DeterministicNoiseLut,
) {
    let width = mid.len();
    if width < 3 || top.len() < width || bot.len() < width || out.len() < width {
        return;
    }

    // Handle first pixel (x = 0)
    out[0] = mid[0];

    // Main 3x3 sliding kernel
    for x in 1..(width - 1) {
        let center = mid[x];
        let threshold = noise_lut.noise_threshold(center);

        let mut sum = center as u16;
        let mut count = 1u16;

        // 8-connected neighbors
        let neighbors = [
            top[x - 1], top[x], top[x + 1],
            mid[x - 1],         mid[x + 1],
            bot[x - 1], bot[x], bot[x + 1],
        ];

        for &n in &neighbors {
            let diff = abs_diff_u8(n, center);
            // If delta is below the physical photon noise threshold, it is random sensor static:
            // include it in the smoothing average.
            // If delta exceeds threshold, it is a real optical edge: do not smooth across!
            if diff < threshold {
                sum += n as u16;
                count += 1;
            }
        }

        // Fast integer division / normalization
        out[x] = (sum / count) as u8;
    }

    // Handle last pixel
    out[width - 1] = mid[width - 1];
}

/// Fixed-width const generic variant of deterministic row filter.
#[inline(always)]
pub fn filter_line_deterministic<const WIDTH: usize>(
    top: &[u8; WIDTH],
    mid: &[u8; WIDTH],
    bot: &[u8; WIDTH],
    out: &mut [u8; WIDTH],
    noise_lut: &DeterministicNoiseLut,
) {
    filter_line_deterministic_slice(top, mid, bot, out, noise_lut);
}

/// Lightweight single-scale integer unsharp mask for local contrast enhancement.
/// Replaces heavy multi-scale Retinex with 2 bit-shifts and 1 subtraction per pixel.
#[inline(always)]
pub fn integer_unsharp_mask(original: u8, blurred: u8, boost_shift: u8) -> u8 {
    let orig = original as i16;
    let blur = blurred as i16;
    let detail = orig - blur;

    // detail * (1 + 1/2) or boost using bitshift
    let boosted = orig + (detail >> boost_shift);
    boosted.clamp(0, 255) as u8
}

