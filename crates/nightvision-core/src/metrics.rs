//! Quantitative evaluation metrics for digital night vision performance.
//!
//! Provides mathematically rigorous metrics:
//! - PSNR (Peak Signal-to-Noise Ratio)
//! - SSIM (Structural Similarity Index)
//! - SNR (Signal-to-Noise Ratio in dB)
//! - CII (Contrast Improvement Index)

use image::RgbImage;

/// Computes Peak Signal-to-Noise Ratio (PSNR) in dB between a reference image and test image.
pub fn calculate_psnr(reference: &RgbImage, test: &RgbImage) -> f64 {
    let (w1, h1) = reference.dimensions();
    let (w2, h2) = test.dimensions();
    if w1 != w2 || h1 != h2 || w1 == 0 || h1 == 0 {
        return 0.0;
    }

    let mut sum_sq_err = 0.0f64;
    let total_samples = (w1 * h1 * 3) as f64;

    for (p1, p2) in reference.pixels().zip(test.pixels()) {
        for c in 0..3 {
            let diff = (p1[c] as f64) - (p2[c] as f64);
            sum_sq_err += diff * diff;
        }
    }

    let mse = sum_sq_err / total_samples;
    if mse <= 1e-10 {
        return 100.0; // Identical images
    }

    20.0 * (255.0 / mse.sqrt()).log10()
}

/// Computes Structural Similarity Index (SSIM) on luminance channel.
pub fn calculate_ssim(img1: &RgbImage, img2: &RgbImage) -> f64 {
    let (w1, h1) = img1.dimensions();
    let (w2, h2) = img2.dimensions();
    if w1 != w2 || h1 != h2 || w1 == 0 || h1 == 0 {
        return 0.0;
    }

    let n = (w1 * h1) as f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;

    let mut luma1 = Vec::with_capacity(w1 as usize * h1 as usize);
    let mut luma2 = Vec::with_capacity(w1 as usize * h1 as usize);

    for (p1, p2) in img1.pixels().zip(img2.pixels()) {
        let y1 = 0.299 * (p1[0] as f64) + 0.587 * (p1[1] as f64) + 0.114 * (p1[2] as f64);
        let y2 = 0.299 * (p2[0] as f64) + 0.587 * (p2[1] as f64) + 0.114 * (p2[2] as f64);
        sum1 += y1;
        sum2 += y2;
        luma1.push(y1);
        luma2.push(y2);
    }

    let mu1 = sum1 / n;
    let mu2 = sum2 / n;

    let mut var1 = 0.0f64;
    let mut var2 = 0.0f64;
    let mut cov12 = 0.0f64;

    for (y1, y2) in luma1.iter().zip(luma2.iter()) {
        let diff1 = y1 - mu1;
        let diff2 = y2 - mu2;
        var1 += diff1 * diff1;
        var2 += diff2 * diff2;
        cov12 += diff1 * diff2;
    }

    var1 /= n;
    var2 /= n;
    cov12 /= n;

    let c1 = (0.01f64 * 255.0).powi(2);
    let c2 = (0.03f64 * 255.0).powi(2);

    let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * cov12 + c2);
    let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (var1 + var2 + c2);

    if denominator == 0.0 {
        1.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

/// Computes Signal-to-Noise Ratio (SNR) in decibels from mean signal and noise standard deviation.
pub fn calculate_snr_db(signal_mean: f64, noise_std: f64) -> f64 {
    if noise_std <= 1e-12 {
        100.0
    } else if signal_mean <= 1e-12 {
        -50.0
    } else {
        20.0 * (signal_mean / noise_std).log10()
    }
}
