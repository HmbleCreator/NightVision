//! High-performance edge-preserving spatial denoising algorithms.
//!
//! Includes:
//! - Guided Image Filter: O(1) edge-preserving smoothing that flattens CMOS noise
//!   while preserving sharp object boundaries, text, and silhouettes.
//! - Chroma Despeckle: Separates luminance (Y) and chrominance (Cb, Cr) to eliminate
//!   severe color blotches characteristic of high-gain digital sensors without blurring edges.

use image::{ImageBuffer, Rgb, RgbImage};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for spatial denoising.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialDenoiseConfig {
    /// Spatial filter radius in pixels (typically 2 to 8).
    pub guided_radius: usize,
    /// Edge regularization parameter epsilon (typically 0.001 to 0.05).
    /// Larger epsilon produces stronger smoothing; smaller preserves finer subtle gradients.
    pub guided_epsilon: f32,
    /// Chroma noise reduction strength (0.0 = disabled, 1.0 = maximum).
    pub chroma_denoise_strength: f32,
    /// Number of guided filter iterations (typically 1 or 2).
    pub iterations: usize,
}

impl Default for SpatialDenoiseConfig {
    fn default() -> Self {
        Self {
            guided_radius: 1,
            guided_epsilon: 0.015,
            chroma_denoise_strength: 0.0,
            iterations: 1,
        }
    }
}

/// Applies separable 1D sliding box blur in strictly O(1) per pixel.
pub fn box_filter_2d(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }

    let mut temp = vec![0.0f32; w * h];

    // Horizontal pass (parallelized over rows)
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

    // Vertical pass: transpose, horizontal box blur, transpose back for strict O(1) efficiency
    let mut transposed = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            transposed[x * h + y] = temp[y * w + x];
        }
    }

    let mut filtered_transposed = vec![0.0f32; w * h];
    filtered_transposed
        .par_chunks_exact_mut(h)
        .zip(transposed.par_chunks_exact(h))
        .for_each(|(out_col, in_col)| {
            let mut sum = 0.0f32;
            let mut count = 0;

            for y in 0..r.min(h) {
                sum += in_col[y];
                count += 1;
            }

            for y in 0..h {
                if y + r < h {
                    sum += in_col[y + r];
                    count += 1;
                }
                if y > r {
                    sum -= in_col[y - r - 1];
                    count -= 1;
                }
                out_col[y] = sum / (count as f32);
            }
        });

    let mut dst = vec![0.0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            dst[y * w + x] = filtered_transposed[x * h + y];
        }
    }

    dst
}

/// Fast O(1) recursive IIR Gaussian filter (Young & van Vliet).
/// Constant execution time independent of sigma.
pub fn recursive_gaussian_blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.5 || w < 4 || h < 4 {
        return src.to_vec();
    }

    let q = if sigma >= 2.5 {
        0.98711 * sigma - 0.96330
    } else {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).max(0.0).sqrt()
    };

    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q * q + 0.422205 * q * q * q;
    let b1 = 2.44413 * q + 2.85619 * q * q + 1.26661 * q * q * q;
    let b2 = -(1.4281 * q * q + 1.26661 * q * q * q);
    let b3 = 0.422205 * q * q * q;

    let c1 = b1 / b0;
    let c2 = b2 / b0;
    let c3 = b3 / b0;
    let b = (1.0 - (c1 + c2 + c3)).max(0.0001);

    // Horizontal filter pass
    let mut temp = vec![0.0f32; w * h];
    temp.par_chunks_exact_mut(w)
        .zip(src.par_chunks_exact(w))
        .for_each(|(out_row, in_row)| {
            let mut fwd = vec![0.0f32; w];
            let mut y1 = in_row[0];
            let mut y2 = in_row[0];
            let mut y3 = in_row[0];
            for x in 0..w {
                let y = b * in_row[x] + c1 * y1 + c2 * y2 + c3 * y3;
                fwd[x] = y;
                y3 = y2;
                y2 = y1;
                y1 = y;
            }

            let mut y1 = fwd[w - 1];
            let mut y2 = fwd[w - 1];
            let mut y3 = fwd[w - 1];
            for x in (0..w).rev() {
                let y = b * fwd[x] + c1 * y1 + c2 * y2 + c3 * y3;
                out_row[x] = y;
                y3 = y2;
                y2 = y1;
                y1 = y;
            }
        });

    // Vertical filter pass (via transpose)
    let mut transposed = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            transposed[x * h + y] = temp[y * w + x];
        }
    }

    let mut filtered_transposed = vec![0.0f32; w * h];
    filtered_transposed
        .par_chunks_exact_mut(h)
        .zip(transposed.par_chunks_exact(h))
        .for_each(|(out_col, in_col)| {
            let mut fwd = vec![0.0f32; h];
            let mut y1 = in_col[0];
            let mut y2 = in_col[0];
            let mut y3 = in_col[0];
            for y in 0..h {
                let val = b * in_col[y] + c1 * y1 + c2 * y2 + c3 * y3;
                fwd[y] = val;
                y3 = y2;
                y2 = y1;
                y1 = val;
            }

            let mut y1 = fwd[h - 1];
            let mut y2 = fwd[h - 1];
            let mut y3 = fwd[h - 1];
            for y in (0..h).rev() {
                let val = b * fwd[y] + c1 * y1 + c2 * y2 + c3 * y3;
                out_col[y] = val;
                y3 = y2;
                y2 = y1;
                y1 = val;
            }
        });

    let mut dst = vec![0.0f32; w * h];
    for x in 0..w {
        for y in 0..h {
            dst[y * w + x] = filtered_transposed[x * h + y];
        }
    }

    dst
}

/// Guided Image Filter (He et al.) on a single channel.
/// Guidance image `guide`, input channel `p`, radius `r`, regularization `eps`.
pub fn guided_filter_channel(guide: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let mean_i = box_filter_2d(guide, w, h, r);
    let mean_p = box_filter_2d(p, w, h, r);

    let ip: Vec<f32> = guide.iter().zip(p.iter()).map(|(&i, &v)| i * v).collect();
    let mean_ip = box_filter_2d(&ip, w, h, r);

    let ii: Vec<f32> = guide.iter().map(|&i| i * i).collect();
    let mean_ii = box_filter_2d(&ii, w, h, r);

    let mut a = vec![0.0f32; w * h];
    let mut b = vec![0.0f32; w * h];

    a.par_iter_mut()
        .zip(b.par_iter_mut())
        .zip(mean_i.par_iter())
        .zip(mean_p.par_iter())
        .zip(mean_ip.par_iter())
        .zip(mean_ii.par_iter())
        .for_each(|(((((out_a, out_b), &m_i), &m_p), &m_ip), &m_ii)| {
            let cov_ip = m_ip - m_i * m_p;
            let var_i = m_ii - m_i * m_i;
            let coeff_a = cov_ip / (var_i + eps);
            let coeff_b = m_p - coeff_a * m_i;
            *out_a = coeff_a;
            *out_b = coeff_b;
        });

    let mean_a = box_filter_2d(&a, w, h, r);
    let mean_b = box_filter_2d(&b, w, h, r);

    let mut q = vec![0.0f32; w * h];
    q.par_iter_mut()
        .zip(mean_a.par_iter())
        .zip(guide.par_iter())
        .zip(mean_b.par_iter())
        .for_each(|(((out_q, &ma), &i_val), &mb)| {
            *out_q = (ma * i_val + mb).clamp(0.0, 1.0);
        });

    q
}

/// Denoises an RGB image using edge-preserving Guided Filtering and Chroma Despeckle.
pub fn denoise_image(image: &RgbImage, config: &SpatialDenoiseConfig) -> RgbImage {
    let (width, height) = image.dimensions();
    let (w, h) = (width as usize, height as usize);
    let n = w * h;

    if n == 0 {
        return image.clone();
    }

    // Convert RGB to normalized f32 YCbCr
    let mut y_chan = vec![0.0f32; n];
    let mut cb_chan = vec![0.0f32; n];
    let mut cr_chan = vec![0.0f32; n];

    for (idx, p) in image.pixels().enumerate() {
        let r = (p[0] as f32) / 255.0;
        let g = (p[1] as f32) / 255.0;
        let b = (p[2] as f32) / 255.0;

        // ITU-R BT.601 conversion
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5;
        let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5;

        y_chan[idx] = y;
        cb_chan[idx] = cb;
        cr_chan[idx] = cr;
    }

    // Denoise luminance channel using self-guided filter (preserving edges)
    let mut filtered_y = y_chan.clone();
    for _ in 0..config.iterations.max(1) {
        filtered_y = guided_filter_channel(
            &filtered_y,
            &filtered_y,
            w,
            h,
            config.guided_radius,
            config.guided_epsilon,
        );
    }

    // Denoise chrominance channels with guided filter or strong smoothing
    let chroma_r = (config.guided_radius * 2).max(4);
    let chroma_eps = config.guided_epsilon * 5.0;

    let filtered_cb = if config.chroma_denoise_strength > 0.0 {
        let smoothed = guided_filter_channel(&filtered_y, &cb_chan, w, h, chroma_r, chroma_eps);
        let strength = config.chroma_denoise_strength;
        cb_chan
            .iter()
            .zip(smoothed.iter())
            .map(|(&orig, &sm)| orig * (1.0 - strength) + sm * strength)
            .collect()
    } else {
        cb_chan
    };

    let filtered_cr = if config.chroma_denoise_strength > 0.0 {
        let smoothed = guided_filter_channel(&filtered_y, &cr_chan, w, h, chroma_r, chroma_eps);
        let strength = config.chroma_denoise_strength;
        cr_chan
            .iter()
            .zip(smoothed.iter())
            .map(|(&orig, &sm)| orig * (1.0 - strength) + sm * strength)
            .collect()
    } else {
        cr_chan
    };

    // Convert back from YCbCr to RGB
    let mut output = ImageBuffer::new(width, height);
    for idx in 0..n {
        let y = filtered_y[idx];
        let cb = filtered_cb[idx] - 0.5;
        let cr = filtered_cr[idx] - 0.5;

        let r = y + 1.402 * cr;
        let g = y - 0.344136 * cb - 0.714136 * cr;
        let b = y + 1.772 * cb;

        let x = (idx % w) as u32;
        let y_coord = (idx / w) as u32;

        let r_u8 = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
        let g_u8 = (g.clamp(0.0, 1.0) * 255.0).round() as u8;
        let b_u8 = (b.clamp(0.0, 1.0) * 255.0).round() as u8;

        output.put_pixel(x, y_coord, Rgb([r_u8, g_u8, b_u8]));
    }

    output
}

