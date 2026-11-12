//! Tactical display modes and color rendering for NightVision.
//!
//! Directly tackles limitation #3 (Poor Image Quality / Monochrome) by offering:
//! - Full-Color Digital Night Vision: Restores natural RGB color even in deep twilight.
//! - Tactical White Phosphor (P45): Crisp high-contrast monochrome preferred in modern SOF units.
//! - Tactical Green Phosphor (P43): Classic military Gen 3 phosphor simulation with glow.
//! - High-Contrast Recon: High-definition silhouette and text outline extraction.
//! - Thermal Fusion: Pseudo-thermal spectral colormap (ironbow) fused with optical contours.

use image::{ImageBuffer, Rgb, RgbImage};
use serde::{Deserialize, Serialize};

/// Display mode selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisplayMode {
    /// Full natural color digital night vision (RGB enhancement)
    FullColor,
    /// Modern military White Phosphor (P45) simulation
    TacticalWhitePhosphor,
    /// Traditional military Gen 3 Green Phosphor (P43) simulation
    TacticalGreenPhosphor,
    /// Edge-enhanced tactical reconnaissance mode (text/silhouette highlight)
    HighContrastRecon,
    /// Pseudo-thermal optical fusion (ironbow colormap)
    ThermalFusion,
}

impl DisplayMode {
    pub fn name(&self) -> &'static str {
        match *self {
            DisplayMode::FullColor => "Full-Color Night Vision",
            DisplayMode::TacticalWhitePhosphor => "Tactical White Phosphor (P45)",
            DisplayMode::TacticalGreenPhosphor => "Tactical Green Phosphor (P43)",
            DisplayMode::HighContrastRecon => "High-Contrast Recon",
            DisplayMode::ThermalFusion => "Thermal Fusion Simulation",
        }
    }
}

/// Renders the enhanced image in the specified tactical display mode.
pub fn render_mode(img: &RgbImage, mode: DisplayMode, saturation_boost: f32) -> RgbImage {
    let (width, height) = img.dimensions();
    let mut out = ImageBuffer::new(width, height);

    match mode {
        DisplayMode::FullColor => {
            // Boost saturation and preserve natural colors
            let sat = saturation_boost.clamp(0.5, 3.0);
            for (p, o) in img.pixels().zip(out.pixels_mut()) {
                let r = p[0] as f32;
                let g = p[1] as f32;
                let b = p[2] as f32;
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;

                let r_sat = gray + (r - gray) * sat;
                let g_sat = gray + (g - gray) * sat;
                let b_sat = gray + (b - gray) * sat;

                *o = Rgb([
                    r_sat.round().clamp(0.0, 255.0) as u8,
                    g_sat.round().clamp(0.0, 255.0) as u8,
                    b_sat.round().clamp(0.0, 255.0) as u8,
                ]);
            }
        }

        DisplayMode::TacticalWhitePhosphor => {
            // White phosphor P45 has a cool, slightly blue-tinted silver phosphor response
            // Luminance emphasized for maximum contrast and text legibility
            for (p, o) in img.pixels().zip(out.pixels_mut()) {
                let y = (0.299 * (p[0] as f32) + 0.587 * (p[1] as f32) + 0.114 * (p[2] as f32))
                    / 255.0;

                // Subtle S-curve for punchy tactical contrast
                let y_curve = (y * 1.08 - 0.04).clamp(0.0, 1.0);
                let val = y_curve * 255.0;

                // P45 white phosphor palette (slight cool-white tint: R: 95%, G: 98%, B: 100%)
                let r = (val * 0.95).round().clamp(0.0, 255.0) as u8;
                let g = (val * 0.98).round().clamp(0.0, 255.0) as u8;
                let b = (val * 1.02).round().clamp(0.0, 255.0) as u8;

                *o = Rgb([r, g, b]);
            }
        }

        DisplayMode::TacticalGreenPhosphor => {
            // Classic P43 military phosphor: peak emission at 545 nm (emerald green)
            for (p, o) in img.pixels().zip(out.pixels_mut()) {
                let y = (0.299 * (p[0] as f32) + 0.587 * (p[1] as f32) + 0.114 * (p[2] as f32))
                    / 255.0;

                // Phosphor response curve
                let val = (y.powf(0.85) * 255.0).clamp(0.0, 255.0);

                // Authentic P43 spectral mapping
                let r = (val * 0.22).round() as u8;
                let g = (val * 1.00).round() as u8;
                let b = (val * 0.32).round() as u8;

                *o = Rgb([r, g, b]);
            }
        }

        DisplayMode::HighContrastRecon => {
            // Edge-boosted reconnaissance mode (highlights text and outlines)
            let (w, h) = (width as usize, height as usize);
            let mut luma = vec![0.0f32; w * h];
            for (idx, p) in img.pixels().enumerate() {
                luma[idx] =
                    (0.299 * (p[0] as f32) + 0.587 * (p[1] as f32) + 0.114 * (p[2] as f32)) / 255.0;
            }

            for y in 0..h {
                for x in 0..w {
                    let center = luma[y * w + x];
                    // Simple 3x3 Laplacian edge kernel
                    let mut edge = 0.0f32;
                    if x > 0 && x + 1 < w && y > 0 && y + 1 < h {
                        let top = luma[(y - 1) * w + x];
                        let bottom = luma[(y + 1) * w + x];
                        let left = luma[y * w + (x - 1)];
                        let right = luma[y * w + (x + 1)];
                        edge = (4.0 * center - top - bottom - left - right).abs();
                    }

                    let val = (center * 0.85 + edge * 1.2).clamp(0.0, 1.0) * 255.0;
                    // Crisp amber/gold recon phosphor
                    let r = (val * 1.00).round().clamp(0.0, 255.0) as u8;
                    let g = (val * 0.82).round().clamp(0.0, 255.0) as u8;
                    let b = (val * 0.40).round().clamp(0.0, 255.0) as u8;

                    out.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
                }
            }
        }

        DisplayMode::ThermalFusion => {
            // Pseudo-thermal ironbow false-color colormap
            for (p, o) in img.pixels().zip(out.pixels_mut()) {
                let y = (0.299 * (p[0] as f32) + 0.587 * (p[1] as f32) + 0.114 * (p[2] as f32))
                    / 255.0;

                // Ironbow transfer function
                let r = (y * 2.5 - 0.2).clamp(0.0, 1.0);
                let g = ((y - 0.35) * 2.2).clamp(0.0, 1.0);
                let b = (1.0 - (y * 2.2 - 0.4).abs()).clamp(0.0, 1.0);

                *o = Rgb([
                    (r * 255.0).round() as u8,
                    (g * 255.0).round() as u8,
                    (b * 255.0).round() as u8,
                ]);
            }
        }
    }

    out
}
