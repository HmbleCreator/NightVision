//! NightVision: Next-Generation Digital Computational Night Vision Engine in Rust.
//!
//! Solves the fundamental limits of analog vacuum-tube night vision and thermal cameras:
//! 1. Prohibitive Cost -> Runs on commodity CMOS sensors.
//! 2. Fragile Hardware & Flare Damage -> Solid state with digital auto-gating.
//! 3. Monochrome Quality -> Full-Color Night Vision with tactical phosphor options.
//! 4. Thermal Limitations -> High-resolution optical transmission through glass with fine text/details.
//! 5. Digital Sensor Noise -> Physics-based CMOS modeling, motion-compensated temporal SNR accumulation,
//!    edge-preserving guided denoising, Retinex reflectance recovery, and CLAHE.

pub mod enhancement;
pub mod isp;
pub mod metrics;
pub mod modes;
pub mod pipeline;
pub mod processor;
pub mod sensor;
pub mod spatial;
pub mod temporal;

pub use enhancement::{apply_clahe, apply_exposure_and_autogating, apply_retinex, EnhancementConfig};
pub use isp::ImageSignalProcessor;
pub use metrics::{calculate_psnr, calculate_snr_db, calculate_ssim};
pub use modes::{render_mode, DisplayMode};
pub use pipeline::{NightVisionPipeline, PipelineConfig, StageBreakdown};
pub use processor::FrameProcessor;
pub use sensor::{CmosSensorConfig, CmosSensorSimulator, LuxLevel, SensorSimulationStats};
pub use spatial::{denoise_image, SpatialDenoiseConfig};
pub use temporal::{AdaptiveTemporalAccumulator, TemporalConfig};

use image::{ImageBuffer, Rgb, RgbImage};

/// Generates a synthetic test chart containing fine resolution bars, text patterns,
/// and color checker blocks for ground-truth benchmarking without external assets.
pub fn generate_synthetic_test_scene(width: u32, height: u32) -> RgbImage {
    let mut img = ImageBuffer::new(width, height);

    // Color checker palette (Macbeth-like reference patches)
    let palette: [[u8; 3]; 12] = [
        [115, 82, 68],    // dark skin
        [194, 150, 130],  // light skin
        [98, 122, 157],   // blue sky
        [87, 108, 67],    // foliage
        [133, 128, 177],  // blue flower
        [103, 189, 170],  // bluish green
        [214, 126, 44],   // orange
        [80, 91, 166],    // purplish blue
        [193, 90, 99],    // moderate red
        [94, 60, 108],    // purple
        [157, 188, 64],   // yellow green
        [224, 163, 46],   // orange yellow
    ];

    for y in 0..height {
        for x in 0..width {
            let pixel = if y < height / 2 && x < width / 2 {
                let cell_x = (x / (width / 8)).min(3) as usize;
                let cell_y = (y / (height / 6)).min(2) as usize;
                let idx = (cell_y * 4 + cell_x) % palette.len();
                palette[idx]
            } else if y < height / 2 && x >= width / 2 {
                let freq = ((x - width / 2) as f32 / 8.0).sin();
                let v = if freq > 0.0 { 220 } else { 35 };
                [v, v, v]
            } else if y >= height / 2 && x < width / 2 {
                let grad = ((x as f32 / (width as f32 / 2.0)) * 220.0) as u8;
                [grad, (grad / 2), (220 - grad)]
            } else {
                let rel_x = x - width / 2;
                let rel_y = y - height / 2;
                if (rel_x % 40 > 10 && rel_x % 40 < 30) && (rel_y % 50 > 15 && rel_y % 50 < 40) {
                    [235, 195, 120]
                } else {
                    [45, 45, 55]
                }
            };

            let mut final_pixel = pixel;

            // High-contrast simulated text strip across the center
            if y >= height / 2 - 15 && y <= height / 2 + 15 {
                let char_pattern = (x / 12) % 2 == 0 && (y % 6 > 1);
                if char_pattern {
                    final_pixel = [250, 250, 250]; // White text
                } else {
                    final_pixel = [15, 15, 15]; // Black background behind text
                }
            }

            img.put_pixel(x, y, Rgb(final_pixel));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_smoke() {
        let test_scene = generate_synthetic_test_scene(160, 120);
        let simulator = CmosSensorSimulator::new(CmosSensorConfig::default());
        let (noisy_frame, stats) = simulator.simulate_low_light(&test_scene, LuxLevel::Starlight, None);

        assert_eq!(noisy_frame.dimensions(), (160, 120));
        assert!(stats.digital_gain_applied > 1.0);

        let mut pipeline = NightVisionPipeline::new(PipelineConfig::default());
        let enhanced = pipeline.process_frame(&noisy_frame);
        assert_eq!(enhanced.dimensions(), (160, 120));

        let psnr_noisy = calculate_psnr(&test_scene, &noisy_frame);
        let psnr_enhanced = calculate_psnr(&test_scene, &enhanced);
        println!("PSNR Noisy: {:.2} dB, Enhanced: {:.2} dB", psnr_noisy, psnr_enhanced);
    }

    #[test]
    fn test_desktop_frame_processor_seam() {
        let test_scene = generate_synthetic_test_scene(160, 120);
        let mut processor = NightVisionPipeline::new(PipelineConfig::default());

        assert_eq!(processor.backend_name(), "Desktop Multi-Core Rayon (f32 FPU)");

        processor.set_display_mode(DisplayMode::TacticalGreenPhosphor);
        assert_eq!(processor.display_mode(), DisplayMode::TacticalGreenPhosphor);

        let mut out_frame = image::RgbImage::new(160, 120);
        processor.process_frame_into(&test_scene, &mut out_frame);
        assert_eq!(out_frame.dimensions(), (160, 120));

        let out_alloc = processor.process_frame(&test_scene);
        assert_eq!(out_alloc.dimensions(), (160, 120));

        processor.reset();
    }

    #[test]
    fn test_isp_planar_zero_alloc_seam() {
        let test_scene = generate_synthetic_test_scene(160, 120);
        let mut isp = ImageSignalProcessor::new(PipelineConfig::fast_realtime());

        assert_eq!(isp.backend_name(), "Desktop Deep ISP (Planar Zero-Alloc FPU)");

        isp.set_display_mode(DisplayMode::TacticalWhitePhosphor);
        assert_eq!(isp.display_mode(), DisplayMode::TacticalWhitePhosphor);

        let mut out_frame = image::RgbImage::new(160, 120);
        isp.process_frame_into(&test_scene, &mut out_frame);
        assert_eq!(out_frame.dimensions(), (160, 120));

        let out_alloc = isp.process_frame(&test_scene);
        assert_eq!(out_alloc.dimensions(), (160, 120));
        isp.reset();
    }
}
