//! End-to-End Computational Night Vision Demonstration Script.
//!
//! Demonstrates the NightVision pipeline by:
//! 1. Generating a synthetic optical test chart.
//! 2. Simulating CMOS sensor degradation in Clear Starlight and Quarter-Moon regimes.
//! 3. Applying motion-compensated temporal accumulation and spatial guided denoising.
//! 4. Rendering tactical phosphor modes (P45 Tactical White, P43 Military Green, Recon Amber).
//! 5. Outputting quantitative SNR, PSNR, SSIM, and 4-panel panoramic comparison strips.
//!
//! Usage:
//!   cargo run --example demo
//!   cargo run --example demo -- --resolution 640x480 --frames 12

use clap::Parser;
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::ProgressBar;
use nightvision_core::{
    calculate_psnr, calculate_ssim, generate_synthetic_test_scene, CmosSensorConfig,
    CmosSensorSimulator, DisplayMode, LuxLevel, NightVisionPipeline, PipelineConfig,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "nightvision-demo",
    about = "End-to-End Computational Night Vision Demonstration Script"
)]
struct DemoArgs {
    /// Directory to store output demonstration images
    #[arg(short, long, default_value = "output/demo")]
    output_dir: PathBuf,

    /// Resolution in WIDTHxHEIGHT format
    #[arg(short, long, default_value = "640x480")]
    resolution: String,

    /// Number of temporal accumulation video frames
    #[arg(short, long, default_value_t = 12)]
    frames: usize,
}

fn parse_resolution(res: &str) -> (u32, u32) {
    let parts: Vec<&str> = res.split('x').collect();
    if parts.len() == 2 {
        let w = parts[0].parse::<u32>().unwrap_or(640);
        let h = parts[1].parse::<u32>().unwrap_or(480);
        (w, h)
    } else {
        (640, 480)
    }
}

/// Creates a horizontal side-by-side composite comparison image
fn create_side_by_side(images: &[(&str, &RgbImage)]) -> RgbImage {
    if images.is_empty() {
        return ImageBuffer::new(1, 1);
    }
    let (w, h) = images[0].1.dimensions();
    let count = images.len() as u32;
    let separator = 4u32;
    let total_w = w * count + separator * (count - 1);
    let mut canvas = ImageBuffer::new(total_w, h);

    for p in canvas.pixels_mut() {
        *p = Rgb([20, 20, 25]);
    }

    for (i, (_, img)) in images.iter().enumerate() {
        let x_offset = (i as u32) * (w + separator);
        for y in 0..h {
            for x in 0..w {
                canvas.put_pixel(x_offset + x, y, *img.get_pixel(x, y));
            }
        }
    }

    canvas
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = DemoArgs::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              NIGHTVISION COMPUTATIONAL IMAGING DEMO            ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    fs::create_dir_all(&args.output_dir)?;
    let (w, h) = parse_resolution(&args.resolution);
    println!("Creating ground-truth optical test chart ({w}x{h})...");
    let reference_chart = generate_synthetic_test_scene(w, h);
    let ref_path = args.output_dir.join("00_reference_ground_truth.png");
    reference_chart.save(&ref_path)?;
    println!(" -> Saved ground truth: {}", ref_path.display());

    let test_lux_scenarios = [
        ("starlight", LuxLevel::Starlight, "Clear Starlight (~0.001 lux)"),
        ("quarter_moon", LuxLevel::QuarterMoon, "Quarter Moon (~0.01 lux)"),
    ];

    let sensor_sim = CmosSensorSimulator::new(CmosSensorConfig::default());

    for (tag, lux, desc) in &test_lux_scenarios {
        println!();
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing Scenario: {desc}");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        // 1. Simulate single-frame raw CMOS capture
        let (raw_noisy_single, stats) = sensor_sim.simulate_low_light(&reference_chart, *lux, None);
        let single_noisy_path = args.output_dir.join(format!("01_{tag}_raw_cmos_single.png"));
        raw_noisy_single.save(&single_noisy_path)?;

        let psnr_raw = calculate_psnr(&reference_chart, &raw_noisy_single);
        let ssim_raw = calculate_ssim(&reference_chart, &raw_noisy_single);
        println!(
            " [Sensor Stats] Mean Photons/pixel: {:.2} e-, Shot Noise: {:.2} e-, Read Noise: {:.2} e-",
            stats.mean_photons_per_pixel, stats.photon_shot_noise_sigma, stats.read_noise_sigma
        );
        println!(
            " [Sensor Stats] Analog/Digital Gain Applied: {:.1}x (ISO equivalent)",
            stats.digital_gain_applied
        );
        println!(
            " [Raw Quality ] PSNR: {:.2} dB | SSIM: {:.4} (Severe static & photon starvation)",
            psnr_raw, ssim_raw
        );

        // 2. Simulate temporal multi-frame sequence
        println!(
            " Simulating {} consecutive video frames with temporal SNR accumulation...",
            args.frames
        );
        let mut pipeline = NightVisionPipeline::new(PipelineConfig::default());
        let mut last_enhanced = raw_noisy_single.clone();

        let pb = ProgressBar::new(args.frames as u64);
        let start_time = Instant::now();

        for _ in 0..args.frames {
            let (noisy_frame, _) = sensor_sim.simulate_low_light(
                &reference_chart,
                *lux,
                Some(stats.digital_gain_applied),
            );
            last_enhanced = pipeline.process_frame(&noisy_frame);
            pb.inc(1);
        }
        pb.finish_and_clear();
        let elapsed = start_time.elapsed();
        let fps = (args.frames as f64) / elapsed.as_secs_f64();

        let psnr_enhanced = calculate_psnr(&reference_chart, &last_enhanced);
        let ssim_enhanced = calculate_ssim(&reference_chart, &last_enhanced);
        let psnr_gain = psnr_enhanced - psnr_raw;

        println!(
            " [Enhancement] PSNR: {:.2} dB (+{:.2} dB gain!) | SSIM: {:.4}",
            psnr_enhanced, psnr_gain, ssim_enhanced
        );
        println!(
            " [Throughput ] Sustained Frame Rate: {:.1} FPS ({:.1} ms/frame)",
            fps,
            elapsed.as_secs_f64() * 1000.0 / (args.frames as f64)
        );

        let color_enhanced_path = args.output_dir.join(format!("02_{tag}_enhanced_full_color.png"));
        last_enhanced.save(&color_enhanced_path)?;

        // 3. Render tactical phosphor modes
        let tactical_modes = [
            (
                DisplayMode::TacticalWhitePhosphor,
                "white_phosphor",
                "Tactical White Phosphor (P45)",
            ),
            (
                DisplayMode::TacticalGreenPhosphor,
                "green_phosphor",
                "Tactical Green Phosphor (P43)",
            ),
            (
                DisplayMode::HighContrastRecon,
                "recon",
                "High-Contrast Recon",
            ),
        ];

        let mut composite_list = vec![
            ("Reference", &reference_chart),
            ("Raw CMOS", &raw_noisy_single),
            ("Full-Color", &last_enhanced),
        ];

        let mut mode_images = Vec::new();
        for (mode, mode_tag, mode_name) in &tactical_modes {
            let mut mode_pipeline = NightVisionPipeline::new(PipelineConfig {
                mode: *mode,
                ..pipeline.config.clone()
            });
            let rendered = mode_pipeline.process_frame(&raw_noisy_single);
            let mode_path = args.output_dir.join(format!("03_{tag}_{mode_tag}.png"));
            rendered.save(&mode_path)?;
            println!(" -> Saved {mode_name}: {}", mode_path.display());
            mode_images.push(rendered);
        }

        for img in &mode_images {
            composite_list.push(("Phosphor", img));
        }

        let sbs = create_side_by_side(&composite_list[..4]);
        let sbs_path = args
            .output_dir
            .join(format!("04_{tag}_side_by_side_comparison.png"));
        sbs.save(&sbs_path)?;
        println!(" -> Saved Composite Comparison: {}", sbs_path.display());
    }

    println!();
    println!("==================================================================");
    println!("All demonstration artifacts saved to: {}", args.output_dir.display());
    println!("==================================================================");

    Ok(())
}

