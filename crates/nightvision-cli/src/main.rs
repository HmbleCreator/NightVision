use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::ProgressBar;
use nightvision_core::{
    calculate_psnr, calculate_ssim, generate_synthetic_test_scene, CmosSensorConfig,
    CmosSensorSimulator, DisplayMode, LuxLevel, NightVisionPipeline, PipelineConfig,
};

#[derive(Parser)]
#[command(
    name = "nightvision-cli",
    about = "Next-Generation Digital Night Vision Computational Photography Engine",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run comprehensive night vision demonstration and generate side-by-side comparisons
    Demo {
        /// Directory to store output demonstration images
        #[arg(short, long, default_value = "output/demo")]
        output_dir: PathBuf,

        /// Resolution in WIDTHxHEIGHT format
        #[arg(short, long, default_value = "640x480")]
        resolution: String,

        /// Number of temporal accumulation video frames
        #[arg(short, long, default_value_t = 12)]
        frames: usize,
    },

    /// Process a custom input image or video frame sequence
    Process {
        /// Path to input image file
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output image file
        #[arg(short, long)]
        output: PathBuf,

        /// Optional simulated lux level (e.g. 0.001 for starlight, 0.01 for quarter-moon)
        #[arg(long)]
        simulate_lux: Option<f64>,

        /// Tactical display mode
        #[arg(short, long, value_enum, default_value_t = ModeArg::FullColor)]
        mode: ModeArg,

        /// Exposure boost multiplier
        #[arg(long, default_value_t = 2.5)]
        exposure: f32,

        /// Adaptive gamma value (0.3 to 1.0)
        #[arg(long, default_value_t = 0.6)]
        gamma: f32,
    },

    /// Benchmark pipeline latency and throughput (FPS) across resolutions
    Benchmark {
        /// Number of iterations per benchmark
        #[arg(short, long, default_value_t = 30)]
        iterations: usize,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug)]
enum ModeArg {
    FullColor,
    WhitePhosphor,
    GreenPhosphor,
    Recon,
    Thermal,
}

impl From<ModeArg> for DisplayMode {
    fn from(arg: ModeArg) -> Self {
        match arg {
            ModeArg::FullColor => DisplayMode::FullColor,
            ModeArg::WhitePhosphor => DisplayMode::TacticalWhitePhosphor,
            ModeArg::GreenPhosphor => DisplayMode::TacticalGreenPhosphor,
            ModeArg::Recon => DisplayMode::HighContrastRecon,
            ModeArg::Thermal => DisplayMode::ThermalFusion,
        }
    }
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

    // Initialize with dark gray
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
    let cli = Cli::parse();

    match cli.command {
        Commands::Demo {
            output_dir,
            resolution,
            frames,
        } => {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║              NIGHTVISION COMPUTATIONAL IMAGING DEMO            ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            println!();

            fs::create_dir_all(&output_dir)?;
            let (w, h) = parse_resolution(&resolution);
            println!("Creating ground-truth optical test chart ({w}x{h})...");
            let reference_chart = generate_synthetic_test_scene(w, h);
            let ref_path = output_dir.join("00_reference_ground_truth.png");
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
                let single_noisy_path = output_dir.join(format!("01_{tag}_raw_cmos_single.png"));
                raw_noisy_single.save(&single_noisy_path)?;

                let psnr_raw = calculate_psnr(&reference_chart, &raw_noisy_single);
                let ssim_raw = calculate_ssim(&reference_chart, &raw_noisy_single);
                println!(" [Sensor Stats] Mean Photons/pixel: {:.2} e-, Shot Noise: {:.2} e-, Read Noise: {:.2} e-",
                    stats.mean_photons_per_pixel, stats.photon_shot_noise_sigma, stats.read_noise_sigma);
                println!(" [Sensor Stats] Analog/Digital Gain Applied: {:.1}x (ISO equivalent)", stats.digital_gain_applied);
                println!(" [Raw Quality ] PSNR: {:.2} dB | SSIM: {:.4} (Severe static & photon starvation)", psnr_raw, ssim_raw);

                // 2. Simulate temporal multi-frame sequence
                println!(" Simulating {frames} consecutive video frames with temporal SNR accumulation...");
                let mut pipeline = NightVisionPipeline::new(PipelineConfig::default());
                let mut last_enhanced = raw_noisy_single.clone();

                let pb = ProgressBar::new(frames as u64);
                let start_time = Instant::now();

                for _ in 0..frames {
                    // Generate new independent Poisson-Gaussian noise frame from sensor
                    let (noisy_frame, _) = sensor_sim.simulate_low_light(&reference_chart, *lux, Some(stats.digital_gain_applied));
                    last_enhanced = pipeline.process_frame(&noisy_frame);
                    pb.inc(1);
                }
                pb.finish_and_clear();
                let elapsed = start_time.elapsed();
                let fps = (frames as f64) / elapsed.as_secs_f64();

                let psnr_enhanced = calculate_psnr(&reference_chart, &last_enhanced);
                let ssim_enhanced = calculate_ssim(&reference_chart, &last_enhanced);
                let psnr_gain = psnr_enhanced - psnr_raw;

                println!(" [Enhancement] PSNR: {:.2} dB (+{:.2} dB gain!) | SSIM: {:.4}",
                    psnr_enhanced, psnr_gain, ssim_enhanced);
                println!(" [Throughput ] Sustained Frame Rate: {:.1} FPS ({:.1} ms/frame)", fps, elapsed.as_secs_f64() * 1000.0 / (frames as f64));

                let color_enhanced_path = output_dir.join(format!("02_{tag}_enhanced_full_color.png"));
                last_enhanced.save(&color_enhanced_path)?;

                // 3. Render tactical phosphor modes
                let tactical_modes = [
                    (DisplayMode::TacticalWhitePhosphor, "white_phosphor", "Tactical White Phosphor (P45)"),
                    (DisplayMode::TacticalGreenPhosphor, "green_phosphor", "Tactical Green Phosphor (P43)"),
                    (DisplayMode::HighContrastRecon, "recon", "High-Contrast Recon"),
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
                    let mode_path = output_dir.join(format!("03_{tag}_{mode_tag}.png"));
                    rendered.save(&mode_path)?;
                    println!(" -> Saved {mode_name}: {}", mode_path.display());
                    mode_images.push(rendered);
                }

                for img in &mode_images {
                    composite_list.push(("Phosphor", img));
                }

                let sbs = create_side_by_side(&composite_list[..4]);
                let sbs_path = output_dir.join(format!("04_{tag}_side_by_side_comparison.png"));
                sbs.save(&sbs_path)?;
                println!(" -> Saved Composite Comparison: {}", sbs_path.display());
            }

            println!();
            println!("==================================================================");
            println!("All demonstration artifacts saved to: {}", output_dir.display());
            println!("==================================================================");
        }

        Commands::Process {
            input,
            output,
            simulate_lux,
            mode,
            exposure,
            gamma,
        } => {
            if !input.exists() {
                eprintln!("Error: Input file does not exist: {}", input.display());
                std::process::exit(1);
            }

            println!("Loading input image: {}", input.display());
            let src_img = image::open(&input)?.to_rgb8();

            let target_img = if let Some(lux) = simulate_lux {
                println!("Simulating low-light sensor noise at {:.4} lux...", lux);
                let sim = CmosSensorSimulator::new(CmosSensorConfig::default());
                let (noisy, stats) = sim.simulate_low_light(&src_img, LuxLevel::Custom(lux), None);
                println!("  Raw SNR: {:.1} dB, Gain: {:.1}x", stats.estimated_snr_db, stats.digital_gain_applied);
                noisy
            } else {
                src_img
            };

            let mut config = PipelineConfig::default();
            config.mode = mode.into();
            config.enhancement.exposure_gain = exposure;
            config.enhancement.gamma = gamma;

            println!("Processing through Digital Night Vision pipeline...");
            let start = Instant::now();
            let mut pipeline = NightVisionPipeline::new(config);
            let result = pipeline.process_frame(&target_img);
            let elapsed = start.elapsed();

            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            result.save(&output)?;
            println!("Successfully processed in {:.2} ms!", elapsed.as_secs_f64() * 1000.0);
            println!("Output saved to: {}", output.display());
        }

        Commands::Benchmark { iterations } => {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║              NIGHTVISION PIPELINE BENCHMARK SUITE              ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            println!();

            let resolutions = [
                ("640x480 (VGA / Standard Tactical HUD)", 640, 480),
                ("1280x720 (HD 720p Real-time Stream)", 1280, 720),
                ("1920x1080 (Full HD 1080p Tactical)", 1920, 1080),
            ];

            for (label, w, h) in resolutions {
                println!("Benchmarking: {label}");
                let frame = generate_synthetic_test_scene(w, h);
                let mut pipeline = NightVisionPipeline::new(PipelineConfig::default());

                // Warm up
                for _ in 0..3 {
                    let _ = pipeline.process_frame(&frame);
                }

                let start = Instant::now();
                for _ in 0..iterations {
                    let _ = pipeline.process_frame(&frame);
                }
                let total_time = start.elapsed();
                let avg_latency_ms = (total_time.as_secs_f64() * 1000.0) / (iterations as f64);
                let fps = (iterations as f64) / total_time.as_secs_f64();

                println!("  Iterations: {iterations}");
                println!("  Avg Latency: {:.2} ms", avg_latency_ms);
                println!("  Throughput:  {:.1} FPS", fps);
                println!();
            }
        }
    }

    Ok(())
}
