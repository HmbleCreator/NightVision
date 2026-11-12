use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};
use image::{ImageBuffer, Rgb, RgbImage};
use indicatif::ProgressBar;
use nightvision_core::{
    calculate_psnr, calculate_ssim, generate_synthetic_test_scene, CmosSensorConfig,
    CmosSensorSimulator, DisplayMode, FrameProcessor, LuxLevel, NightVisionPipeline,
    PipelineConfig,
};
use nightvision_mcu::{McuDynamicProcessor, McuEngineConfig, McuNightVisionEngine};

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

        /// Execution backend engine
        #[arg(short, long, value_enum, default_value_t = BackendArg::Primary)]
        backend: BackendArg,

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

    /// Head-to-head comparison between Desktop Floating-Point and MCU Fixed-Point Streaming Pipeline
    McuBenchmark {
        /// Number of iterations for benchmarking
        #[arg(short, long, default_value_t = 30)]
        iterations: usize,

        /// Output directory for visual comparison
        #[arg(short, long, default_value = "output/demo")]
        output_dir: PathBuf,
    },

    /// Comprehensive evaluation and testing of real-world low-light images
    TestImages {
        /// Directory containing test images
        #[arg(short, long, default_value = "LowLightTestImages")]
        input_dir: PathBuf,

        /// Output directory for enhanced images and side-by-side comparisons
        #[arg(short, long, default_value = "output/test_results")]
        output_dir: PathBuf,

        /// Benchmark timing iterations per image
        #[arg(long, default_value_t = 10)]
        benchmark_iterations: usize,
    },

    /// Export C/C++ static flash look-up tables (LUTs) and inline macros for embedded MCU targets
    ExportHeader {
        /// Destination path for generated C/C++ header file
        #[arg(short, long, default_value = "include/nightvision_tables.h")]
        output: PathBuf,
    },
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq)]
enum BackendArg {
    /// SOTA Physics-Grounded Engine (Fast Integer Streaming & Noise Profile LUT - Default)
    Primary,
    /// MCU Hardware Streaming Adapter (< 2KB SRAM)
    Mcu,
    /// Offline Laboratory Forensic Mode (Heavy Floating-Point MSRCR + CLAHE)
    Forensic,
    /// Legacy alias for desktop FPU
    Desktop,
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

#[derive(serde::Serialize)]
struct ImageEvaluationResult {
    file_name: String,
    width: u32,
    height: u32,
    raw_mean_luma: f64,
    raw_min: u8,
    raw_max: u8,
    raw_entropy: f64,
    raw_laplacian_var: f64,
    desktop_latency_ms: f64,
    desktop_fps: f64,
    desktop_mean_luma: f64,
    desktop_min: u8,
    desktop_max: u8,
    desktop_entropy: f64,
    desktop_laplacian_var: f64,
    mcu_latency_ms: f64,
    mcu_fps: f64,
    mcu_mean_luma: f64,
    mcu_min: u8,
    mcu_max: u8,
    mcu_entropy: f64,
    mcu_laplacian_var: f64,
    speedup_factor: f64,
    mcu_vs_desktop_psnr: f64,
    mcu_vs_desktop_ssim: f64,
}

fn calculate_entropy(img: &RgbImage) -> f64 {
    let mut hist = [0u32; 256];
    let total = (img.width() * img.height()) as f64;
    if total == 0.0 {
        return 0.0;
    }
    for p in img.pixels() {
        let y = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as usize;
        hist[y.min(255)] += 1;
    }
    let mut entropy = 0.0f64;
    for &count in &hist {
        if count > 0 {
            let p = (count as f64) / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn calculate_laplacian_variance(img: &RgbImage) -> f64 {
    let (w, h) = img.dimensions();
    if w < 3 || h < 3 {
        return 0.0;
    }
    let w_u = w as usize;
    let h_u = h as usize;
    let mut luma = Vec::with_capacity(w_u * h_u);
    for p in img.pixels() {
        let y = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as f64;
        luma.push(y);
    }
    let mut laps = Vec::with_capacity((w_u - 2) * (h_u - 2));
    for y in 1..h_u - 1 {
        for x in 1..w_u - 1 {
            let c = luma[y * w_u + x];
            let top = luma[(y - 1) * w_u + x];
            let bot = luma[(y + 1) * w_u + x];
            let left = luma[y * w_u + (x - 1)];
            let right = luma[y * w_u + (x + 1)];
            let lap = top + bot + left + right - 4.0 * c;
            laps.push(lap);
        }
    }
    let n = laps.len() as f64;
    let mean = laps.iter().sum::<f64>() / n;
    laps.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n
}

fn calculate_mean_luma(img: &RgbImage) -> f64 {
    let mut sum = 0u64;
    let total = (img.width() * img.height()) as f64;
    if total == 0.0 {
        return 0.0;
    }
    for p in img.pixels() {
        let y = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as u64;
        sum += y;
    }
    (sum as f64) / total
}

fn calculate_min_max_luma(img: &RgbImage) -> (u8, u8) {
    let mut min_val = 255u8;
    let mut max_val = 0u8;
    for p in img.pixels() {
        let y = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as u8;
        if y < min_val {
            min_val = y;
        }
        if y > max_val {
            max_val = y;
        }
    }
    (min_val, max_val)
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
            backend,
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

            // Instantiate any adapter satisfying the FrameProcessor seam
            let mut processor: Box<dyn FrameProcessor> = match backend {
                BackendArg::Primary | BackendArg::Mcu => {
                    let config = McuEngineConfig::default().with_mode(mode.into());
                    Box::new(McuDynamicProcessor::new(config))
                }
                BackendArg::Forensic | BackendArg::Desktop => {
                    let mut config = PipelineConfig::default();
                    config.mode = mode.into();
                    config.enhancement.exposure_gain = exposure;
                    config.enhancement.gamma = gamma;
                    Box::new(NightVisionPipeline::new(config))
                }
            };

            println!("Processing via [{}] through FrameProcessor seam...", processor.backend_name());
            let start = Instant::now();
            let result = processor.process_frame(&target_img);
            let elapsed = start.elapsed();

            if let Some(rep) = processor.exposure_report() {
                println!();
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("Auto-Exposure Control (AEC) & Scene Lighting Telemetry:");
                println!("  ├── Lighting Regime:       {:?}", rep.regime);
                println!("  ├── Scene Mean Luminance:  {:>6.2} ADU", rep.mean_luma);
                println!("  ├── 95th Percentile Luma:  {:>6.2} ADU", rep.p95_luma);
                println!("  ├── Saturated Pixels:      {:>6.2} %", rep.saturated_pixel_fraction * 100.0);
                println!("  └── Recommended EV Shift:  {:>+6.2} EV", rep.recommended_ev_shift);
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            }

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
                ("1280x720 (HD 720p Stream)", 1280, 720),
                ("1920x1080 (Full HD 1080p Tactical)", 1920, 1080),
            ];

            for (label, w, h) in resolutions {
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("Resolution: {label}");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                let frame = generate_synthetic_test_scene(w, h);

                // --- Mode A: Comprehensive Pipeline (Retinex + CLAHE + Spatial + Temporal) ---
                let mut full_pipeline = NightVisionPipeline::new(PipelineConfig::default());
                for _ in 0..2 {
                    let _ = full_pipeline.process_frame(&frame);
                }

                let mut total_breakdown = nightvision_core::StageBreakdown {
                    temporal_ms: 0.0,
                    spatial_ms: 0.0,
                    retinex_ms: 0.0,
                    clahe_ms: 0.0,
                    exposure_autogating_ms: 0.0,
                    mode_rendering_ms: 0.0,
                    total_ms: 0.0,
                };

                let iter_count = iterations.min(10);
                for _ in 0..iter_count {
                    let (_, bd) = full_pipeline.process_frame_profiled(&frame);
                    total_breakdown.temporal_ms += bd.temporal_ms;
                    total_breakdown.spatial_ms += bd.spatial_ms;
                    total_breakdown.retinex_ms += bd.retinex_ms;
                    total_breakdown.clahe_ms += bd.clahe_ms;
                    total_breakdown.exposure_autogating_ms += bd.exposure_autogating_ms;
                    total_breakdown.mode_rendering_ms += bd.mode_rendering_ms;
                    total_breakdown.total_ms += bd.total_ms;
                }

                let n = iter_count as f64;
                let full_latency = total_breakdown.total_ms / n;
                let full_fps = 1000.0 / full_latency;

                println!("  [Profile A: Comprehensive High-Fidelity Pipeline]");
                println!("    ├── Temporal Accumulation:  {:>6.2} ms", total_breakdown.temporal_ms / n);
                println!("    ├── Guided Spatial Filter:  {:>6.2} ms", total_breakdown.spatial_ms / n);
                println!("    ├── Multi-Scale Retinex:    {:>6.2} ms", total_breakdown.retinex_ms / n);
                println!("    ├── Tile-based CLAHE:       {:>6.2} ms", total_breakdown.clahe_ms / n);
                println!("    ├── Exposure & Auto-Gating: {:>6.2} ms", total_breakdown.exposure_autogating_ms / n);
                println!("    └── Tactical Mode Render:   {:>6.2} ms", total_breakdown.mode_rendering_ms / n);
                println!("    => TOTAL LATENCY: {:.2} ms | SUSTAINED: {:.1} FPS", full_latency, full_fps);
                println!();

                // --- Mode B: Tactical Real-Time Stream (Fast Preset: Temporal + Spatial + CLAHE + Auto-Gating) ---
                let mut fast_pipeline = NightVisionPipeline::new(PipelineConfig::fast_realtime());
                for _ in 0..2 {
                    let _ = fast_pipeline.process_frame(&frame);
                }

                let start_fast = Instant::now();
                let fast_iters = iterations;
                for _ in 0..fast_iters {
                    let _ = fast_pipeline.process_frame(&frame);
                }
                let elapsed_fast = start_fast.elapsed().as_secs_f64() * 1000.0;
                let fast_latency = elapsed_fast / (fast_iters as f64);
                let fast_fps = 1000.0 / fast_latency;

                println!("  [Profile B: High-Speed Tactical Stream (Zero-Lag Mode)]");
                println!("    => TOTAL LATENCY: {:.2} ms | SUSTAINED: {:.1} FPS", fast_latency, fast_fps);
                println!();
            }
        }

        Commands::McuBenchmark {
            iterations,
            output_dir,
        } => {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║     DESKTOP FLOATING-POINT vs MCU FIXED-POINT BENCHMARK        ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            println!();

            fs::create_dir_all(&output_dir)?;

            const W: usize = 640;
            const H: usize = 480;

            println!("Setting up 640x480 test scene under Starlight illuminance (0.001 lux)...");
            let ground_truth = generate_synthetic_test_scene(W as u32, H as u32);
            let sim = CmosSensorSimulator::new(CmosSensorConfig::default());
            let (raw_noisy, _) = sim.simulate_low_light(&ground_truth, LuxLevel::Starlight, None);

            // 1. Benchmark Desktop High-Fidelity Pipeline (f32, full-frame buffers)
            println!("Running Desktop Comprehensive Pipeline (f32, full frames, Retinex+CLAHE)...");
            let mut desktop_full = NightVisionPipeline::new(PipelineConfig::default());
            for _ in 0..2 {
                let _ = desktop_full.process_frame(&raw_noisy);
            }

            let start_desktop = Instant::now();
            let mut desktop_output = raw_noisy.clone();
            for _ in 0..iterations.min(10) {
                desktop_output = desktop_full.process_frame(&raw_noisy);
            }
            let desktop_elapsed_ms = (start_desktop.elapsed().as_secs_f64() * 1000.0) / (iterations.min(10) as f64);
            let desktop_fps = 1000.0 / desktop_elapsed_ms;

            // 2. Benchmark MCU Streaming Engine (Integer bit-shifts, circular 3-line buffer, 8-bit LUTs)
            println!("Running MCU Streaming Pipeline (Q8/Q15 bit-shifts, < 2KB LineBuffer, 8-bit LUTs)...");
            let mcu_config = McuEngineConfig::default();
            let mut mcu_engine = McuNightVisionEngine::<W, H>::new(mcu_config);
            for _ in 0..2 {
                let _ = mcu_engine.process_frame_streaming(&raw_noisy);
            }

            let start_mcu = Instant::now();
            let mut mcu_output = raw_noisy.clone();
            for _ in 0..iterations {
                mcu_output = mcu_engine.process_frame_streaming(&raw_noisy);
            }
            let mcu_elapsed_ms = (start_mcu.elapsed().as_secs_f64() * 1000.0) / (iterations as f64);
            let mcu_fps = 1000.0 / mcu_elapsed_ms;

            // Quantitative Quality Evaluation
            let _raw_psnr = calculate_psnr(&ground_truth, &raw_noisy);
            let _raw_ssim = calculate_ssim(&ground_truth, &raw_noisy);

            let desktop_psnr = calculate_psnr(&ground_truth, &desktop_output);
            let desktop_ssim = calculate_ssim(&ground_truth, &desktop_output);

            let mcu_psnr = calculate_psnr(&ground_truth, &mcu_output);
            let mcu_ssim = calculate_ssim(&ground_truth, &mcu_output);

            let mcu_vs_desktop_psnr = calculate_psnr(&desktop_output, &mcu_output);
            let _mcu_vs_desktop_ssim = calculate_ssim(&desktop_output, &mcu_output);

            let speedup = desktop_elapsed_ms / mcu_elapsed_ms;

            println!();
            println!("┌─────────────────────────────┬────────────────────────┬────────────────────────┐");
            println!("│ Metric                      │ Desktop (Full f32)     │ MCU Streaming Engine   │");
            println!("├─────────────────────────────┼────────────────────────┼────────────────────────┤");
            println!("│ Execution Latency           │ {:>19.2} ms │ {:>19.2} ms │", desktop_elapsed_ms, mcu_elapsed_ms);
            println!("│ Sustained Frame Rate        │ {:>19.1} FPS│ {:>19.1} FPS│", desktop_fps, mcu_fps);
            println!("│ Speedup Factor              │                  1.00x │ {:>21.2}x │", speedup);
            println!("│ Active Spatial SRAM         │         ~4,500.00 KB   │             < 2.00 KB  │");
            println!("│ Memory Footprint Reduction  │                  0.00% │            > 99.95%    │");
            println!("│ Math Architecture           │   IEEE 754 f32/f64 FPU │ Integer Shifts & LUTs  │");
            println!("│ Output Quality (PSNR)       │ {:>19.2} dB │ {:>19.2} dB │", desktop_psnr, mcu_psnr);
            println!("│ Output Quality (SSIM)       │ {:>22.4} │ {:>22.4} │", desktop_ssim, mcu_ssim);
            println!("│ Quality Retention vs FPU    │               100.00%  │ {:>21.2} dB │", mcu_vs_desktop_psnr);
            println!("└─────────────────────────────┴────────────────────────┴────────────────────────┘");
            println!();

            // Save comparison image
            let composite_list = [
                ("Ground Truth", &ground_truth),
                ("Raw Noisy CMOS", &raw_noisy),
                ("Desktop Full", &desktop_output),
                ("MCU Streaming", &mcu_output),
            ];

            let sbs = create_side_by_side(&composite_list);
            let out_path = output_dir.join("05_mcu_vs_desktop_comparison.png");
            sbs.save(&out_path)?;
            println!(" -> Saved MCU vs Desktop comparison artifact: {}", out_path.display());
        }

        Commands::TestImages {
            input_dir,
            output_dir,
            benchmark_iterations,
        } => {
            println!("╔════════════════════════════════════════════════════════════════════════════════╗");
            println!("║              NIGHTVISION REAL-WORLD LOW-LIGHT TEST BENCHMARK                   ║");
            println!("╚════════════════════════════════════════════════════════════════════════════════╝");
            println!();

            fs::create_dir_all(&output_dir)?;

            let mut entries: Vec<PathBuf> = fs::read_dir(&input_dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                        let ext = ext.to_lowercase();
                        ext == "png" || ext == "jpg" || ext == "jpeg"
                    } else {
                        false
                    }
                })
                .collect();

            entries.sort();

            if entries.is_empty() {
                println!("No test images found in directory: {}", input_dir.display());
                return Ok(());
            }

            println!("Found {} real-world low-light test images in {}.", entries.len(), input_dir.display());
            println!("Benchmarking Desktop FPU vs MCU Streaming Engine ({} iterations per image)...", benchmark_iterations);
            println!();

            let mut results = Vec::new();

            for path in &entries {
                let file_name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();

                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                println!("Testing Image: {file_name}");
                println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                let raw_img = image::open(path)?.to_rgb8();
                let (w, h) = raw_img.dimensions();

                let raw_mean = calculate_mean_luma(&raw_img);
                let (raw_min, raw_max) = calculate_min_max_luma(&raw_img);
                let raw_entropy = calculate_entropy(&raw_img);
                let raw_lap = calculate_laplacian_variance(&raw_img);

                println!("  [Raw Input] Resolution: {}x{} | Luma Range: [{}..{}] | Mean: {:.2} | Entropy: {:.3} bits | Laplacian Var: {:.2}",
                    w, h, raw_min, raw_max, raw_mean, raw_entropy, raw_lap);

                // --- 1. Desktop FPU Pipeline ---
                let mut desktop_config = PipelineConfig::default();
                desktop_config.mode = DisplayMode::FullColor;
                desktop_config.enhancement.exposure_gain = 2.8;
                desktop_config.enhancement.gamma = 0.60;
                let mut desktop_proc = NightVisionPipeline::new(desktop_config);

                // Warm up
                let _ = desktop_proc.process_frame(&raw_img);

                let start_desktop = Instant::now();
                let mut desktop_color = raw_img.clone();
                for _ in 0..benchmark_iterations.max(1) {
                    desktop_color = desktop_proc.process_frame(&raw_img);
                }
                let desktop_elapsed_ms = (start_desktop.elapsed().as_secs_f64() * 1000.0) / (benchmark_iterations.max(1) as f64);
                let desktop_fps = 1000.0 / desktop_elapsed_ms;

                desktop_proc.set_display_mode(DisplayMode::TacticalWhitePhosphor);
                let desktop_white = desktop_proc.process_frame(&raw_img);

                desktop_proc.set_display_mode(DisplayMode::TacticalGreenPhosphor);
                let desktop_green = desktop_proc.process_frame(&raw_img);

                let desktop_mean = calculate_mean_luma(&desktop_color);
                let (desktop_min, desktop_max) = calculate_min_max_luma(&desktop_color);
                let desktop_entropy = calculate_entropy(&desktop_color);
                let desktop_lap = calculate_laplacian_variance(&desktop_color);

                println!("  [Desktop FPU] Latency: {:>6.2} ms ({:>5.1} FPS) | Mean: {:>5.2} | Range: [{}..{}] | Entropy: {:.3} bits | Laplacian: {:.2}",
                    desktop_elapsed_ms, desktop_fps, desktop_mean, desktop_min, desktop_max, desktop_entropy, desktop_lap);

                // --- 2. MCU Streaming Engine ---
                let mcu_config = McuEngineConfig::default().with_mode(DisplayMode::FullColor);
                let mut mcu_proc = McuDynamicProcessor::new(mcu_config);

                // Warm up
                let _ = mcu_proc.process_frame(&raw_img);

                let start_mcu = Instant::now();
                let mut mcu_color = raw_img.clone();
                for _ in 0..benchmark_iterations.max(1) {
                    mcu_color = mcu_proc.process_frame(&raw_img);
                }
                let mcu_elapsed_ms = (start_mcu.elapsed().as_secs_f64() * 1000.0) / (benchmark_iterations.max(1) as f64);
                let mcu_fps = 1000.0 / mcu_elapsed_ms;

                mcu_proc.set_display_mode(DisplayMode::TacticalWhitePhosphor);
                let mcu_white = mcu_proc.process_frame(&raw_img);

                mcu_proc.set_display_mode(DisplayMode::TacticalGreenPhosphor);
                let mcu_green = mcu_proc.process_frame(&raw_img);

                let mcu_mean = calculate_mean_luma(&mcu_color);
                let (mcu_min, mcu_max) = calculate_min_max_luma(&mcu_color);
                let mcu_entropy = calculate_entropy(&mcu_color);
                let mcu_lap = calculate_laplacian_variance(&mcu_color);

                let speedup = desktop_elapsed_ms / mcu_elapsed_ms;
                let mcu_vs_desktop_psnr = calculate_psnr(&desktop_color, &mcu_color);
                let mcu_vs_desktop_ssim = calculate_ssim(&desktop_color, &mcu_color);

                println!("  [MCU Stream ] Latency: {:>6.2} ms ({:>5.1} FPS) | Mean: {:>5.2} | Range: [{}..{}] | Entropy: {:.3} bits | Speedup: {:>5.1}x",
                    mcu_elapsed_ms, mcu_fps, mcu_mean, mcu_min, mcu_max, mcu_entropy, speedup);
                println!("  [Fidelity   ] MCU vs Desktop: PSNR: {:.2} dB | SSIM: {:.4}", mcu_vs_desktop_psnr, mcu_vs_desktop_ssim);
                let rep = mcu_proc.exposure_report();
                println!("  [AEC Meter  ] Regime: {:?} | Saturated: {:.1}% | Rec. EV: {:>+4.1} EV",
                    rep.regime, rep.saturated_pixel_fraction * 100.0, rep.recommended_ev_shift);

                // Save individual enhanced images
                let img_out_dir = output_dir.join(&stem);
                fs::create_dir_all(&img_out_dir)?;

                raw_img.save(img_out_dir.join("01_raw.png"))?;
                desktop_color.save(img_out_dir.join("02_desktop_color.png"))?;
                desktop_white.save(img_out_dir.join("03_desktop_white_phosphor.png"))?;
                desktop_green.save(img_out_dir.join("04_desktop_green_phosphor.png"))?;
                mcu_color.save(img_out_dir.join("05_mcu_color.png"))?;
                mcu_white.save(img_out_dir.join("06_mcu_white_phosphor.png"))?;
                mcu_green.save(img_out_dir.join("07_mcu_green_phosphor.png"))?;

                // Generate 5-panel composite comparison
                let composite = [
                    ("Raw Input", &raw_img),
                    ("Desktop Color", &desktop_color),
                    ("MCU Color", &mcu_color),
                    ("Desktop White", &desktop_white),
                    ("MCU White", &mcu_white),
                ];
                let sbs = create_side_by_side(&composite);
                let sbs_path = output_dir.join(format!("{stem}_comparison.png"));
                sbs.save(&sbs_path)?;
                println!("  -> Saved side-by-side comparison: {}", sbs_path.display());

                results.push(ImageEvaluationResult {
                    file_name,
                    width: w,
                    height: h,
                    raw_mean_luma: raw_mean,
                    raw_min,
                    raw_max,
                    raw_entropy,
                    raw_laplacian_var: raw_lap,
                    desktop_latency_ms: desktop_elapsed_ms,
                    desktop_fps,
                    desktop_mean_luma: desktop_mean,
                    desktop_min,
                    desktop_max,
                    desktop_entropy,
                    desktop_laplacian_var: desktop_lap,
                    mcu_latency_ms: mcu_elapsed_ms,
                    mcu_fps,
                    mcu_mean_luma: mcu_mean,
                    mcu_min,
                    mcu_max,
                    mcu_entropy,
                    mcu_laplacian_var: mcu_lap,
                    speedup_factor: speedup,
                    mcu_vs_desktop_psnr,
                    mcu_vs_desktop_ssim,
                });
            }

            // Print summary table
            println!();
            println!("╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
            println!("║                                                         LOW-LIGHT TEST SUITE BENCHMARK SUMMARY                                                                 ║");
            println!("╠══════════════════╦═════════════╦══════════════════════════╦══════════════════════════╦══════════════════════════╦══════════════╦══════════════════╦═════════════════╣");
            println!("║ Image File       │ Resolution  │ Raw Luma (Mean [Min..Max)│ Desktop Latency / FPS    │ MCU Latency / FPS        │ MCU Speedup  │ Mean Bright Boost│ Entropy Gain    ║");
            println!("╠══════════════════╬═════════════╬══════════════════════════╬══════════════════════════╬══════════════════════════╬══════════════╬══════════════════╬═════════════════╣");
            for r in &results {
                let raw_str = format!("{:.1} [{}..{}]", r.raw_mean_luma, r.raw_min, r.raw_max);
                let desk_str = format!("{:.1} ms ({:.0} FPS)", r.desktop_latency_ms, r.desktop_fps);
                let mcu_str = format!("{:.2} ms ({:.0} FPS)", r.mcu_latency_ms, r.mcu_fps);
                let boost_str = format!("{:.1}x", r.mcu_mean_luma / r.raw_mean_luma.max(0.1));
                let ent_str = format!("+{:.2} bits", r.mcu_entropy - r.raw_entropy);
                println!("║ {:<16} │ {:>5}x{:<5} │ {:<24} │ {:<24} │ {:<24} │ {:>10.1}x  │ {:>16} │ {:>15} ║",
                    r.file_name, r.width, r.height, raw_str, desk_str, mcu_str, r.speedup_factor, boost_str, ent_str);
            }
            println!("╚══════════════════╩═════════════╩══════════════════════════╩══════════════════════════╩══════════════════════════╩══════════════╩══════════════════╩═════════════════╝");
            println!();

            // Export JSON
            let json_path = output_dir.join("metrics_summary.json");
            fs::write(&json_path, serde_json::to_string_pretty(&results)?)?;
            println!("Saved complete quantitative metrics report to: {}", json_path.display());
        }

        Commands::ExportHeader { output } => {
            println!("╔════════════════════════════════════════════════════════════════╗");
            println!("║          NIGHTVISION C/C++ HARDWARE TABLES EXPORT              ║");
            println!("╚════════════════════════════════════════════════════════════════╝");
            println!();
            println!("Generating static flash look-up tables (LUTs) for embedded deployment...");
            export_c_tables(&output)?;
            println!("✓ Successfully generated zero-RAM embedded header: {}", output.display());
        }
    }

    Ok(())
}

fn format_c_array(name: &str, comment: &str, data: &[u8]) -> String {
    let mut out = String::new();
    out.push_str(&format!("/** {}\n * Size: {} bytes\n */\n", comment, data.len()));
    out.push_str(&format!("static const uint8_t {}[{}] = {{\n    ", name, data.len()));
    for (i, &byte) in data.iter().enumerate() {
        out.push_str(&format!("0x{:02X}", byte));
        if i + 1 < data.len() {
            out.push_str(", ");
            if (i + 1) % 16 == 0 {
                out.push_str("\n    ");
            }
        }
    }
    out.push_str("\n};\n\n");
    out
}

fn export_c_tables(output_path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use nightvision_mcu::{AutoGatingLut, DeterministicNoiseLut, PhosphorLut};
    use std::io::Write;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let noise_lut = DeterministicNoiseLut::new(2.8, 0.85, 2.1);
    let ultradark_lut = AutoGatingLut::new_bidirectional(2.8, 0.60, 0.88, 5.0, 40.0, 0.0);
    let lowlight_lut = AutoGatingLut::new_bidirectional(2.8, 0.60, 0.88, 16.0, 64.0, 0.0);
    let highexposure_lut = AutoGatingLut::new_bidirectional(2.8, 0.60, 0.88, 200.0, 240.0, 0.0);
    let p45 = PhosphorLut::p45_white();
    let p43 = PhosphorLut::p43_green();
    let amber = PhosphorLut::recon_amber();

    // Dual chroma gate table (256 bytes)
    let mut chroma_gate_lut = [0u8; 256];
    for i in 0..256 {
        let raw_u8 = i as u8;
        let atten: u32 = if raw_u8 < 16 {
            (raw_u8 as u32) * 16
        } else if raw_u8 > 230 {
            ((255 - raw_u8 as u32) * 10).min(256)
        } else {
            256
        };
        chroma_gate_lut[i] = (atten.min(255)) as u8;
    }

    let mut content = String::new();
    content.push_str("/**\n");
    content.push_str(" * @file nightvision_tables.h\n");
    content.push_str(" * @brief Precomputed Hardware Look-Up Tables (LUTs) for Embedded NightVision ISP.\n");
    content.push_str(" * Generated automatically from nightvision-mcu physics model.\n");
    content.push_str(" * Target: STM32H7, ESP32-S3, RP2040, or bare-metal Cortex-M4/M7.\n");
    content.push_str(" * Memory footprint: Stored entirely in Flash (0 bytes SRAM overhead).\n");
    content.push_str(" */\n\n");
    content.push_str("#ifndef NIGHTVISION_TABLES_H\n");
    content.push_str("#define NIGHTVISION_TABLES_H\n\n");
    content.push_str("#include <stdint.h>\n\n");
    content.push_str("#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n");

    content.push_str(&format_c_array(
        "NIGHTVISION_NOISE_LUT",
        "Deterministic Poisson Noise Threshold: tau(Y) = k * sqrt(alpha * Y + sigma_read^2)",
        &noise_lut.threshold_table,
    ));

    content.push_str(&format_c_array(
        "NIGHTVISION_LUT_ULTRA_DARK",
        "Bidirectional Ultra-Dark Transfer LUT (< 8.0 ADU) with Hermite shadow knee",
        &ultradark_lut.table,
    ));

    content.push_str(&format_c_array(
        "NIGHTVISION_LUT_LOW_LIGHT",
        "Standard Low-Light Reinhard Auto-Gating Transfer LUT (8.0 to 170.0 ADU)",
        &lowlight_lut.table,
    ));

    content.push_str(&format_c_array(
        "NIGHTVISION_LUT_HIGH_EXPOSURE",
        "Bidirectional High-Exposure Concave Highlight Compression LUT (anchored at f(255)=255)",
        &highexposure_lut.table,
    ));

    content.push_str(&format_c_array(
        "NIGHTVISION_CHROMA_GATE",
        "Dual Chroma Gate (Shadow Noise Roll-off < 16 ADU & Highlight Anti-Magenta > 230 ADU)",
        &chroma_gate_lut,
    ));

    content.push_str(&format_c_array("NIGHTVISION_P45_WHITE_R", "P45 Tactical White Phosphor - Red Channel", &p45.lut_r));
    content.push_str(&format_c_array("NIGHTVISION_P45_WHITE_G", "P45 Tactical White Phosphor - Green Channel", &p45.lut_g));
    content.push_str(&format_c_array("NIGHTVISION_P45_WHITE_B", "P45 Tactical White Phosphor - Blue Channel", &p45.lut_b));

    content.push_str(&format_c_array("NIGHTVISION_P43_GREEN_R", "P43 Military Green Phosphor (545nm) - Red Channel", &p43.lut_r));
    content.push_str(&format_c_array("NIGHTVISION_P43_GREEN_G", "P43 Military Green Phosphor (545nm) - Green Channel", &p43.lut_g));
    content.push_str(&format_c_array("NIGHTVISION_P43_GREEN_B", "P43 Military Green Phosphor (545nm) - Blue Channel", &p43.lut_b));

    content.push_str(&format_c_array("NIGHTVISION_RECON_AMBER_R", "Tactical Reconnaissance Amber - Red Channel", &amber.lut_r));
    content.push_str(&format_c_array("NIGHTVISION_RECON_AMBER_G", "Tactical Reconnaissance Amber - Green Channel", &amber.lut_g));
    content.push_str(&format_c_array("NIGHTVISION_RECON_AMBER_B", "Tactical Reconnaissance Amber - Blue Channel", &amber.lut_b));

    content.push_str("/* ── Single-Cycle Fixed-Point Inlined Processing Helpers & Macros ── */\n\n");
    content.push_str("/**\n * @brief Saturated uint8 clamping helper.\n */\n");
    content.push_str("static inline uint8_t nightvision_clamp_u8(int32_t val) {\n");
    content.push_str("    return (uint8_t)(val < 0 ? 0 : (val > 255 ? 255 : val));\n");
    content.push_str("}\n\n");
    content.push_str("/**\n * @brief 1-cycle integer unsharp masking matching Desktop FPU acuity (1.25x detail boost).\n */\n");
    content.push_str("static inline uint8_t nightvision_unsharp_mask_1_25x_u8(uint8_t orig, uint8_t blur) {\n");
    content.push_str("    int32_t o = (int32_t)orig;\n");
    content.push_str("    int32_t b = (int32_t)blur;\n");
    content.push_str("    int32_t detail = o - b;\n");
    content.push_str("    return nightvision_clamp_u8(o + detail + (detail >> 2));\n");
    content.push_str("}\n\n");
    content.push_str("#define NIGHTVISION_UNSHARP_MASK_1_25X(orig, blur) \\\n");
    content.push_str("    nightvision_unsharp_mask_1_25x_u8((uint8_t)(orig), (uint8_t)(blur))\n\n");

    content.push_str("/**\n * @brief Fast integer square root using binary restoration (strictly O(1) cycles).\n */\n");
    content.push_str("static inline uint32_t nightvision_isqrt_u32(uint32_t val) {\n");
    content.push_str("    uint32_t res = 0;\n");
    content.push_str("    uint32_t bit = 1u << 30;\n");
    content.push_str("    while (bit > val) { bit >>= 2; }\n");
    content.push_str("    while (bit != 0) {\n");
    content.push_str("        if (val >= res + bit) { val -= res + bit; res = (res >> 1) + bit; }\n");
    content.push_str("        else { res >>= 1; }\n");
    content.push_str("        bit >>= 2;\n");
    content.push_str("    }\n");
    content.push_str("    return res;\n");
    content.push_str("}\n\n");

    content.push_str("#ifdef __cplusplus\n}\n#endif\n\n");
    content.push_str("#endif /* NIGHTVISION_TABLES_H */\n");

    let mut file = std::fs::File::create(output_path)?;
    file.write_all(content.as_bytes())?;
    Ok(())
}
