use std::time::Instant;

use eframe::egui;
use image::{ImageBuffer, Rgb, RgbImage};
use nightvision_core::{
    calculate_psnr, calculate_ssim, generate_synthetic_test_scene, CmosSensorConfig,
    CmosSensorSimulator, DisplayMode, LuxLevel, NightVisionPipeline, PipelineConfig,
    SensorSimulationStats,
};

#[derive(PartialEq, Clone, Copy)]
enum ViewMode {
    SideBySide,
    EnhancedOnly,
    RawOnly,
}

struct NightVisionApp {
    sensor_sim: CmosSensorSimulator,
    pipeline: NightVisionPipeline,

    // Scene & Sensor State
    reference_scene: RgbImage,
    current_raw_frame: RgbImage,
    current_enhanced_frame: RgbImage,
    sensor_stats: Option<SensorSimulationStats>,

    // Controls
    lux_slider: f64,
    exposure_gain: f32,
    gamma: f32,
    temporal_weight: f32,
    guided_radius: usize,
    clahe_clip: f32,
    auto_gating_thresh: f32,
    display_mode: DisplayMode,
    view_mode: ViewMode,
    continuous_stream: bool,

    // Metrics
    last_psnr: f64,
    last_ssim: f64,
    last_frame_ms: f64,

    // Egui texture handles
    preview_texture: Option<egui::TextureHandle>,
}

impl Default for NightVisionApp {
    fn default() -> Self {
        let reference_scene = generate_synthetic_test_scene(480, 320);
        let sensor_sim = CmosSensorSimulator::new(CmosSensorConfig::default());
        let pipeline = NightVisionPipeline::new(PipelineConfig::default());

        let mut app = Self {
            sensor_sim,
            pipeline,
            reference_scene: reference_scene.clone(),
            current_raw_frame: reference_scene.clone(),
            current_enhanced_frame: reference_scene,
            sensor_stats: None,

            lux_slider: 0.001, // Starlight by default
            exposure_gain: 2.8,
            gamma: 0.58,
            temporal_weight: 0.92,
            guided_radius: 3,
            clahe_clip: 3.2,
            auto_gating_thresh: 0.88,
            display_mode: DisplayMode::FullColor,
            view_mode: ViewMode::SideBySide,
            continuous_stream: false,

            last_psnr: 0.0,
            last_ssim: 0.0,
            last_frame_ms: 0.0,
            preview_texture: None,
        };

        app.trigger_capture_and_enhance();
        app
    }
}

impl NightVisionApp {
    fn trigger_capture_and_enhance(&mut self) {
        let start = Instant::now();

        // 1. Physical sensor capture simulation
        let lux_level = LuxLevel::Custom(self.lux_slider);
        let (raw, stats) = self
            .sensor_sim
            .simulate_low_light(&self.reference_scene, lux_level, None);
        self.current_raw_frame = raw.clone();
        self.sensor_stats = Some(stats);

        // 2. Synchronize pipeline config
        self.pipeline.config.mode = self.display_mode;
        self.pipeline.config.enhancement.exposure_gain = self.exposure_gain;
        self.pipeline.config.enhancement.gamma = self.gamma;
        self.pipeline.config.temporal.static_blend_weight = self.temporal_weight;
        self.pipeline.config.spatial.guided_radius = self.guided_radius;
        self.pipeline.config.enhancement.clahe_clip_limit = self.clahe_clip;
        self.pipeline.config.enhancement.auto_gating_threshold = self.auto_gating_thresh;

        // 3. Process through NightVision computational pipeline
        self.current_enhanced_frame = self.pipeline.process_frame(&raw);

        self.last_psnr = calculate_psnr(&self.reference_scene, &self.current_enhanced_frame);
        self.last_ssim = calculate_ssim(&self.reference_scene, &self.current_enhanced_frame);
        self.last_frame_ms = start.elapsed().as_secs_f64() * 1000.0;
    }

    fn composite_view_image(&self) -> RgbImage {
        match self.view_mode {
            ViewMode::EnhancedOnly => self.current_enhanced_frame.clone(),
            ViewMode::RawOnly => self.current_raw_frame.clone(),
            ViewMode::SideBySide => {
                let (w, h) = self.current_raw_frame.dimensions();
                let sep = 4u32;
                let mut composite = ImageBuffer::new(w * 2 + sep, h);

                for y in 0..h {
                    for x in 0..w {
                        composite.put_pixel(x, y, *self.current_raw_frame.get_pixel(x, y));
                    }
                }
                for y in 0..h {
                    for x in 0..sep {
                        composite.put_pixel(w + x, y, Rgb([60, 60, 75]));
                    }
                }
                for y in 0..h {
                    for x in 0..w {
                        composite.put_pixel(
                            w + sep + x,
                            y,
                            *self.current_enhanced_frame.get_pixel(x, y),
                        );
                    }
                }
                composite
            }
        }
    }
}

impl eframe::App for NightVisionApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.continuous_stream {
            self.trigger_capture_and_enhance();
            ctx.request_repaint();
        }

        // Top Header
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🌙 NIGHTVISION : Digital Computational Night Vision Engine");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.continuous_stream {
                        ui.colored_label(egui::Color32::from_rgb(50, 220, 50), "● STREAMING ACTIVE");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○ IDLE");
                    }
                });
            });
        });

        // Left Control Panel
        egui::SidePanel::left("controls")
            .min_width(320.0)
            .show(ctx, |ui| {
                ui.spacing_mut().item_spacing.y = 8.0;

                ui.heading("Illuminance & Sensor");
                ui.horizontal(|ui| {
                    if ui.button("Overcast (0.0001 lx)").clicked() {
                        self.lux_slider = 0.0001;
                        self.trigger_capture_and_enhance();
                    }
                    if ui.button("Starlight (0.001 lx)").clicked() {
                        self.lux_slider = 0.001;
                        self.trigger_capture_and_enhance();
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Quarter Moon (0.01 lx)").clicked() {
                        self.lux_slider = 0.01;
                        self.trigger_capture_and_enhance();
                    }
                    if ui.button("Twilight (1.0 lx)").clicked() {
                        self.lux_slider = 1.0;
                        self.trigger_capture_and_enhance();
                    }
                });

                ui.add(
                    egui::Slider::new(&mut self.lux_slider, 0.00005..=2.0)
                        .logarithmic(true)
                        .text("Lux"),
                );

                ui.separator();

                ui.heading("Tactical Display Mode");
                ui.radio_value(
                    &mut self.display_mode,
                    DisplayMode::FullColor,
                    "🌈 Full-Color Night Vision",
                );
                ui.radio_value(
                    &mut self.display_mode,
                    DisplayMode::TacticalWhitePhosphor,
                    "⚪ Tactical White Phosphor (P45)",
                );
                ui.radio_value(
                    &mut self.display_mode,
                    DisplayMode::TacticalGreenPhosphor,
                    "🟢 Tactical Green Phosphor (P43)",
                );
                ui.radio_value(
                    &mut self.display_mode,
                    DisplayMode::HighContrastRecon,
                    "🟡 High-Contrast Recon (Text/Edges)",
                );
                ui.radio_value(
                    &mut self.display_mode,
                    DisplayMode::ThermalFusion,
                    "🔥 Thermal Fusion (Ironbow)",
                );

                ui.separator();

                ui.heading("Pipeline Tuning");
                ui.add(
                    egui::Slider::new(&mut self.exposure_gain, 1.0..=10.0)
                        .text("Exposure Gain"),
                );
                ui.add(egui::Slider::new(&mut self.gamma, 0.3..=1.2).text("Adaptive Gamma"));
                ui.add(
                    egui::Slider::new(&mut self.temporal_weight, 0.0..=0.98)
                        .text("Temporal Accumulation"),
                );
                ui.add(egui::Slider::new(&mut self.guided_radius, 1..=8).text("Guided Radius"));
                ui.add(egui::Slider::new(&mut self.clahe_clip, 1.0..=8.0).text("CLAHE Clip"));
                ui.add(
                    egui::Slider::new(&mut self.auto_gating_thresh, 0.5..=1.0)
                        .text("Auto-Gating"),
                );

                ui.separator();

                ui.horizontal(|ui| {
                    if ui.button("📸 Single Capture").clicked() {
                        self.trigger_capture_and_enhance();
                    }
                    if ui
                        .button(if self.continuous_stream {
                            "⏹ Stop Stream"
                        } else {
                            "▶ Stream Live"
                        })
                        .clicked()
                    {
                        self.continuous_stream = !self.continuous_stream;
                    }
                    if ui.button("↺ Reset").clicked() {
                        self.pipeline.reset();
                        self.trigger_capture_and_enhance();
                    }
                });
            });

        // Bottom Telemetry Panel
        egui::TopBottomPanel::bottom("telemetry").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(stats) = &self.sensor_stats {
                    ui.label(format!(
                        "Photons/px: {:.2} e- | Read Noise: {:.2} e- | Gain: {:.1}x (ISO)",
                        stats.mean_photons_per_pixel,
                        stats.read_noise_sigma,
                        stats.digital_gain_applied
                    ));
                    ui.separator();
                }
                ui.label(format!("PSNR: {:.2} dB", self.last_psnr));
                ui.separator();
                ui.label(format!("SSIM: {:.4}", self.last_ssim));
                ui.separator();
                ui.label(format!(
                    "Latency: {:.1} ms ({:.1} FPS)",
                    self.last_frame_ms,
                    if self.last_frame_ms > 0.0 {
                        1000.0 / self.last_frame_ms
                    } else {
                        0.0
                    }
                ));
            });
        });

        // Central Viewer Panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Display Canvas:");
                ui.radio_value(&mut self.view_mode, ViewMode::SideBySide, "Split (Raw vs Enhanced)");
                ui.radio_value(&mut self.view_mode, ViewMode::EnhancedOnly, "Enhanced Only");
                ui.radio_value(&mut self.view_mode, ViewMode::RawOnly, "Raw Sensor Only");
            });

            ui.separator();

            let composite = self.composite_view_image();
            let (w, h) = composite.dimensions();

            let color_image = egui::ColorImage::from_rgb(
                [w as usize, h as usize],
                composite.as_flat_samples().samples,
            );

            let texture = self
                .preview_texture
                .get_or_insert_with(|| {
                    ctx.load_texture("view_texture", color_image.clone(), egui::TextureOptions::LINEAR)
                });
            texture.set(color_image, egui::TextureOptions::LINEAR);

            let available_size = ui.available_size();
            let aspect = (w as f32) / (h as f32);
            let target_w = available_size.x.min(available_size.y * aspect);
            let target_h = target_w / aspect;

            ui.image((texture.id(), egui::vec2(target_w, target_h)));
        });
    }
}

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1120.0, 720.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("NightVision Tactical Dashboard"),
        ..Default::default()
    };

    eframe::run_native(
        "NightVision Dashboard",
        native_options,
        Box::new(|_cc| Ok(Box::new(NightVisionApp::default()))),
    )
}
