//! End-to-end Digital Night Vision pipeline orchestrator.
//!
//! Connects the complete computational imaging chain:
//! Frame In -> Temporal Accumulation -> Edge-Preserving Denoising -> Retinex / CLAHE -> Auto-Gating -> Tactical Mode Out.

use image::RgbImage;
use serde::{Deserialize, Serialize};

use crate::enhancement::{
    apply_clahe, apply_exposure_and_autogating, apply_retinex, EnhancementConfig,
};
use crate::modes::{render_mode, DisplayMode};
use crate::processor::FrameProcessor;
use crate::spatial::{denoise_image, SpatialDenoiseConfig};
use crate::temporal::{AdaptiveTemporalAccumulator, TemporalConfig};

/// Complete pipeline configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub temporal: TemporalConfig,
    pub spatial: SpatialDenoiseConfig,
    pub enhancement: EnhancementConfig,
    pub mode: DisplayMode,
    pub saturation_boost: f32,
    pub enable_temporal: bool,
    pub enable_spatial: bool,
    pub enable_retinex: bool,
    pub enable_clahe: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            temporal: TemporalConfig::default(),
            spatial: SpatialDenoiseConfig::default(),
            enhancement: EnhancementConfig::default(),
            mode: DisplayMode::FullColor,
            saturation_boost: 1.35,
            enable_temporal: true,
            enable_spatial: true,
            enable_retinex: true,
            enable_clahe: true,
        }
    }
}

impl PipelineConfig {
    /// High-throughput real-time preset optimized for sustained 30+ FPS tactical streams.
    /// Uses low-overhead guided spatial filtering, temporal accumulation, and adaptive tone curves.
    pub fn fast_realtime() -> Self {
        Self {
            temporal: TemporalConfig {
                static_blend_weight: 0.90,
                motion_blend_weight: 0.10,
                motion_threshold: 0.08,
                enable_motion_dilation: false,
            },
            spatial: SpatialDenoiseConfig {
                guided_radius: 2,
                guided_epsilon: 0.015,
                chroma_denoise_strength: 0.70,
                iterations: 1,
            },
            enhancement: EnhancementConfig {
                retinex_weight: 0.0,
                clahe_clip_limit: 2.5,
                clahe_grid_size: 6,
                gamma: 0.60,
                exposure_gain: 2.8,
                auto_gating_threshold: 0.88,
            },
            mode: DisplayMode::TacticalWhitePhosphor,
            saturation_boost: 1.0,
            enable_temporal: true,
            enable_spatial: true,
            enable_retinex: false,
            enable_clahe: true,
        }
    }
}

/// Execution latency breakdown per pipeline stage.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct StageBreakdown {
    pub temporal_ms: f64,
    pub spatial_ms: f64,
    pub retinex_ms: f64,
    pub clahe_ms: f64,
    pub exposure_autogating_ms: f64,
    pub mode_rendering_ms: f64,
    pub total_ms: f64,
}

/// Integrated Digital Night Vision Pipeline.
pub struct NightVisionPipeline {
    pub config: PipelineConfig,
    temporal_accumulator: AdaptiveTemporalAccumulator,
}

impl NightVisionPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let temporal_accumulator = AdaptiveTemporalAccumulator::new(config.temporal.clone());
        Self {
            config,
            temporal_accumulator,
        }
    }

    /// Reset internal state (buffers, motion tracking).
    pub fn reset(&mut self) {
        self.temporal_accumulator.reset();
    }

    /// Processes an individual frame through the entire computational night vision pipeline.
    pub fn process_frame(&mut self, frame: &RgbImage) -> RgbImage {
        // Step 1: Multi-frame temporal accumulation (SNR boost)
        let frame_after_temporal = if self.config.enable_temporal {
            self.temporal_accumulator.config = self.config.temporal.clone();
            self.temporal_accumulator.process_frame(frame)
        } else {
            frame.clone()
        };

        // Step 2: Edge-preserving spatial denoising & chroma despeckle
        let frame_after_spatial = if self.config.enable_spatial {
            denoise_image(&frame_after_temporal, &self.config.spatial)
        } else {
            frame_after_temporal
        };

        // Step 3: Multi-Scale Retinex enhancement
        let frame_after_retinex = if self.config.enable_retinex {
            apply_retinex(
                &frame_after_spatial,
                self.config.enhancement.retinex_weight,
            )
        } else {
            frame_after_spatial
        };

        // Step 4: CLAHE local contrast enhancement
        let frame_after_clahe = if self.config.enable_clahe {
            apply_clahe(
                &frame_after_retinex,
                self.config.enhancement.clahe_grid_size,
                self.config.enhancement.clahe_clip_limit,
            )
        } else {
            frame_after_retinex
        };

        // Step 5: Adaptive exposure boost, gamma lift, and flare protection (auto-gating)
        let frame_after_exposure = apply_exposure_and_autogating(
            &frame_after_clahe,
            self.config.enhancement.exposure_gain,
            self.config.enhancement.gamma,
            self.config.enhancement.auto_gating_threshold,
        );

        // Step 6: Render in selected tactical display mode (Full-Color, White Phosphor, Green Phosphor, etc.)
        render_mode(
            &frame_after_exposure,
            self.config.mode,
            self.config.saturation_boost,
        )
    }

    /// Processes frame and measures execution latency for each algorithmic stage.
    pub fn process_frame_profiled(&mut self, frame: &RgbImage) -> (RgbImage, StageBreakdown) {
        use std::time::Instant;

        let total_start = Instant::now();

        // 1. Temporal
        let t0 = Instant::now();
        let frame_after_temporal = if self.config.enable_temporal {
            self.temporal_accumulator.config = self.config.temporal.clone();
            self.temporal_accumulator.process_frame(frame)
        } else {
            frame.clone()
        };
        let temporal_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // 2. Spatial
        let t1 = Instant::now();
        let frame_after_spatial = if self.config.enable_spatial {
            denoise_image(&frame_after_temporal, &self.config.spatial)
        } else {
            frame_after_temporal
        };
        let spatial_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // 3. Retinex
        let t2 = Instant::now();
        let frame_after_retinex = if self.config.enable_retinex {
            apply_retinex(
                &frame_after_spatial,
                self.config.enhancement.retinex_weight,
            )
        } else {
            frame_after_spatial
        };
        let retinex_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 4. CLAHE
        let t3 = Instant::now();
        let frame_after_clahe = if self.config.enable_clahe {
            apply_clahe(
                &frame_after_retinex,
                self.config.enhancement.clahe_grid_size,
                self.config.enhancement.clahe_clip_limit,
            )
        } else {
            frame_after_retinex
        };
        let clahe_ms = t3.elapsed().as_secs_f64() * 1000.0;

        // 5. Exposure & Auto-Gating
        let t4 = Instant::now();
        let frame_after_exposure = apply_exposure_and_autogating(
            &frame_after_clahe,
            self.config.enhancement.exposure_gain,
            self.config.enhancement.gamma,
            self.config.enhancement.auto_gating_threshold,
        );
        let exposure_autogating_ms = t4.elapsed().as_secs_f64() * 1000.0;

        // 6. Mode rendering
        let t5 = Instant::now();
        let output = render_mode(
            &frame_after_exposure,
            self.config.mode,
            self.config.saturation_boost,
        );
        let mode_rendering_ms = t5.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let breakdown = StageBreakdown {
            temporal_ms,
            spatial_ms,
            retinex_ms,
            clahe_ms,
            exposure_autogating_ms,
            mode_rendering_ms,
            total_ms,
        };

        (output, breakdown)
    }
}

impl FrameProcessor for NightVisionPipeline {
    fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage) {
        let (w, h) = input.dimensions();
        if output.dimensions() != (w, h) {
            *output = image::ImageBuffer::new(w, h);
        }
        let enhanced = self.process_frame(input);
        output.copy_from_slice(enhanced.as_raw());
    }

    fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        self.process_frame(input)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.config.mode = mode;
    }

    fn display_mode(&self) -> DisplayMode {
        self.config.mode
    }

    fn backend_name(&self) -> &'static str {
        "Desktop Multi-Core Rayon (f32 FPU)"
    }
}

