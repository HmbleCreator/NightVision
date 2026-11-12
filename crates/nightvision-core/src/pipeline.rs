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
}
