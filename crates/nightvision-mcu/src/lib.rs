//! Embedded Real-Time NightVision ISP Engine for Microcontrollers.
//!
//! Designed for resource-constrained hardware (ESP32-S3, STM32H7, RP2040):
//! - Line-Buffered Streaming: < 2KB active spatial SRAM footprint (avoids 900KB full-frame buffers).
//! - Fixed-Point Integer Math: Zero floating-point division; all operations use bit-shifting (`>> 3`, `>> 2`).
//! - Precomputed 8-bit LUTs: Auto-gating, gamma, Reinhard compression, and tactical phosphors in 1-cycle lookups.
//! - Deterministic Physics Noise Profiling: Evaluates Poisson-Gaussian noise envelopes without runtime variance calculation.

pub mod deterministic_filter;
pub mod fixed_math;
pub mod line_buffer;
pub mod lut;
pub mod mcu_pipeline;
pub mod optimization_engine;

pub use deterministic_filter::{filter_line_deterministic, integer_unsharp_mask};
pub use fixed_math::{
    abs_diff_u8, ema_blend_deep_u8, ema_blend_motion_u8, ema_blend_static_u8, isqrt_u32, q8_mul,
};
pub use line_buffer::LineBuffer3;
pub use lut::{AutoGatingLut, DeterministicNoiseLut, PhosphorLut};
pub use mcu_pipeline::{McuDynamicProcessor, McuEngineConfig, McuNightVisionEngine};
pub use optimization_engine::{McuOptimizationPackage, OptimizationEngine, SensorHardwareProfile};

#[cfg(test)]
mod tests {
    use super::*;
    use nightvision_core::generate_synthetic_test_scene;

    #[test]
    fn test_mcu_fixed_math_sanity() {
        assert_eq!(isqrt_u32(144), 12);
        assert_eq!(isqrt_u32(10000), 100);
        assert_eq!(abs_diff_u8(150, 50), 100);
        assert_eq!(abs_diff_u8(50, 150), 100);

        // EMA static blend test
        let acc = ema_blend_static_u8(100, 20); // (100 + 140) >> 3 = 30
        assert_eq!(acc, 30);
    }

    #[test]
    fn test_mcu_streaming_pipeline_smoke() {
        let test_scene = generate_synthetic_test_scene(320, 240);
        let config = McuEngineConfig::default();
        let mut engine = McuNightVisionEngine::<320, 240>::new(config);

        let output = engine.process_frame_streaming(&test_scene);
        assert_eq!(output.dimensions(), (320, 240));
    }

    #[test]
    fn test_mcu_frame_processor_seam() {
        use nightvision_core::{DisplayMode, FrameProcessor};

        let test_scene = generate_synthetic_test_scene(160, 120);

        // Test static const-generic engine through FrameProcessor seam
        let mut static_engine = McuNightVisionEngine::<160, 120>::new(McuEngineConfig::default());
        static_engine.set_display_mode(DisplayMode::TacticalWhitePhosphor);
        assert_eq!(static_engine.display_mode(), DisplayMode::TacticalWhitePhosphor);

        let out1 = static_engine.process_frame(&test_scene);
        assert_eq!(out1.dimensions(), (160, 120));

        // Test dynamic adapter through FrameProcessor seam
        let mut dyn_engine = McuDynamicProcessor::new(McuEngineConfig::default());
        dyn_engine.set_display_mode(DisplayMode::TacticalGreenPhosphor);
        assert_eq!(dyn_engine.display_mode(), DisplayMode::TacticalGreenPhosphor);

        let mut out2 = image::RgbImage::new(160, 120);
        dyn_engine.process_frame_into(&test_scene, &mut out2);
        assert_eq!(out2.dimensions(), (160, 120));
    }
}

