//! Embedded Real-Time NightVision ISP Engine for Microcontrollers.
//!
//! Designed for resource-constrained hardware (ESP32-S3, STM32H7, RP2040):
//! - Line-Buffered Streaming: < 2KB active spatial SRAM footprint (avoids 900KB full-frame buffers).
//! - Fixed-Point Integer Math: Zero floating-point division; all operations use bit-shifting (`>> 3`, `>> 2`).
//! - Precomputed 8-bit LUTs: Auto-gating, gamma, Reinhard compression, and tactical phosphors in 1-cycle lookups.
//! - Deterministic Physics Noise Profiling: Evaluates Poisson-Gaussian noise envelopes without runtime variance calculation.

pub mod deterministic_filter;
pub mod dma;
pub mod ffi;
pub mod fixed_math;
pub mod line_buffer;
pub mod lut;
pub mod mcu_pipeline;
pub mod optimization_engine;

pub use deterministic_filter::{filter_line_deterministic, integer_unsharp_mask};
pub use dma::{DmaBank, DmaPingPongBuffer};
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

    #[test]
    fn test_dma_ping_pong_buffer() {
        let mut dma_buf = DmaPingPongBuffer::<64>::new();

        // 1. Fill Bank A via DMA
        let rx_a = dma_buf.dma_rx_buffer_mut();
        rx_a.fill(42);

        // 2. Swap banks upon DMA Transfer Complete interrupt
        dma_buf.swap_banks();

        // 3. CPU ISP processes Bank A while DMA fills Bank B
        assert_eq!(dma_buf.processing_buffer()[0], 42);

        let rx_b = dma_buf.dma_rx_buffer_mut();
        rx_b.fill(99);

        dma_buf.swap_banks();
        assert_eq!(dma_buf.processing_buffer()[0], 99);
    }

    #[test]
    fn test_ffi_c_seam() {
        use std::ffi::CStr;

        let version = unsafe { CStr::from_ptr(ffi::nightvision_version()) };
        assert!(version.to_str().unwrap().contains("NightVision Embedded"));

        unsafe {
            let handle = ffi::nightvision_create(320, 240, 1);
            assert!(!handle.is_null());

            ffi::nightvision_set_display_mode(handle, 2);
            ffi::nightvision_reset(handle);

            let in_bytes = vec![20u8; 320 * 240 * 3];
            let mut out_bytes = vec![0u8; 320 * 240 * 3];

            let res = ffi::nightvision_process_frame(
                handle,
                320,
                240,
                in_bytes.as_ptr(),
                out_bytes.as_mut_ptr(),
            );
            assert_eq!(res, 0);
            assert!(out_bytes[0] > 0);

            // Test line streaming C-FFI
            let top = [10u8; 64];
            let mid = [12u8; 64];
            let bot = [11u8; 64];
            let mut out_line = [0u8; 64];

            let line_res = ffi::nightvision_process_line_luma(
                top.as_ptr(),
                mid.as_ptr(),
                bot.as_ptr(),
                out_line.as_mut_ptr(),
                64,
                std::ptr::null(),
                std::ptr::null(),
            );
            assert_eq!(line_res, 0);

            // Test C-FFI AEC exposure report retrieval
            let mut rep = ffi::nightvision_exposure_report_t {
                mean_luma: 0.0,
                p95_luma: 0.0,
                saturated_pixel_fraction: 0.0,
                regime: 0,
                recommended_ev_shift: 0.0,
            };
            let rep_res = ffi::nightvision_get_exposure_report(handle, &mut rep);
            assert_eq!(rep_res, 0);
            assert!(rep.mean_luma > 0.0);

            ffi::nightvision_destroy(handle);
        }
    }

    #[test]
    fn test_mcu_bidirectional_hdr_and_chroma_gate() {
        use crate::lut::AutoGatingLut;
        use nightvision_core::DisplayMode;

        // 1. Verify AutoGatingLut across regimes
        let lut_dark = AutoGatingLut::new_bidirectional(2.8, 0.60, 0.88, 5.0, 30.0, 0.0);
        let lut_norm = AutoGatingLut::new_bidirectional(1.5, 0.80, 0.88, 80.0, 160.0, 0.0);
        let lut_high = AutoGatingLut::new_bidirectional(1.0, 1.00, 0.88, 220.0, 250.0, 0.25);

        // Deep shadow code (10 ADU) lifted higher in dark regime than normal
        assert!(lut_dark.lookup(10) > lut_norm.lookup(10));
        // High exposure knee smoothly rolls off top codes
        assert!(lut_high.lookup(255) >= 240);
        assert!(lut_high.lookup(200) < lut_high.lookup(255));

        // 2. Test MCU processor on extreme saturation (Y = 255)
        let config = McuEngineConfig::default().with_mode(DisplayMode::FullColor);
        let mut proc = McuDynamicProcessor::new(config);
        let mut sat_frame = image::RgbImage::new(64, 64);
        for p in sat_frame.pixels_mut() {
            *p = image::Rgb([255, 255, 255]);
        }
        let out_sat = proc.process_frame(&sat_frame);
        let rep = proc.exposure_report();
        assert_eq!(rep.regime, nightvision_core::SceneLightingRegime::FullSaturation);
        assert!(rep.saturated_pixel_fraction > 0.90);

        // Verify Dual Chroma Gate: saturated pixel is neutral white, no color tint
        let p = out_sat.get_pixel(32, 32);
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
        assert!(p[0] >= 240);
    }
}

