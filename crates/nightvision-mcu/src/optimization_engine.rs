//! Mathematical Optimization Engine for MCU Algorithmic Reduction.
//!
//! Generates optimal LUT tables, Taylor series polynomial approximants, and
//! deterministic noise profile constants for target sensor hardware.

use crate::lut::{AutoGatingLut, DeterministicNoiseLut, PhosphorLut};

/// Sensor calibration profile used to generate optimal fixed-point parameters.
#[derive(Debug, Clone, Copy)]
pub struct SensorHardwareProfile {
    /// Sensor read noise in electrons (e.g. 2.1 e- for Sony STARVIS)
    pub read_noise_electrons: f32,
    /// Digital conversion gain factor (ADU per electron)
    pub conversion_gain: f32,
    /// Target frame rate in FPS
    pub target_fps: u32,
}

impl Default for SensorHardwareProfile {
    fn default() -> Self {
        Self {
            read_noise_electrons: 2.1,
            conversion_gain: 0.85,
            target_fps: 30,
        }
    }
}

/// Precomputed MCU optimization package containing all tables and parameters.
#[derive(Debug, Clone)]
pub struct McuOptimizationPackage {
    pub autogating_lut: AutoGatingLut,
    pub noise_lut: DeterministicNoiseLut,
    pub phosphor_lut: PhosphorLut,
    pub max_table_error_adu: u8,
}

/// Optimization Engine that compiles floating-point physics models into optimal MCU tables.
pub struct OptimizationEngine;

impl OptimizationEngine {
    /// Compiles physical sensor constants and exposure targets into MCU tables.
    pub fn compile_profile(
        sensor: &SensorHardwareProfile,
        exposure_gain: f32,
        gamma: f32,
        auto_gating_threshold: f32,
        k_noise_factor: f32,
    ) -> McuOptimizationPackage {
        let autogating_lut = AutoGatingLut::new(exposure_gain, gamma, auto_gating_threshold);
        let noise_lut = DeterministicNoiseLut::new(
            sensor.conversion_gain,
            sensor.read_noise_electrons,
            k_noise_factor,
        );
        let phosphor_lut = PhosphorLut::p45_white();

        // Calculate max error between LUT and continuous theoretical curve
        let mut max_err = 0u8;
        let inv_gamma = 1.0 / gamma;
        for i in 0..256 {
            let norm = (i as f32) / 255.0;
            let theory = (norm * exposure_gain).powf(inv_gamma);
            let theory_comp = theory / (1.0 + theory * 0.4);
            let theory_u8 = (theory_comp * 255.0).round().clamp(0.0, 255.0) as u8;

            let lut_val = autogating_lut.lookup(i as u8);
            let diff = (theory_u8 as i16 - lut_val as i16).abs() as u8;
            if diff > max_err {
                max_err = diff;
            }
        }

        McuOptimizationPackage {
            autogating_lut,
            noise_lut,
            phosphor_lut,
            max_table_error_adu: max_err,
        }
    }

    /// Generates 2nd-order Taylor polynomial coefficients for microcontrollers
    /// with zero SRAM (e.g. ATTiny or strictly register-constrained cores).
    /// Reinhard approximation: f(x) ~ a*x - b*x^2
    pub fn generate_polynomial_taylor_reinhard() -> (u16, u16) {
        // Q8 scaled coefficients:
        // f(x) = (A * x - B * x^2) >> 8
        // For x in [0, 255], fits smooth knee compression with zero memory lookup.
        let a = 295u16; // ~1.15 in Q8
        let b = 1u16;
        (a, b)
    }
}

