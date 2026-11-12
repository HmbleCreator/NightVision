//! Physics-based low-light CMOS sensor simulator.
//!
//! Accurately models the physical degradation of digital sensors in near-total darkness:
//! - Photon arrival statistics (Poisson shot noise)
//! - Sensor quantum efficiency (QE) and pixel active area
//! - Dark current thermal noise
//! - Read noise (Gaussian ADC & amplifier noise)
//! - Analog & digital gain amplification (ISO)
//! - Quantization and dynamic range saturation

use image::{ImageBuffer, Rgb, RgbImage};
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::{Normal, Poisson};
use serde::{Deserialize, Serialize};

/// Standard environmental illuminance presets in Lux.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LuxLevel {
    /// Overcast moonless night (near-total pitch black) ~ 0.0001 lux
    OvercastNight,
    /// Clear starlight night (moonless) ~ 0.001 lux
    Starlight,
    /// Quarter moon illumination ~ 0.01 lux
    QuarterMoon,
    /// Full moon illumination ~ 0.1 lux
    FullMoon,
    /// Deep civil twilight ~ 1.0 lux
    Twilight,
    /// Dim indoor lighting ~ 10.0 lux
    IndoorDim,
    /// Custom illuminance in lux
    Custom(f64),
}

impl LuxLevel {
    pub fn as_lux(&self) -> f64 {
        match *self {
            LuxLevel::OvercastNight => 0.0001,
            LuxLevel::Starlight => 0.001,
            LuxLevel::QuarterMoon => 0.01,
            LuxLevel::FullMoon => 0.1,
            LuxLevel::Twilight => 1.0,
            LuxLevel::IndoorDim => 10.0,
            LuxLevel::Custom(val) => val.max(1e-6),
        }
    }

    pub fn description(&self) -> &'static str {
        match *self {
            LuxLevel::OvercastNight => "Overcast Night (~0.0001 lux, extreme photon starvation)",
            LuxLevel::Starlight => "Clear Starlight (~0.001 lux, heavy sensor noise)",
            LuxLevel::QuarterMoon => "Quarter Moon (~0.01 lux, deep shadows)",
            LuxLevel::FullMoon => "Full Moon (~0.1 lux, low light)",
            LuxLevel::Twilight => "Twilight (~1.0 lux, dusk)",
            LuxLevel::IndoorDim => "Indoor Dim (~10.0 lux)",
            LuxLevel::Custom(_) => "Custom Lux Level",
        }
    }
}

/// Physical parameters of a CMOS image sensor (e.g. Sony STARVIS / IMX series).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmosSensorConfig {
    /// Quantum Efficiency (fraction of incident photons converted to photoelectrons, 0.0 - 1.0)
    pub quantum_efficiency: f64,
    /// Physical pixel area in square micrometers (e.g., 2.9um x 2.9um = 8.41 um^2)
    pub pixel_area_um2: f64,
    /// Exposure time in seconds (e.g., 1/30s = 0.0333s)
    pub exposure_time_sec: f64,
    /// Read noise standard deviation in electrons (e-)
    pub read_noise_electrons: f64,
    /// Dark current generation rate in electrons per second per pixel
    pub dark_current_e_per_sec: f64,
    /// Full well capacity in electrons before pixel saturation
    pub full_well_capacity: f64,
    /// Base ISO calibration factor
    pub base_iso_scale: f64,
}

impl Default for CmosSensorConfig {
    fn default() -> Self {
        Self {
            quantum_efficiency: 0.72,
            pixel_area_um2: 8.41,          // 2.9um x 2.9um sensor pixel (typical high-end security/action CMOS)
            exposure_time_sec: 1.0 / 30.0, // 30 FPS video
            read_noise_electrons: 2.1,     // Modern back-illuminated CMOS read noise
            dark_current_e_per_sec: 0.8,   // Low dark current at 25°C
            full_well_capacity: 18000.0,   // 18k e- full well
            base_iso_scale: 1.0,
        }
    }
}

/// Statistics reported from sensor simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSimulationStats {
    pub mean_photons_per_pixel: f64,
    pub photon_shot_noise_sigma: f64,
    pub read_noise_sigma: f64,
    pub estimated_snr_db: f64,
    pub digital_gain_applied: f64,
}

/// Physical CMOS Sensor Simulator.
#[derive(Debug, Clone)]
pub struct CmosSensorSimulator {
    pub config: CmosSensorConfig,
}

impl CmosSensorSimulator {
    pub fn new(config: CmosSensorConfig) -> Self {
        Self { config }
    }

    /// Simulates capturing a clean ground-truth scene under extreme low-light conditions.
    ///
    /// Computes incident photon flux per pixel based on scene luminance and target lux level,
    /// samples Poisson arrivals, adds dark current and Gaussian read noise, applies ISO amplifier gain,
    /// and quantizes to 8-bit digital output.
    pub fn simulate_low_light(
        &self,
        clean_rgb: &RgbImage,
        lux: LuxLevel,
        target_iso_gain: Option<f64>,
    ) -> (RgbImage, SensorSimulationStats) {
        let (width, height) = clean_rgb.dimensions();
        let lux_val = lux.as_lux();

        // 1 Lux of green light (555 nm) corresponds to ~ 4.09 x 10^15 photons / (s * m^2).
        // For a pixel of area A_um2: Photons/sec = Lux * (A_um2 * 1e-12) * 4.09e15 = Lux * A_um2 * 4090.0
        let photons_per_lux_sec_pixel = self.config.pixel_area_um2 * 4090.0;
        let nominal_photons_max =
            lux_val * photons_per_lux_sec_pixel * self.config.exposure_time_sec;

        // Auto-compute ISO gain needed to stretch the weak signal to a visible digital range,
        // mirroring what a real camera's Auto-Exposure / Auto-Gain does in extreme darkness.
        let auto_gain = if nominal_photons_max > 0.0 {
            (self.config.full_well_capacity * 0.7 / nominal_photons_max).clamp(1.0, 15000.0)
        } else {
            1000.0
        };
        let gain = target_iso_gain.unwrap_or(auto_gain);

        let dark_current_electrons =
            self.config.dark_current_e_per_sec * self.config.exposure_time_sec;
        let read_noise_dist = Normal::new(0.0, self.config.read_noise_electrons)
            .unwrap_or_else(|_| Normal::new(0.0, 1.0).unwrap());

        let mut noisy_image = ImageBuffer::new(width, height);
        let mut rng = thread_rng();

        let mut total_photons = 0.0;
        let total_pixels = (width * height) as f64;

        for y in 0..height {
            for x in 0..width {
                let p = clean_rgb.get_pixel(x, y);
                let mut out_rgb = [0u8; 3];

                for c in 0..3 {
                    let normalized_intensity = (p[c] as f64) / 255.0;

                    // Average incident photons on this color channel
                    let lambda_photons = (nominal_photons_max * normalized_intensity)
                        * self.config.quantum_efficiency;

                    // Poisson photon arrival
                    let sampled_photons = if lambda_photons > 0.0 {
                        if let Ok(poisson) = Poisson::new(lambda_photons) {
                            poisson.sample(&mut rng)
                        } else {
                            lambda_photons
                        }
                    } else {
                        0.0
                    };

                    total_photons += sampled_photons;

                    // Dark current (thermal noise)
                    let sampled_dark = if dark_current_electrons > 0.0 {
                        if let Ok(poisson_dark) = Poisson::new(dark_current_electrons) {
                            poisson_dark.sample(&mut rng)
                        } else {
                            dark_current_electrons
                        }
                    } else {
                        0.0
                    };

                    // Sensor read noise (Gaussian)
                    let read_noise = read_noise_dist.sample(&mut rng);

                    // Total electron charge at pixel node
                    let total_charge = sampled_photons + sampled_dark + read_noise;

                    // Analog-to-digital conversion with gain & full-well clamping
                    let amplified = total_charge * gain;
                    let normalized_dn = amplified / self.config.full_well_capacity;
                    let dn_8bit = (normalized_dn * 255.0).round().clamp(0.0, 255.0) as u8;

                    out_rgb[c] = dn_8bit;
                }

                noisy_image.put_pixel(x, y, Rgb(out_rgb));
            }
        }

        let mean_photons = total_photons / (total_pixels * 3.0);
        let shot_noise_sigma = mean_photons.sqrt();
        let read_noise_sigma = self.config.read_noise_electrons;
        let total_noise = (shot_noise_sigma.powi(2) + read_noise_sigma.powi(2)).sqrt();
        let snr_db = if total_noise > 0.0 && mean_photons > 0.0 {
            20.0 * (mean_photons / total_noise).log10()
        } else {
            -20.0
        };

        let stats = SensorSimulationStats {
            mean_photons_per_pixel: mean_photons,
            photon_shot_noise_sigma: shot_noise_sigma,
            read_noise_sigma,
            estimated_snr_db: snr_db,
            digital_gain_applied: gain,
        };

        (noisy_image, stats)
    }
}

/// Scene lighting regime detected by Auto-Exposure Control (AEC).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneLightingRegime {
    /// Extreme photon starvation (< 8.0 ADU mean) requiring Hermite shadow-knee expansion.
    UltraDark,
    /// Dim night illumination (8.0 to 45.0 ADU mean).
    LowLight,
    /// Balanced standard scene (45.0 to 170.0 ADU mean).
    NormalDaylight,
    /// Glare, strong backlighting, or high dynamic range highlights (> 170.0 ADU or p95 > 225.0 ADU).
    HighExposure,
    /// Severe photodiode well saturation (> 30% pixels clipped at 254..255 ADU).
    FullSaturation,
}

/// Comprehensive Auto-Exposure Control (AEC) and scene metering report.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExposureMeteringReport {
    /// Scene mean luminance (0.0 to 255.0 ADU)
    pub mean_luma: f32,
    /// 95th percentile luminance (0.0 to 255.0 ADU)
    pub p95_luma: f32,
    /// Fraction of pixels reaching photodiode saturation (250..255 ADU)
    pub saturated_pixel_fraction: f32,
    /// Detected scene lighting regime
    pub regime: SceneLightingRegime,
    /// Recommended Electronic Shutter / Analog Gain EV shift (e.g. -4.0 EV for blinding sun, +3.0 EV for starlight)
    pub recommended_ev_shift: f32,
}

impl Default for ExposureMeteringReport {
    fn default() -> Self {
        Self {
            mean_luma: 16.0,
            p95_luma: 64.0,
            saturated_pixel_fraction: 0.0,
            regime: SceneLightingRegime::LowLight,
            recommended_ev_shift: 2.0,
        }
    }
}

impl ExposureMeteringReport {
    /// Computes exposure metrics, lighting regime, and recommended EV compensation.
    pub fn compute(mean_luma: f32, p95_luma: f32, saturated_fraction: f32) -> Self {
        let regime = if saturated_fraction > 0.30 || (mean_luma >= 254.0 && p95_luma >= 254.0) {
            SceneLightingRegime::FullSaturation
        } else if mean_luma < 8.0 {
            SceneLightingRegime::UltraDark
        } else if mean_luma > 170.0 || p95_luma > 225.0 || saturated_fraction > 0.08 {
            SceneLightingRegime::HighExposure
        } else if mean_luma <= 45.0 {
            SceneLightingRegime::LowLight
        } else {
            SceneLightingRegime::NormalDaylight
        };

        // Midtone target ~ 100 ADU (~18% gray in sRGB space)
        let target = 100.0f32;
        let recommended_ev_shift = if regime == SceneLightingRegime::FullSaturation {
            -4.0 - (saturated_fraction * 3.0)
        } else if mean_luma > target {
            let ratio = (mean_luma / target).max(1.0);
            -ratio.log2().clamp(0.0, 6.0)
        } else {
            let ratio = (target / mean_luma.max(1.0)).max(1.0);
            ratio.log2().clamp(0.0, 6.0)
        };

        Self {
            mean_luma,
            p95_luma,
            saturated_pixel_fraction: saturated_fraction,
            regime,
            recommended_ev_shift,
        }
    }
}


