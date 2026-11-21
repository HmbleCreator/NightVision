//! Single-Pass Streaming MCU ISP Engine.
//!
//! Fuses Temporal Accumulation, Deterministic Edge-Preserving Spatial Denoising,
//! Auto-Gating, and Phosphor Rendering into a single-pass streaming architecture.

use crate::deterministic_filter::{
    filter_line_deterministic, filter_line_deterministic_slice, integer_unsharp_mask,
};
use crate::fixed_math::{abs_diff_u8, ema_blend_motion_u8, ema_blend_static_u8};
use crate::line_buffer::LineBuffer3;
use crate::lut::{AutoGatingLut, DeterministicNoiseLut, PhosphorLut};
use image::{ImageBuffer, Rgb, RgbImage};
use nightvision_core::{DisplayMode, ExposureMeteringReport, FrameProcessor};

/// Smooth C1-continuous chroma roll-off in Q8 fixed-point without branch jumps.
/// - Below 16 ADU: scales smoothly from 0 to 256 to suppress dark noise chrominance.
/// - Between 16 and 235 ADU: strict 256 (1.0x), 100% full rich color preserved (e.g. blue doorframe).
/// - Above 235 ADU: smooth Hermite cubic smoothstep roll-off from 256 down to 0 at 255 ADU (neutral white).
#[inline(always)]
fn compute_chroma_atten_q8(raw_u8: u8) -> u32 {
    if raw_u8 < 16 {
        (raw_u8 as u32) * 16
    } else if raw_u8 > 235 {
        let t_q8 = (((raw_u8 as u32 - 235) * 256) / 20).min(256);
        let s_q8 = (t_q8 * t_q8 * (768 - 2 * t_q8)) >> 16;
        256 - s_q8.min(256)
    } else {
        256
    }
}

/// Unified, 100% monotonic pixel rendering for both MCU streaming and dynamic engines.
/// In high-exposure mode: evaluates optical veiling glare floor subtraction + 55/45 power curve.
/// In normal/dark mode: evaluates Reinhard auto-gating and C1 chroma attenuation.
#[inline(always)]
fn render_mcu_pixel(
    p: image::Rgb<u8>,
    raw_u8: u8,
    tone_mapped: u8,
    is_high_exposure: bool,
    glare_sub: f32,
    denom: f32,
    gamma: f32,
    display_mode: DisplayMode,
    phosphor_lut: &PhosphorLut,
) -> [u8; 3] {
    if is_high_exposure {
        let r_in = p[0] as f32;
        let g_in = p[1] as f32;
        let b_in = p[2] as f32;

        let r_d = ((r_in - glare_sub).max(0.0) / denom).clamp(0.0, 1.0);
        let g_d = ((g_in - glare_sub).max(0.0) / denom).clamp(0.0, 1.0);
        let b_d = ((b_in - glare_sub).max(0.0) / denom).clamp(0.0, 1.0);

        let r_rgb = r_d.powf(gamma);
        let g_rgb = g_d.powf(gamma);
        let b_rgb = b_d.powf(gamma);

        let y_d = 0.299 * r_d + 0.587 * g_d + 0.114 * b_d;
        let f_yd = y_d.powf(gamma);
        let scale = f_yd / y_d.max(1e-4);

        let r_luma = r_d * scale;
        let g_luma = g_d * scale;
        let b_luma = b_d * scale;

        let r_out = (0.55 * r_rgb + 0.45 * r_luma).clamp(0.0, 1.0);
        let g_out = (0.55 * g_rgb + 0.45 * g_luma).clamp(0.0, 1.0);
        let b_out = (0.55 * b_rgb + 0.45 * b_luma).clamp(0.0, 1.0);

        match display_mode {
            DisplayMode::FullColor => [
                (r_out * 255.0).round().clamp(0.0, 255.0) as u8,
                (g_out * 255.0).round().clamp(0.0, 255.0) as u8,
                (b_out * 255.0).round().clamp(0.0, 255.0) as u8,
            ],
            _ => {
                let comp = (0.299 * r_out + 0.587 * g_out + 0.114 * b_out) * 255.0;
                let tm = comp.round().clamp(0.0, 255.0) as u8;
                phosphor_lut.map_rgb(tm)
            }
        }
    } else {
        if display_mode == DisplayMode::FullColor {
            let true_y = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8).max(1);
            let tm = tone_mapped as u32;
            let ratio_q8 = (tm << 8) / true_y;

            let r_col = ((p[0] as u32 * ratio_q8) >> 8).min(255);
            let g_col = ((p[1] as u32 * ratio_q8) >> 8).min(255);
            let b_col = ((p[2] as u32 * ratio_q8) >> 8).min(255);

            let chroma_atten_q8 = compute_chroma_atten_q8(raw_u8);
            [
                (((r_col * chroma_atten_q8 + tm * (256 - chroma_atten_q8)) >> 8).min(255)) as u8,
                (((g_col * chroma_atten_q8 + tm * (256 - chroma_atten_q8)) >> 8).min(255)) as u8,
                (((b_col * chroma_atten_q8 + tm * (256 - chroma_atten_q8)) >> 8).min(255)) as u8,
            ]
        } else {
            phosphor_lut.map_rgb(tone_mapped)
        }
    }
}

/// Complete configuration for the MCU Streaming NightVision Engine.
#[derive(Debug, Clone)]
pub struct McuEngineConfig {
    pub autogating_lut: AutoGatingLut,
    pub noise_lut: DeterministicNoiseLut,
    pub phosphor_lut: PhosphorLut,
    pub display_mode: DisplayMode,
    /// Motion threshold in ADU (typically 12 to 24 on 0..255 scale)
    pub motion_threshold_adu: u8,
    /// Enable local contrast boost via integer unsharp mask
    pub enable_unsharp_boost: bool,
    /// Enable adaptive shadow-knee expansion for ultra-dark scenes (< 8.0 ADU)
    pub enable_adaptive_knee: bool,
}

impl Default for McuEngineConfig {
    fn default() -> Self {
        Self {
            autogating_lut: AutoGatingLut::new(2.8, 0.60, 0.88),
            noise_lut: DeterministicNoiseLut::new(0.85, 2.1, 2.5),
            phosphor_lut: PhosphorLut::p45_white(),
            display_mode: DisplayMode::TacticalWhitePhosphor,
            motion_threshold_adu: 16,
            enable_unsharp_boost: true,
            enable_adaptive_knee: true,
        }
    }
}

impl McuEngineConfig {
    /// Update configuration with specified tactical display mode and appropriate LUT.
    pub fn with_mode(mut self, mode: DisplayMode) -> Self {
        self.display_mode = mode;
        self.phosphor_lut = match mode {
            DisplayMode::TacticalGreenPhosphor => PhosphorLut::p43_green(),
            DisplayMode::TacticalWhitePhosphor => PhosphorLut::p45_white(),
            DisplayMode::HighContrastRecon => PhosphorLut::recon_amber(),
            _ => PhosphorLut::p45_white(),
        };
        self
    }
}

/// Real-Time MCU NightVision Engine with row-by-row streaming.
pub struct McuNightVisionEngine<const WIDTH: usize, const HEIGHT: usize> {
    pub config: McuEngineConfig,
    /// Circular 3-line spatial buffer (< 2KB SRAM)
    line_buffer: LineBuffer3<WIDTH>,
    /// Single 8-bit rolling temporal luma buffer (e.g. in internal SRAM or ESP32-S3 PSRAM)
    temporal_luma_buffer: Vec<u8>,
    /// Pre-allocated scratch row for spatial filter output
    scratch_row: [u8; WIDTH],
    frame_count: usize,
    smoothed_luma: f32,
    applied_knee_luma: f32,
    pub last_p2_luma: f32,
    pub last_exposure_report: ExposureMeteringReport,
}

impl<const WIDTH: usize, const HEIGHT: usize> McuNightVisionEngine<WIDTH, HEIGHT> {
    pub fn new(config: McuEngineConfig) -> Self {
        Self {
            config,
            line_buffer: LineBuffer3::new(),
            temporal_luma_buffer: vec![0u8; WIDTH * HEIGHT],
            scratch_row: [0u8; WIDTH],
            frame_count: 0,
            smoothed_luma: 16.0,
            applied_knee_luma: 16.0,
            last_p2_luma: 0.0,
            last_exposure_report: ExposureMeteringReport::default(),
        }
    }

    /// Reset engine internal state.
    pub fn reset(&mut self) {
        self.line_buffer.reset();
        self.temporal_luma_buffer.fill(0);
        self.frame_count = 0;
        self.smoothed_luma = 16.0;
        self.applied_knee_luma = 16.0;
        self.last_p2_luma = 0.0;
        self.last_exposure_report = ExposureMeteringReport::default();
    }

    /// Returns the latest exposure metering report.
    pub fn exposure_report(&self) -> ExposureMeteringReport {
        self.last_exposure_report
    }

    /// Process a complete video frame using the MCU streaming line-buffered pipeline.
    pub fn process_frame_streaming(&mut self, input_frame: &RgbImage) -> RgbImage {
        let mut output_frame = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        self.line_buffer.reset();

        // Auto-exposure scene luminance tracking with 16-frame EMA smoothing, 256-bin histogram, and saturation metering
        if self.config.enable_adaptive_knee && (WIDTH * HEIGHT) > 0 {
            let mut sample_sum = 0u64;
            let mut hist = [0u16; 256];
            let mut sat_samples = 0u32;
            let n = WIDTH * HEIGHT;
            let sample_target = 2048.min(n);
            let stride = ((n / sample_target) | 1).max(1);
            let mut idx = 17usize;
            let mut samples = 0u64;
            for _ in 0..sample_target {
                let i = idx % n;
                let x = (i % WIDTH) as u32;
                let y = (i / WIDTH) as u32;
                let p = input_frame.get_pixel(x, y);
                let y_val = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as usize;
                sample_sum += y_val as u64;
                hist[y_val.min(255)] += 1;
                if y_val >= 250 {
                    sat_samples += 1;
                }
                samples += 1;
                idx = (idx + stride).wrapping_add(7);
            }
            let current_mean = (sample_sum as f32) / (samples as f32).max(1.0);
            let sat_ratio = (sat_samples as f32) / (samples as f32).max(1.0);

            let target_p2 = (samples as f32 * 0.02) as u16;
            let target_p95 = (samples as f32 * 0.95) as u16;
            let mut cum = 0u16;
            let mut p2_bin = 0usize;
            let mut p95_bin = 255usize;
            let mut found_p2 = false;
            for (bin, &count) in hist.iter().enumerate() {
                cum += count;
                if !found_p2 && cum >= target_p2 {
                    p2_bin = bin;
                    found_p2 = true;
                }
                if cum >= target_p95 {
                    p95_bin = bin;
                    break;
                }
            }
            let p2_luma = p2_bin as f32;
            let p95_luma = (p95_bin as f32).min(255.0);
            self.last_p2_luma = p2_luma;
            self.last_exposure_report = ExposureMeteringReport::compute(current_mean, p95_luma, sat_ratio);

            let smoothed = if self.frame_count == 0 {
                current_mean
            } else {
                (current_mean + 15.0 * self.smoothed_luma) / 16.0
            };
            self.smoothed_luma = smoothed;

            if self.frame_count == 0 || (smoothed - self.applied_knee_luma).abs() >= 0.5 {
                self.applied_knee_luma = smoothed;
                self.config.autogating_lut = AutoGatingLut::new_bidirectional(
                    2.8,
                    0.60,
                    0.88,
                    smoothed,
                    p95_luma,
                    sat_ratio,
                );
            }
        }

        let motion_thresh = self.config.motion_threshold_adu;
        let is_high_exposure = self.applied_knee_luma > 170.0
            || self.last_exposure_report.p95_luma > 225.0
            || self.last_exposure_report.saturated_pixel_fraction > 0.08;

        let p2 = self.last_p2_luma;
        let mean = self.applied_knee_luma;
        let t_s = ((mean - 145.0) / 30.0).clamp(0.0, 1.0);
        let s = t_s * t_s * (3.0 - 2.0 * t_s);
        let glare_sub = (0.65 * p2 * s).clamp(0.0, 32.0);
        let denom = (255.0 - glare_sub).max(1e-4);

        let gamma_factor = ((mean - 140.0) / 75.0).clamp(0.0, 1.0);
        let gamma = 1.0 + 0.55 * gamma_factor;

        for y in 0..HEIGHT {
            let mut temporal_line = [0u8; WIDTH];

            for x in 0..WIDTH {
                let pixel = input_frame.get_pixel(x as u32, y as u32);
                let raw_y = (((pixel[0] as u32) * 77 + (pixel[1] as u32) * 150 + (pixel[2] as u32) * 29) >> 8) as u8;
                let eff_y = raw_y;

                let buf_idx = y * WIDTH + x;
                let old_acc = self.temporal_luma_buffer[buf_idx];

                let new_acc = if self.frame_count == 0 {
                    eff_y
                } else {
                    let diff = abs_diff_u8(eff_y, old_acc);
                    if diff < motion_thresh {
                        ema_blend_static_u8(eff_y, old_acc)
                    } else {
                        ema_blend_motion_u8(eff_y, old_acc)
                    }
                };

                self.temporal_luma_buffer[buf_idx] = new_acc;
                temporal_line[x] = new_acc;
            }

            self.line_buffer.push_line(&temporal_line);

            if self.line_buffer.is_ready() && y >= 1 {
                let (top, mid, bot) = self.line_buffer.get_window();

                filter_line_deterministic(top, mid, bot, &mut self.scratch_row, &self.config.noise_lut);

                let out_y = (y - 1) as u32;
                for x in 0..WIDTH {
                    let filtered_val = self.scratch_row[x];
                    let contrast_val = if is_high_exposure {
                        mid[x]
                    } else if self.config.enable_unsharp_boost {
                        integer_unsharp_mask(mid[x], filtered_val, 1)
                    } else {
                        filtered_val
                    };

                    let tone_mapped = self.config.autogating_lut.lookup(contrast_val);
                    let p = input_frame.get_pixel(x as u32, out_y);
                    let rgb = render_mcu_pixel(
                        *p,
                        mid[x],
                        tone_mapped,
                        is_high_exposure,
                        glare_sub,
                        denom,
                        gamma,
                        self.config.display_mode,
                        &self.config.phosphor_lut,
                    );
                    output_frame.put_pixel(x as u32, out_y, Rgb(rgb));
                }
            }
        }

        if HEIGHT > 1 {
            let last_y = (HEIGHT - 1) as u32;
            let (top, mid, _) = self.line_buffer.get_window();
            filter_line_deterministic(top, mid, mid, &mut self.scratch_row, &self.config.noise_lut);

            for x in 0..WIDTH {
                let filtered = self.scratch_row[x];
                let contrast = if is_high_exposure {
                    mid[x]
                } else if self.config.enable_unsharp_boost {
                    integer_unsharp_mask(mid[x], filtered, 1)
                } else {
                    filtered
                };
                let tone_mapped = self.config.autogating_lut.lookup(contrast);
                let p = input_frame.get_pixel(x as u32, last_y);
                let rgb = render_mcu_pixel(
                    *p,
                    mid[x],
                    tone_mapped,
                    is_high_exposure,
                    glare_sub,
                    denom,
                    gamma,
                    self.config.display_mode,
                    &self.config.phosphor_lut,
                );
                output_frame.put_pixel(x as u32, last_y, Rgb(rgb));
            }
        }

        self.frame_count += 1;
        output_frame
    }
}

impl<const WIDTH: usize, const HEIGHT: usize> FrameProcessor for McuNightVisionEngine<WIDTH, HEIGHT> {
    fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage) {
        let (w, h) = input.dimensions();
        if (w as usize) != WIDTH || (h as usize) != HEIGHT {
            return;
        }
        if output.dimensions() != (w, h) {
            *output = ImageBuffer::new(w, h);
        }
        let enhanced = self.process_frame_streaming(input);
        output.copy_from_slice(enhanced.as_raw());
    }

    fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        self.process_frame_streaming(input)
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.config.display_mode = mode;
        self.config.phosphor_lut = match mode {
            DisplayMode::TacticalGreenPhosphor => PhosphorLut::p43_green(),
            DisplayMode::TacticalWhitePhosphor => PhosphorLut::p45_white(),
            DisplayMode::HighContrastRecon => PhosphorLut::recon_amber(),
            _ => PhosphorLut::p45_white(),
        };
    }

    fn display_mode(&self) -> DisplayMode {
        self.config.display_mode
    }

    fn backend_name(&self) -> &'static str {
        "MCU Fixed-Point Streaming (Static Memory Buffer)"
    }

    fn exposure_report(&self) -> Option<ExposureMeteringReport> {
        Some(self.last_exposure_report)
    }
}

/// Dynamic resolution adapter for MCU streaming night vision engine.
///
/// Implements FrameProcessor for any runtime resolution (VGA, 720p, etc.)
/// using a minimal 3-line sliding window and integer bit-shifts.
pub struct McuDynamicProcessor {
    pub config: McuEngineConfig,
    temporal_luma_buffer: Vec<u8>,
    line_buffer: Vec<u8>,
    scratch_row: Vec<u8>,
    width: usize,
    height: usize,
    frame_count: usize,
    smoothed_luma: f32,
    applied_knee_luma: f32,
    pub last_p2_luma: f32,
    pub last_exposure_report: ExposureMeteringReport,
}

impl McuDynamicProcessor {
    pub fn new(config: McuEngineConfig) -> Self {
        Self {
            config,
            temporal_luma_buffer: Vec::new(),
            line_buffer: Vec::new(),
            scratch_row: Vec::new(),
            width: 0,
            height: 0,
            frame_count: 0,
            smoothed_luma: 16.0,
            applied_knee_luma: 16.0,
            last_p2_luma: 0.0,
            last_exposure_report: ExposureMeteringReport::default(),
        }
    }

    pub fn reset(&mut self) {
        self.temporal_luma_buffer.fill(0);
        self.line_buffer.fill(0);
        self.scratch_row.fill(0);
        self.frame_count = 0;
        self.smoothed_luma = 16.0;
        self.applied_knee_luma = 16.0;
        self.last_p2_luma = 0.0;
        self.last_exposure_report = ExposureMeteringReport::default();
    }

    /// Process a frame into a newly allocated RgbImage.
    pub fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        let mut out = ImageBuffer::new(input.width(), input.height());
        self.process_frame_into(input, &mut out);
        out
    }

    /// Returns the latest exposure metering report.
    pub fn exposure_report(&self) -> ExposureMeteringReport {
        self.last_exposure_report
    }
}

impl FrameProcessor for McuDynamicProcessor {
    fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage) {
        let (w_u32, h_u32) = input.dimensions();
        let (w, h) = (w_u32 as usize, h_u32 as usize);

        if self.width != w || self.height != h {
            self.width = w;
            self.height = h;
            self.temporal_luma_buffer = vec![0u8; w * h];
            self.line_buffer = vec![0u8; w * 3];
            self.scratch_row = vec![0u8; w];
            self.frame_count = 0;
            self.smoothed_luma = 16.0;
            self.applied_knee_luma = 16.0;
            self.last_p2_luma = 0.0;
            self.last_exposure_report = ExposureMeteringReport::default();
        }

        if output.dimensions() != (w_u32, h_u32) {
            *output = ImageBuffer::new(w_u32, h_u32);
        }

        // Auto-exposure scene luminance tracking with 16-frame EMA smoothing, 256-bin histogram, and saturation metering
        if self.config.enable_adaptive_knee && (w * h) > 0 {
            let mut sample_sum = 0u64;
            let mut hist = [0u16; 256];
            let mut sat_samples = 0u32;
            let n = w * h;
            let sample_target = 2048.min(n);
            let stride = ((n / sample_target) | 1).max(1);
            let mut idx = 17usize;
            let mut samples = 0u64;
            for _ in 0..sample_target {
                let i = idx % n;
                let x = (i % w) as u32;
                let y = (i / w) as u32;
                let p = input.get_pixel(x, y);
                let y_val = (((p[0] as u32) * 77 + (p[1] as u32) * 150 + (p[2] as u32) * 29) >> 8) as usize;
                sample_sum += y_val as u64;
                hist[y_val.min(255)] += 1;
                if y_val >= 250 {
                    sat_samples += 1;
                }
                samples += 1;
                idx = (idx + stride).wrapping_add(7);
            }
            let current_mean = (sample_sum as f32) / (samples as f32).max(1.0);
            let sat_ratio = (sat_samples as f32) / (samples as f32).max(1.0);

            let target_p2 = (samples as f32 * 0.02) as u16;
            let target_p95 = (samples as f32 * 0.95) as u16;
            let mut cum = 0u16;
            let mut p2_bin = 0usize;
            let mut p95_bin = 255usize;
            let mut found_p2 = false;
            for (bin, &count) in hist.iter().enumerate() {
                cum += count;
                if !found_p2 && cum >= target_p2 {
                    p2_bin = bin;
                    found_p2 = true;
                }
                if cum >= target_p95 {
                    p95_bin = bin;
                    break;
                }
            }
            let p2_luma = p2_bin as f32;
            let p95_luma = (p95_bin as f32).min(255.0);
            self.last_p2_luma = p2_luma;
            self.last_exposure_report = ExposureMeteringReport::compute(current_mean, p95_luma, sat_ratio);

            let smoothed = if self.frame_count == 0 {
                current_mean
            } else {
                (current_mean + 15.0 * self.smoothed_luma) / 16.0
            };
            self.smoothed_luma = smoothed;

            if self.frame_count == 0 || (smoothed - self.applied_knee_luma).abs() >= 0.5 {
                self.applied_knee_luma = smoothed;
                self.config.autogating_lut = AutoGatingLut::new_bidirectional(
                    2.8,
                    0.60,
                    0.88,
                    smoothed,
                    p95_luma,
                    sat_ratio,
                );
            }
        }

        let motion_thresh = self.config.motion_threshold_adu;
        let is_high_exposure = self.applied_knee_luma > 170.0
            || self.last_exposure_report.p95_luma > 225.0
            || self.last_exposure_report.saturated_pixel_fraction > 0.08;

        let p2 = self.last_p2_luma;
        let mean = self.applied_knee_luma;
        let t_s = ((mean - 145.0) / 30.0).clamp(0.0, 1.0);
        let s = t_s * t_s * (3.0 - 2.0 * t_s);
        let glare_sub = (0.65 * p2 * s).clamp(0.0, 32.0);
        let denom = (255.0 - glare_sub).max(1e-4);

        let gamma_factor = ((mean - 140.0) / 75.0).clamp(0.0, 1.0);
        let gamma = 1.0 + 0.55 * gamma_factor;

        let mut cur_line_idx = 0usize;
        let mut lines_pushed = 0usize;

        for y in 0..h {
            let line_offset = cur_line_idx * w;
            for x in 0..w {
                let pixel = input.get_pixel(x as u32, y as u32);
                let raw_y = (((pixel[0] as u32) * 77 + (pixel[1] as u32) * 150 + (pixel[2] as u32) * 29) >> 8) as u8;
                let eff_y = raw_y;

                let buf_idx = y * w + x;
                let old_acc = self.temporal_luma_buffer[buf_idx];

                let new_acc = if self.frame_count == 0 {
                    eff_y
                } else {
                    let diff = abs_diff_u8(eff_y, old_acc);
                    if diff < motion_thresh {
                        ema_blend_static_u8(eff_y, old_acc)
                    } else {
                        ema_blend_motion_u8(eff_y, old_acc)
                    }
                };

                self.temporal_luma_buffer[buf_idx] = new_acc;
                self.line_buffer[line_offset + x] = new_acc;
            }

            cur_line_idx = (cur_line_idx + 1) % 3;
            lines_pushed += 1;

            if lines_pushed >= 2 && y >= 1 {
                let top_idx = cur_line_idx * w;
                let mid_idx = ((cur_line_idx + 1) % 3) * w;
                let bot_idx = ((cur_line_idx + 2) % 3) * w;

                let top = &self.line_buffer[top_idx..top_idx + w];
                let mid = &self.line_buffer[mid_idx..mid_idx + w];
                let bot = &self.line_buffer[bot_idx..bot_idx + w];

                filter_line_deterministic_slice(top, mid, bot, &mut self.scratch_row, &self.config.noise_lut);

                let out_y = (y - 1) as u32;
                for x in 0..w {
                    let filtered = self.scratch_row[x];
                    let contrast = if is_high_exposure {
                        mid[x]
                    } else if self.config.enable_unsharp_boost {
                        integer_unsharp_mask(mid[x], filtered, 1)
                    } else {
                        filtered
                    };
                    let tone_mapped = self.config.autogating_lut.lookup(contrast);
                    let p = input.get_pixel(x as u32, out_y);
                    let rgb = render_mcu_pixel(
                        *p,
                        mid[x],
                        tone_mapped,
                        is_high_exposure,
                        glare_sub,
                        denom,
                        gamma,
                        self.config.display_mode,
                        &self.config.phosphor_lut,
                    );
                    output.put_pixel(x as u32, out_y, Rgb(rgb));
                }
            }
        }

        if h > 1 {
            let last_y = (h - 1) as u32;
            let top_idx = cur_line_idx * w;
            let mid_idx = ((cur_line_idx + 1) % 3) * w;

            let top = &self.line_buffer[top_idx..top_idx + w];
            let mid = &self.line_buffer[mid_idx..mid_idx + w];

            filter_line_deterministic_slice(top, mid, mid, &mut self.scratch_row, &self.config.noise_lut);

            for x in 0..w {
                let filtered = self.scratch_row[x];
                let contrast = if is_high_exposure {
                    mid[x]
                } else if self.config.enable_unsharp_boost {
                    integer_unsharp_mask(mid[x], filtered, 1)
                } else {
                    filtered
                };
                let tone_mapped = self.config.autogating_lut.lookup(contrast);
                let p = input.get_pixel(x as u32, last_y);
                let rgb = render_mcu_pixel(
                    *p,
                    mid[x],
                    tone_mapped,
                    is_high_exposure,
                    glare_sub,
                    denom,
                    gamma,
                    self.config.display_mode,
                    &self.config.phosphor_lut,
                );
                output.put_pixel(x as u32, last_y, Rgb(rgb));
            }
        }

        self.frame_count += 1;
    }

    fn reset(&mut self) {
        self.reset();
    }

    fn set_display_mode(&mut self, mode: DisplayMode) {
        self.config.display_mode = mode;
        self.config.phosphor_lut = match mode {
            DisplayMode::TacticalGreenPhosphor => PhosphorLut::p43_green(),
            DisplayMode::TacticalWhitePhosphor => PhosphorLut::p45_white(),
            DisplayMode::HighContrastRecon => PhosphorLut::recon_amber(),
            _ => PhosphorLut::p45_white(),
        };
    }

    fn display_mode(&self) -> DisplayMode {
        self.config.display_mode
    }

    fn backend_name(&self) -> &'static str {
        "MCU Fixed-Point Streaming (Dynamic Adapter)"
    }

    fn exposure_report(&self) -> Option<ExposureMeteringReport> {
        Some(self.last_exposure_report)
    }
}
