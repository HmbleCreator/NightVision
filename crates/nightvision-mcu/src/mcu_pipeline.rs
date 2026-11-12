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
use nightvision_core::{DisplayMode, FrameProcessor};

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
}

impl<const WIDTH: usize, const HEIGHT: usize> McuNightVisionEngine<WIDTH, HEIGHT> {
    pub fn new(config: McuEngineConfig) -> Self {
        Self {
            config,
            line_buffer: LineBuffer3::new(),
            temporal_luma_buffer: vec![0u8; WIDTH * HEIGHT],
            scratch_row: [0u8; WIDTH],
            frame_count: 0,
        }
    }

    /// Reset engine internal state.
    pub fn reset(&mut self) {
        self.line_buffer.reset();
        self.temporal_luma_buffer.fill(0);
        self.frame_count = 0;
    }

    /// Process a complete video frame using the MCU streaming line-buffered pipeline.
    pub fn process_frame_streaming(&mut self, input_frame: &RgbImage) -> RgbImage {
        let mut output_frame = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        self.line_buffer.reset();

        let motion_thresh = self.config.motion_threshold_adu;

        for y in 0..HEIGHT {
            let mut temporal_line = [0u8; WIDTH];

            for x in 0..WIDTH {
                let pixel = input_frame.get_pixel(x as u32, y as u32);
                let raw_y = (((pixel[0] as u32) * 77 + (pixel[1] as u32) * 150 + (pixel[2] as u32) * 29) >> 8) as u8;

                let buf_idx = y * WIDTH + x;
                let old_acc = self.temporal_luma_buffer[buf_idx];

                let new_acc = if self.frame_count == 0 {
                    raw_y
                } else {
                    let diff = abs_diff_u8(raw_y, old_acc);
                    if diff < motion_thresh {
                        ema_blend_static_u8(raw_y, old_acc)
                    } else {
                        ema_blend_motion_u8(raw_y, old_acc)
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
                    let contrast_val = if self.config.enable_unsharp_boost {
                        integer_unsharp_mask(mid[x], filtered_val, 1)
                    } else {
                        filtered_val
                    };

                    let tone_mapped = self.config.autogating_lut.lookup(contrast_val);
                    let rgb = if self.config.display_mode == DisplayMode::FullColor {
                        let p = input_frame.get_pixel(x as u32, out_y);
                        let raw_y = (mid[x] as u32).max(1);
                        let ratio_q8 = ((tone_mapped as u32) << 8) / raw_y;
                        [
                            (((p[0] as u32 * ratio_q8) >> 8).min(255)) as u8,
                            (((p[1] as u32 * ratio_q8) >> 8).min(255)) as u8,
                            (((p[2] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        ]
                    } else {
                        self.config.phosphor_lut.map_rgb(tone_mapped)
                    };
                    output_frame.put_pixel(x as u32, out_y, Rgb(rgb));
                }
            }
        }

        if HEIGHT > 1 {
            let last_y = (HEIGHT - 1) as u32;
            let (top, mid, _) = self.line_buffer.get_window();
            filter_line_deterministic(top, mid, mid, &mut self.scratch_row, &self.config.noise_lut);

            for x in 0..WIDTH {
                let tone_mapped = self.config.autogating_lut.lookup(self.scratch_row[x]);
                let rgb = if self.config.display_mode == DisplayMode::FullColor {
                    let p = input_frame.get_pixel(x as u32, last_y);
                    let raw_y = (mid[x] as u32).max(1);
                    let ratio_q8 = ((tone_mapped as u32) << 8) / raw_y;
                    [
                        (((p[0] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        (((p[1] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        (((p[2] as u32 * ratio_q8) >> 8).min(255)) as u8,
                    ]
                } else {
                    self.config.phosphor_lut.map_rgb(tone_mapped)
                };
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
        }
    }

    pub fn reset(&mut self) {
        self.temporal_luma_buffer.fill(0);
        self.line_buffer.fill(0);
        self.scratch_row.fill(0);
        self.frame_count = 0;
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
        }

        if output.dimensions() != (w_u32, h_u32) {
            *output = ImageBuffer::new(w_u32, h_u32);
        }

        let motion_thresh = self.config.motion_threshold_adu;
        let mut cur_line_idx = 0usize;
        let mut lines_pushed = 0usize;

        for y in 0..h {
            let line_offset = cur_line_idx * w;
            for x in 0..w {
                let pixel = input.get_pixel(x as u32, y as u32);
                let raw_y = (((pixel[0] as u32) * 77 + (pixel[1] as u32) * 150 + (pixel[2] as u32) * 29) >> 8) as u8;
                let buf_idx = y * w + x;
                let old_acc = self.temporal_luma_buffer[buf_idx];

                let new_acc = if self.frame_count == 0 {
                    raw_y
                } else {
                    let diff = abs_diff_u8(raw_y, old_acc);
                    if diff < motion_thresh {
                        ema_blend_static_u8(raw_y, old_acc)
                    } else {
                        ema_blend_motion_u8(raw_y, old_acc)
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
                    let contrast = if self.config.enable_unsharp_boost {
                        integer_unsharp_mask(mid[x], filtered, 1)
                    } else {
                        filtered
                    };
                    let tone_mapped = self.config.autogating_lut.lookup(contrast);
                    let rgb = if self.config.display_mode == DisplayMode::FullColor {
                        let p = input.get_pixel(x as u32, out_y);
                        let raw_y = (mid[x] as u32).max(1);
                        let ratio_q8 = ((tone_mapped as u32) << 8) / raw_y;
                        [
                            (((p[0] as u32 * ratio_q8) >> 8).min(255)) as u8,
                            (((p[1] as u32 * ratio_q8) >> 8).min(255)) as u8,
                            (((p[2] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        ]
                    } else {
                        self.config.phosphor_lut.map_rgb(tone_mapped)
                    };
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
                let tone_mapped = self.config.autogating_lut.lookup(self.scratch_row[x]);
                let rgb = if self.config.display_mode == DisplayMode::FullColor {
                    let p = input.get_pixel(x as u32, last_y);
                    let raw_y = (mid[x] as u32).max(1);
                    let ratio_q8 = ((tone_mapped as u32) << 8) / raw_y;
                    [
                        (((p[0] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        (((p[1] as u32 * ratio_q8) >> 8).min(255)) as u8,
                        (((p[2] as u32 * ratio_q8) >> 8).min(255)) as u8,
                    ]
                } else {
                    self.config.phosphor_lut.map_rgb(tone_mapped)
                };
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
}
