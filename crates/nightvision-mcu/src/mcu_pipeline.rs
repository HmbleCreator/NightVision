//! Single-Pass Streaming MCU ISP Engine.
//!
//! Fuses Temporal Accumulation, Deterministic Edge-Preserving Spatial Denoising,
//! Auto-Gating, and Phosphor Rendering into a single-pass streaming architecture.

use crate::deterministic_filter::{filter_line_deterministic, integer_unsharp_mask};
use crate::fixed_math::{abs_diff_u8, ema_blend_motion_u8, ema_blend_static_u8};
use crate::line_buffer::LineBuffer3;
use crate::lut::{AutoGatingLut, DeterministicNoiseLut, PhosphorLut};
use image::{ImageBuffer, Rgb, RgbImage};

/// Complete configuration for the MCU Streaming NightVision Engine.
#[derive(Debug, Clone)]
pub struct McuEngineConfig {
    pub autogating_lut: AutoGatingLut,
    pub noise_lut: DeterministicNoiseLut,
    pub phosphor_lut: PhosphorLut,
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
            motion_threshold_adu: 16,
            enable_unsharp_boost: true,
        }
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
    ///
    /// Simulates hardware streaming: Camera row comes in -> Temporal bitshift -> Line buffer ->
    /// Spatial filter -> LUT Auto-Gating & Phosphor -> RGB row out!
    pub fn process_frame_streaming(&mut self, input_frame: &RgbImage) -> RgbImage {
        let mut output_frame = ImageBuffer::new(WIDTH as u32, HEIGHT as u32);
        self.line_buffer.reset();

        let motion_thresh = self.config.motion_threshold_adu;

        for y in 0..HEIGHT {
            // 1. Convert incoming row to 8-bit Luminance and apply Bit-Shift Temporal EMA
            let mut temporal_line = [0u8; WIDTH];

            for x in 0..WIDTH {
                let pixel = input_frame.get_pixel(x as u32, y as u32);
                // Fast integer luma: Y = (R*77 + G*150 + B*29) >> 8
                let raw_y = (((pixel[0] as u32) * 77 + (pixel[1] as u32) * 150 + (pixel[2] as u32) * 29) >> 8) as u8;

                let buf_idx = y * WIDTH + x;
                let old_acc = self.temporal_luma_buffer[buf_idx];

                if self.frame_count == 0 {
                    self.temporal_luma_buffer[buf_idx] = raw_y;
                    temporal_line[x] = raw_y;
                } else {
                    let diff = abs_diff_u8(raw_y, old_acc);
                    let new_acc = if diff < motion_thresh {
                        // Static noise: apply deep integer EMA (new * 1 + old * 7) >> 3
                        ema_blend_static_u8(raw_y, old_acc)
                    } else {
                        // Dynamic motion: fast response (new * 3 + old * 1) >> 2
                        ema_blend_motion_u8(raw_y, old_acc)
                    };
                    self.temporal_luma_buffer[buf_idx] = new_acc;
                    temporal_line[x] = new_acc;
                }
            }

            // 2. Push row into circular 3-line spatial buffer
            self.line_buffer.push_line(&temporal_line);

            // 3. When 3 lines are available, filter the middle line (y - 1)
            if self.line_buffer.is_ready() && y >= 1 {
                let (top, mid, bot) = self.line_buffer.get_window();

                // Deterministic physics noise-profile spatial filter
                filter_line_deterministic(top, mid, bot, &mut self.scratch_row, &self.config.noise_lut);

                // 4. Output tone mapping and tactical phosphor rendering
                let out_y = (y - 1) as u32;
                for x in 0..WIDTH {
                    let filtered_val = self.scratch_row[x];

                    // Optional integer unsharp contrast boost
                    let contrast_val = if self.config.enable_unsharp_boost {
                        integer_unsharp_mask(mid[x], filtered_val, 1)
                    } else {
                        filtered_val
                    };

                    // LUT Auto-Gating & Gamma
                    let tone_mapped = self.config.autogating_lut.lookup(contrast_val);

                    // LUT Phosphor Rendering
                    let rgb = self.config.phosphor_lut.map_rgb(tone_mapped);
                    output_frame.put_pixel(x as u32, out_y, Rgb(rgb));
                }
            }
        }

        // Flush last remaining row
        if HEIGHT > 1 {
            let last_y = (HEIGHT - 1) as u32;
            let (top, mid, _) = self.line_buffer.get_window();
            filter_line_deterministic(top, mid, mid, &mut self.scratch_row, &self.config.noise_lut);

            for x in 0..WIDTH {
                let tone_mapped = self.config.autogating_lut.lookup(self.scratch_row[x]);
                let rgb = self.config.phosphor_lut.map_rgb(tone_mapped);
                output_frame.put_pixel(x as u32, last_y, Rgb(rgb));
            }
        }

        self.frame_count += 1;
        output_frame
    }
}
