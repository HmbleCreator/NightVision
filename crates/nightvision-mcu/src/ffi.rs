//! C-FFI Seam for Embedded Firmware Integration.
//!
//! Exposes an ABI-stable C interface allowing C/C++ embedded camera firmware
//! (ESP-IDF, STM32CubeHAL, Zephyr RTOS, FreeRTOS) to invoke the zero-copy
//! NightVision MCU engine directly.

use std::os::raw::c_char;
use std::slice;

use crate::deterministic_filter::filter_line_deterministic_slice;
use crate::lut::{AutoGatingLut, DeterministicNoiseLut};
use crate::mcu_pipeline::{McuDynamicProcessor, McuEngineConfig};
use nightvision_core::{DisplayMode, FrameProcessor};

/// Maps C enum value to internal DisplayMode.
fn parse_c_display_mode(mode: u8) -> DisplayMode {
    match mode {
        0 => DisplayMode::FullColor,
        1 => DisplayMode::TacticalWhitePhosphor,
        2 => DisplayMode::TacticalGreenPhosphor,
        3 => DisplayMode::HighContrastRecon,
        4 => DisplayMode::ThermalFusion,
        _ => DisplayMode::TacticalWhitePhosphor,
    }
}

/// Allocates and initializes an MCU NightVision pipeline instance.
/// Must be freed using `nightvision_destroy`.
#[no_mangle]
pub unsafe extern "C" fn nightvision_create(
    _width: u32,
    _height: u32,
    display_mode: u8,
) -> *mut McuDynamicProcessor {
    let mode = parse_c_display_mode(display_mode);
    let config = McuEngineConfig::default().with_mode(mode);
    let processor = Box::new(McuDynamicProcessor::new(config));
    Box::into_raw(processor)
}

/// Deallocates an MCU NightVision pipeline instance.
#[no_mangle]
pub unsafe extern "C" fn nightvision_destroy(handle: *mut McuDynamicProcessor) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Sets the tactical display mode (0 = FullColor, 1 = WhitePhosphor, 2 = GreenPhosphor, 3 = Recon, 4 = Thermal).
#[no_mangle]
pub unsafe extern "C" fn nightvision_set_display_mode(
    handle: *mut McuDynamicProcessor,
    display_mode: u8,
) {
    if let Some(proc) = handle.as_mut() {
        proc.set_display_mode(parse_c_display_mode(display_mode));
    }
}

/// Resets internal temporal accumulator and motion history.
#[no_mangle]
pub unsafe extern "C" fn nightvision_reset(handle: *mut McuDynamicProcessor) {
    if let Some(proc) = handle.as_mut() {
        proc.reset();
    }
}

/// Processes a full RGB8 frame in-place or from input buffer to output buffer.
/// `in_rgb` and `out_rgb` must point to buffers of size `width * height * 3`.
#[no_mangle]
pub unsafe extern "C" fn nightvision_process_frame(
    handle: *mut McuDynamicProcessor,
    width: u32,
    height: u32,
    in_rgb: *const u8,
    out_rgb: *mut u8,
) -> i32 {
    if handle.is_null() || in_rgb.is_null() || out_rgb.is_null() {
        return -1;
    }

    let proc = &mut *handle;
    let byte_count = (width as usize) * (height as usize) * 3;

    let in_slice = slice::from_raw_parts(in_rgb, byte_count);
    let out_slice = slice::from_raw_parts_mut(out_rgb, byte_count);

    if let Some(in_img) = image::RgbImage::from_raw(width, height, in_slice.to_vec()) {
        let mut out_img = image::ImageBuffer::new(width, height);
        proc.process_frame_into(&in_img, &mut out_img);
        out_slice.copy_from_slice(out_img.as_raw());
        0
    } else {
        -2
    }
}

/// Zero-copy single row processing for streaming embedded camera interrupts.
/// Executes deterministic edge-preserving spatial filtering and tone-mapping
/// across 3 sliding lines directly into `out_line`.
#[no_mangle]
pub unsafe extern "C" fn nightvision_process_line_luma(
    top_line: *const u8,
    mid_line: *const u8,
    bot_line: *const u8,
    out_line: *mut u8,
    width: u32,
    noise_lut_ptr: *const u8,
    gating_lut_ptr: *const u8,
) -> i32 {
    if top_line.is_null() || mid_line.is_null() || bot_line.is_null() || out_line.is_null() {
        return -1;
    }

    let w = width as usize;
    let top = slice::from_raw_parts(top_line, w);
    let mid = slice::from_raw_parts(mid_line, w);
    let bot = slice::from_raw_parts(bot_line, w);
    let out = slice::from_raw_parts_mut(out_line, w);

    let noise_lut = if !noise_lut_ptr.is_null() {
        let threshold_table = *(noise_lut_ptr as *const [u8; 256]);
        DeterministicNoiseLut { threshold_table }
    } else {
        DeterministicNoiseLut::new(0.85, 2.1, 2.5)
    };

    let gating_lut = if !gating_lut_ptr.is_null() {
        let table = *(gating_lut_ptr as *const [u8; 256]);
        AutoGatingLut { table }
    } else {
        AutoGatingLut::new(2.8, 0.60, 0.88)
    };

    filter_line_deterministic_slice(top, mid, bot, out, &noise_lut);

    // Tone map filtered result in-place
    for val in out.iter_mut() {
        *val = gating_lut.lookup(*val);
    }

    0
}

/// Returns library version string.
#[no_mangle]
pub extern "C" fn nightvision_version() -> *const c_char {
    static VERSION: &[u8] = b"NightVision Embedded 0.1.0\0";
    VERSION.as_ptr() as *const c_char
}
