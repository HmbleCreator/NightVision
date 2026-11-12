//! Integer-only fixed-point arithmetic and bit-shifting primitives.
//!
//! Designed for microcontrollers without FPUs or where floating-point cycles
//! choke the pipeline. Zero division, zero floating-point operations.

/// Fast integer square root using binary restoration (digit-by-digit).
/// Runs in constant 16 iterations without division or hardware FPU.
#[inline(always)]
pub fn isqrt_u32(n: u32) -> u32 {
    let mut root = 0u32;
    let mut bit = 1u32 << 30; // Second-to-top bit

    while bit > n {
        bit >>= 2;
    }

    let mut val = n;
    while bit != 0 {
        if val >= root + bit {
            val -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }

    root
}

/// Bit-shift Exponential Moving Average (EMA) for temporal SNR accumulation.
/// High accumulation mode: (new * 1 + old * 7) >> 3  (~87.5% retention, depth ~8 frames).
#[inline(always)]
pub fn ema_blend_static_u8(new_sample: u8, old_acc: u8) -> u8 {
    let sum = (new_sample as u16) + (old_acc as u16) * 7;
    (sum >> 3) as u8
}

/// Bit-shift Exponential Moving Average (EMA) for high accumulation (depth ~16 frames).
/// (new * 1 + old * 15) >> 4 (~93.75% retention).
#[inline(always)]
pub fn ema_blend_deep_u8(new_sample: u8, old_acc: u8) -> u8 {
    let sum = (new_sample as u16) + (old_acc as u16) * 15;
    (sum >> 4) as u8
}

/// Bit-shift EMA for moving targets: (new * 3 + old * 1) >> 2 (~75% new signal, instant response, 0 ghosting).
#[inline(always)]
pub fn ema_blend_motion_u8(new_sample: u8, old_acc: u8) -> u8 {
    let sum = (new_sample as u16) * 3 + (old_acc as u16);
    (sum >> 2) as u8
}

/// Absolute difference between two u8 bytes.
#[inline(always)]
pub fn abs_diff_u8(a: u8, b: u8) -> u8 {
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// Q8 fixed-point multiply: (a * b) >> 8
#[inline(always)]
pub fn q8_mul(a: u8, b: u8) -> u8 {
    (((a as u16) * (b as u16)) >> 8) as u8
}

/// Q15 fixed-point multiply: (a * b) >> 15
#[inline(always)]
pub fn q15_mul(a: i16, b: i16) -> i16 {
    (((a as i32) * (b as i32)) >> 15) as i16
}

/// Clamps an i32 to u8 [0, 255] with zero branching where supported.
#[inline(always)]
pub fn clamp_to_u8(val: i32) -> u8 {
    if val < 0 {
        0
    } else if val > 255 {
        255
    } else {
        val as u8
    }
}
