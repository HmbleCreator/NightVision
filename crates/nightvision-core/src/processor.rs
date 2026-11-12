//! Canonical Frame Processor seam for NightVision.
//!
//! Decouples callers (CLI, GUI, benchmarks, stream pipelines) from the underlying
//! computational imaging math and hardware execution backends (Desktop Rayon vs. MCU Fixed-Point).

use image::{ImageBuffer, RgbImage};
use crate::modes::DisplayMode;
use crate::sensor::ExposureMeteringReport;

/// Common interface for night vision frame enhancement engines.
pub trait FrameProcessor {
    /// In-place frame processing into a pre-allocated output buffer.
    ///
    /// Provides zero-allocation streaming for real-time and embedded pipelines.
    fn process_frame_into(&mut self, input: &RgbImage, output: &mut RgbImage);

    /// Convenience method: processes frame and returns a newly allocated buffer.
    fn process_frame(&mut self, input: &RgbImage) -> RgbImage {
        let (w, h) = input.dimensions();
        let mut out = ImageBuffer::new(w, h);
        self.process_frame_into(input, &mut out);
        out
    }

    /// Reset internal state (temporal accumulators, line buffers) on scene cut or mode change.
    fn reset(&mut self);

    /// Switch the active tactical display mode (FullColor, WhitePhosphor, GreenPhosphor, etc.).
    fn set_display_mode(&mut self, mode: DisplayMode);

    /// Get the current active tactical display mode.
    fn display_mode(&self) -> DisplayMode;

    /// Human-readable name of the backend engine (e.g. "Desktop Rayon FPU", "MCU Fixed-Point Streaming").
    fn backend_name(&self) -> &'static str;

    /// Real-time Auto-Exposure Control (AEC) and scene lighting analysis.
    fn exposure_report(&self) -> Option<ExposureMeteringReport> {
        None
    }
}

