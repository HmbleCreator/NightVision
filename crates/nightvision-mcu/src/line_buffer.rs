//! Circular line-buffer for streaming row-by-row MCU processing.
//!
//! Prevents Out-Of-Memory on microcontrollers:
//! Instead of allocating 900KB for full RGB frames, maintains a sliding
//! 3-line window requiring less than 2KB of SRAM for 640-wide VGA video.

/// 3-line circular ring buffer for 3x3 kernel sliding-window processing.
pub struct LineBuffer3<const WIDTH: usize> {
    lines: [[u8; WIDTH]; 3],
    current_line_idx: usize,
    total_lines_pushed: usize,
}

impl<const WIDTH: usize> LineBuffer3<WIDTH> {
    pub const fn new() -> Self {
        Self {
            lines: [[0u8; WIDTH]; 3],
            current_line_idx: 0,
            total_lines_pushed: 0,
        }
    }

    /// Reset line buffer state on frame boundary.
    pub fn reset(&mut self) {
        self.current_line_idx = 0;
        self.total_lines_pushed = 0;
    }

    /// Push a new incoming sensor row into the circular line buffer.
    pub fn push_line(&mut self, new_line: &[u8]) {
        let copy_len = new_line.len().min(WIDTH);
        self.lines[self.current_line_idx][..copy_len].copy_from_slice(&new_line[..copy_len]);

        self.current_line_idx = (self.current_line_idx + 1) % 3;
        self.total_lines_pushed += 1;
    }

    /// Returns true if at least 2 lines have been pushed and 3x3 processing can begin.
    #[inline(always)]
    pub fn is_ready(&self) -> bool {
        self.total_lines_pushed >= 2
    }

    /// Access the 3 active spatial lines: [top_line, mid_line, bot_line].
    #[inline(always)]
    pub fn get_window(&self) -> (&[u8; WIDTH], &[u8; WIDTH], &[u8; WIDTH]) {
        // current_line_idx points to where the NEXT line will be written.
        // So the most recently pushed line is (current_line_idx + 2) % 3 (bottom line).
        // Middle line is (current_line_idx + 1) % 3.
        // Top line is current_line_idx % 3.
        let top = &self.lines[self.current_line_idx];
        let mid = &self.lines[(self.current_line_idx + 1) % 3];
        let bot = &self.lines[(self.current_line_idx + 2) % 3];
        (top, mid, bot)
    }
}

