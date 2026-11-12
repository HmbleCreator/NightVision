//! Hardware DMA Ping-Pong Double-Buffer Streaming Module.
//!
//! Designed for camera sensor DMA controllers (ESP32-S3 GDMA, STM32H7 DCMI + DMA2D).
//! Facilitates lock-free, zero-copy line ingestion: DMA streams row N into the receiving bank
//! while the CPU/ISP engine filters row N-1 from the processing bank.

use crate::deterministic_filter::filter_line_deterministic;
use crate::lut::DeterministicNoiseLut;

/// Active ping-pong DMA bank identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaBank {
    BankA,
    BankB,
}

/// Zero-copy ping-pong DMA double-buffer for embedded camera drivers.
pub struct DmaPingPongBuffer<const WIDTH: usize> {
    pub bank_a: [u8; WIDTH],
    pub bank_b: [u8; WIDTH],
    active_rx: DmaBank,
}

impl<const WIDTH: usize> DmaPingPongBuffer<WIDTH> {
    pub const fn new() -> Self {
        Self {
            bank_a: [0u8; WIDTH],
            bank_b: [0u8; WIDTH],
            active_rx: DmaBank::BankA,
        }
    }

    /// Reset double-buffer memory.
    pub fn reset(&mut self) {
        self.bank_a.fill(0);
        self.bank_b.fill(0);
        self.active_rx = DmaBank::BankA;
    }

    /// Returns a mutable slice to the bank currently being populated by hardware DMA.
    /// Hardware camera drivers (e.g. ESP-IDF `cam_hal` or STM32 `HAL_DCMI_Start_DMA`)
    /// configure this buffer pointer as the DMA destination.
    #[inline(always)]
    pub fn dma_rx_buffer_mut(&mut self) -> &mut [u8; WIDTH] {
        match self.active_rx {
            DmaBank::BankA => &mut self.bank_a,
            DmaBank::BankB => &mut self.bank_b,
        }
    }

    /// Returns an immutable slice to the completed bank ready for CPU ISP execution.
    #[inline(always)]
    pub fn processing_buffer(&self) -> &[u8; WIDTH] {
        match self.active_rx {
            DmaBank::BankA => &self.bank_b,
            DmaBank::BankB => &self.bank_a,
        }
    }

    /// Swaps the active DMA and processing banks upon receiving a DMA Transfer Complete (`DMA_TC`) interrupt.
    #[inline(always)]
    pub fn swap_banks(&mut self) {
        self.active_rx = match self.active_rx {
            DmaBank::BankA => DmaBank::BankB,
            DmaBank::BankB => DmaBank::BankA,
        };
    }

    /// Filters the ready processing row using adjacent spatial context and the deterministic noise LUT.
    #[inline(always)]
    pub fn filter_processing_line(
        &self,
        top_line: &[u8; WIDTH],
        bot_line: &[u8; WIDTH],
        out_line: &mut [u8; WIDTH],
        noise_lut: &DeterministicNoiseLut,
    ) {
        let mid_line = self.processing_buffer();
        filter_line_deterministic(top_line, mid_line, bot_line, out_line, noise_lut);
    }
}

