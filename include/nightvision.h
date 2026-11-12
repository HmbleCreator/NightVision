/**
 * @file nightvision.h
 * @brief Next-Generation Digital Computational Night Vision Embedded C-API.
 *
 * Designed for microcontrollers (ESP32-S3, STM32H7, RP2040) and camera sensors.
 * Provides line-buffered zero-copy streaming, integer bit-shifts, and precomputed LUTs.
 */

#ifndef NIGHTVISION_H
#define NIGHTVISION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Tactical display phosphor mode */
typedef enum {
    NIGHTVISION_MODE_FULL_COLOR       = 0,
    NIGHTVISION_MODE_WHITE_PHOSPHOR   = 1,
    NIGHTVISION_MODE_GREEN_PHOSPHOR   = 2,
    NIGHTVISION_MODE_RECON_AMBER      = 3,
    NIGHTVISION_MODE_THERMAL_FUSION   = 4
} nightvision_mode_t;

/** Opaque handle to the dynamic streaming night vision engine */
typedef struct nightvision_engine nightvision_engine_t;

/**
 * @brief Allocates and initializes a NightVision engine instance.
 * @param width Native frame width (e.g. 640 for VGA).
 * @param height Native frame height (e.g. 480 for VGA).
 * @param display_mode Tactical display mode (0=FullColor, 1=P45 White, 2=P43 Green).
 * @return Pointer to allocated engine instance, or NULL on error.
 */
nightvision_engine_t* nightvision_create(uint32_t width, uint32_t height, uint8_t display_mode);

/**
 * @brief Frees a previously created NightVision engine instance.
 */
void nightvision_destroy(nightvision_engine_t* handle);

/**
 * @brief Updates the tactical display phosphor mode on the fly.
 */
void nightvision_set_display_mode(nightvision_engine_t* handle, uint8_t display_mode);

/**
 * @brief Resets temporal accumulation and motion tracking buffers.
 */
void nightvision_reset(nightvision_engine_t* handle);

/**
 * @brief Processes an entire RGB8 frame.
 * @param in_rgb Input interleaved RGB buffer (width * height * 3 bytes).
 * @param out_rgb Destination buffer (width * height * 3 bytes).
 * @return 0 on success, negative error code on failure.
 */
int32_t nightvision_process_frame(
    nightvision_engine_t* handle,
    uint32_t width,
    uint32_t height,
    const uint8_t* in_rgb,
    uint8_t* out_rgb
);

/**
 * @brief Zero-copy line-buffered streaming filter for DMA interrupt handlers.
 *
 * Ingests 3 adjacent luminance lines (Top, Mid, Bottom) and executes the
 * physics-based deterministic edge-preserving filter directly into out_line.
 * Requires zero dynamic heap allocations (< 2KB SRAM).
 *
 * @param top_line Pointer to row (Y - 1) of luminance pixels.
 * @param mid_line Pointer to row (Y) of luminance pixels.
 * @param bot_line Pointer to row (Y + 1) of luminance pixels.
 * @param out_line Destination row of filtered, tone-mapped pixels.
 * @param width Row width in pixels.
 * @param noise_lut Optional pointer to 256-byte Poisson noise LUT (or NULL for default).
 * @param gating_lut Optional pointer to 256-byte Reinhard auto-gating LUT (or NULL for default).
 * @return 0 on success, negative error code on failure.
 */
int32_t nightvision_process_line_luma(
    const uint8_t* top_line,
    const uint8_t* mid_line,
    const uint8_t* bot_line,
    uint8_t* out_line,
    uint32_t width,
    const uint8_t* noise_lut,
    const uint8_t* gating_lut
);

/**
 * @brief Returns the library version string.
 */
const char* nightvision_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NIGHTVISION_H */
