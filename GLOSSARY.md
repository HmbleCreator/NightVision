# NightVision Domain Context

Computational imaging and digital night vision pipeline transforming extreme low-light CMOS sensor inputs into high-fidelity tactical and natural full-color output in real time.

## Language

**Frame Processor**:
A stateful computational engine that processes consecutive raw sensor frames through temporal accumulation, spatial denoising, and tone mapping to produce enhanced night vision frames.
_Avoid_: Pipeline, filter chain, handler, service

**Raw Sensor Frame**:
An unenhanced image matrix of raw digital numbers captured directly from an image sensor under low photon flux, characterized by Poisson shot noise, Gaussian read noise, and dark current.
_Avoid_: Input image, raw buffer, capture

**Enhanced Frame**:
A processed frame where photon noise has been suppressed, shadow reflectance recovered, and dynamic range tonemapped for human observation or downstream machine vision.
_Avoid_: Output image, filtered buffer, result

**Tactical Display Mode**:
The spectral and contrast rendering target applied to enhanced luminance, such as P45 White Phosphor, P43 Green Phosphor, or Full-Color digital night vision.
_Avoid_: Color scheme, theme, filter mode, LUT mode

**Digital Auto-Gating**:
Instantaneous software-driven highlight compression that prevents observer blinding and dynamic range clipping during sudden bright optical flares (headlights, muzzle flashes).
_Avoid_: Flare guard, highlight limiter, clamp

**Temporal Accumulator**:
The recursive stateful module that integrates photon flux across consecutive temporal frames while adapting blending weights to prevent motion ghosting.
_Avoid_: History buffer, frame blender, averager

**Image Signal Processor (ISP)**:
The deep computational engine that transforms raw sensor frames through color space conversion, edge-preserving denoising, illumination recovery, and tone mapping using pre-allocated planar scratch buffers.
_Avoid_: Filter pipeline, image processor, graphics engine

