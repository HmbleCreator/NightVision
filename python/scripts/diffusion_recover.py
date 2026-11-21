import os
import sys
import time
import argparse
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage
import torch
from optimum.intel.openvino import OVStableDiffusionInpaintPipeline
from diffusers import LCMScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="NightVision Stage 2: Generative Highlight Recovery")
    parser.add_argument("--image", type=str, default="data/HighExposureImages/Portrait_Overexposed_Raw.jpg", help="Raw input image")
    parser.add_argument("--stage1", type=str, default="output/portrait_desk.png", help="Stage 1 ISP output")
    parser.add_argument("--output", type=str, default="output/portrait_diffusion.png", help="Output path")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU", "AUTO"], help="Inference device")
    parser.add_argument("--max_clip_ratio", type=float, default=0.45, help="Blowout guard threshold (0.0 to 1.0)")
    parser.add_argument("--force-inpaint", action="store_true", help="Bypass blowout guard")
    parser.add_argument("--steps", type=int, default=8, help="Number of inference steps")
    parser.add_argument("--strength", type=float, default=0.75, help="Inpainting strength")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def extract_raw_mask(raw_img_path, clip_thresh=248.0, dilate_px=5, sigma=2.5):
    raw_img = Image.open(raw_img_path).convert('RGB')
    raw_np = np.array(raw_img, dtype=np.float32)
    raw_min = np.min(raw_np, axis=2)
    
    mask_binary = (raw_min >= clip_thresh)
    clip_ratio = float(np.mean(mask_binary))
    
    struct = scipy.ndimage.generate_binary_structure(2, 1)
    dilated = scipy.ndimage.binary_dilation(mask_binary, structure=struct, iterations=dilate_px)
    
    alpha = scipy.ndimage.gaussian_filter(dilated.astype(np.float32), sigma=sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    return mask_binary, dilated, alpha, clip_ratio

def prepare_gentle_seed(s1_np, alpha, pull_factor=0.88, sigma=24.0):
    s1_blur = scipy.ndimage.gaussian_filter(s1_np.astype(np.float32), sigma=(sigma, sigma, 0))
    s1_seed = s1_blur * pull_factor
    alpha_3d = np.expand_dims(alpha, axis=-1)
    init_np = (1.0 - alpha_3d) * s1_np.astype(np.float32) + alpha_3d * s1_seed
    return np.clip(np.round(init_np), 0.0, 255.0).astype(np.uint8)

def load_accelerated_pipeline(device_pref="GPU"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    ov_dir = os.path.join(script_dir, "..", "models", "sd15_inpaint_lcm_ov")
    ov_dir = os.path.abspath(ov_dir)
    
    active_device = f"Intel Iris Xe Graphics ({device_pref} via OpenVINO)"
    print(f"Loading pre-exported OpenVINO IR from {ov_dir} targeting {device_pref}...")
    t0 = time.time()
    
    pipe = OVStableDiffusionInpaintPipeline.from_pretrained(
        ov_dir,
        device=device_pref,
        compile=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.to(device_pref)
    pipe.compile()
    
    print(f"Pipeline ready on {active_device} in {time.time() - t0:.2f}s")
    return pipe, active_device

def run_recovery(raw_path, stage1_path, out_path, pipe, active_device, 
                 steps=8, strength=0.75, guidance_scale=1.5, seed=42, 
                 max_clip_ratio=0.45, force_inpaint=False, prompt=None, neg_prompt=None):
    
    raw_img = Image.open(raw_path).convert('RGB')
    s1_img = Image.open(stage1_path).convert('RGB')
    orig_w, orig_h = raw_img.size
    
    mask_binary, dilated, alpha_native, clip_ratio = extract_raw_mask(raw_path, clip_thresh=248.0, dilate_px=5, sigma=2.5)
    print(f"Raw image {orig_w}x{orig_h}: {mask_binary.sum()} triple-clipped pixels ({clip_ratio*100:.2f}%)")
    
    if clip_ratio > max_clip_ratio and not force_inpaint:
        print(f"[BLOWOUT SAFETY GUARD] Clipped ratio {clip_ratio*100:.2f}% > threshold {max_clip_ratio*100:.1f}%.")
        print("Bypassing generative pass to prevent hallucinated spires/pillars. Returning clean Stage 1 output.")
        s1_img.save(out_path)
        return {
            "status": "Bypassed (RequiresHardwareAec)",
            "clip_ratio": clip_ratio,
            "latency_s": 0.0,
            "device": active_device,
            "unmasked_max_diff": 0,
        }
        
    s1_np = np.array(s1_img, dtype=np.uint8)
    init_seeded_np = prepare_gentle_seed(s1_np, alpha_native, pull_factor=0.88, sigma=24.0)
    init_seeded = Image.fromarray(init_seeded_np)
    
    inf_w, inf_h = 512, 680
    init_scaled = init_seeded.resize((inf_w, inf_h), Image.Resampling.LANCZOS)
    
    mask_scaled_np = scipy.ndimage.zoom(dilated.astype(np.float32), (inf_h/orig_h, inf_w/orig_w), order=0) > 0.5
    mask_scaled = Image.fromarray((mask_scaled_np * 255).astype(np.uint8), mode='L')
    
    if prompt is None:
        prompt = "natural photograph of a person, realistic skin texture, soft cheek highlight, warm skin tones, photographic detail, matching lighting"
    if neg_prompt is None:
        neg_prompt = "dark blotches, weird objects, pillars, extra shapes, distorted geometry, plastic skin, artifacts, dark patches, shadows"
        
    print(f"Running full-frame inpainting ({inf_w}x{inf_h}) with steps={steps}, strength={strength}, guidance={guidance_scale}...")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.time()
    
    res = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=init_scaled,
        mask_image=mask_scaled,
        num_inference_steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
        generator=gen,
    ).images[0]
    
    latency = time.time() - t0
    print(f"Inference completed in {latency:.2f}s on {active_device}")
    
    res_native = res.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    res_np = np.array(res_native, dtype=np.float32)
    
    alpha_3d = np.expand_dims(alpha_native, axis=-1)
    comp_np = (1.0 - alpha_3d) * s1_np.astype(np.float32) + alpha_3d * res_np
    comp_np = np.clip(np.round(comp_np), 0.0, 255.0).astype(np.uint8)
    final_img = Image.fromarray(comp_np)
    
    final_img.save(out_path)
    print(f"Saved recovered image to {out_path}")
    
    unmasked = (alpha_native == 0.0)
    diff_unmasked = int(np.max(np.abs(comp_np[unmasked].astype(np.int32) - s1_np[unmasked].astype(np.int32))))
    print(f"Bit-exact unmasked check: max diff = {diff_unmasked} ADU -> {'PASS' if diff_unmasked == 0 else 'FAIL'}")
    
    return {
        "status": "Recovered",
        "clip_ratio": clip_ratio,
        "latency_s": latency,
        "device": active_device,
        "unmasked_max_diff": diff_unmasked,
    }

def main():
    args = parse_args()
    pipe, active_device = load_accelerated_pipeline(args.device)
    
    run_recovery(
        raw_path=args.image,
        stage1_path=args.stage1,
        out_path=args.output,
        pipe=pipe,
        active_device=active_device,
        steps=args.steps,
        strength=args.strength,
        guidance_scale=args.guidance,
        seed=args.seed,
        max_clip_ratio=args.max_clip_ratio,
        force_inpaint=args.force_inpaint,
    )

if __name__ == "__main__":
    main()
