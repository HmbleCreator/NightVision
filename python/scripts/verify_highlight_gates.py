import numpy as np
from PIL import Image

def evaluate_pipeline(name, img_path, raw_path):
    img = np.array(Image.open(img_path), dtype=np.float32)
    raw = np.array(Image.open(raw_path), dtype=np.float32)
    h, w, _ = raw.shape

    raw_y = (raw[:,:,0]*77.0 + raw[:,:,1]*150.0 + raw[:,:,2]*29.0) / 256.0
    raw_min = np.minimum(np.minimum(raw[:,:,0], raw[:,:,1]), raw[:,:,2])

    face_mask = np.zeros((h, w), dtype=bool)
    face_mask[250:650, 300:600] = True
    skin_mid = face_mask & (raw_y >= 120.0) & (raw_y <= 215.0)
    trans_ring = face_mask & (raw_y >= 220.0) & (raw_y <= 244.0)
    cheek_core = face_mask & (raw_min >= 248.0)

    # Wall blown: x < 180, raw_min >= 240
    wall_blown = (np.arange(w)[None, :] < 180) & (raw_min >= 240.0)

    # RGB means
    rgb_skin = [float(np.mean(img[skin_mid, c])) for c in range(3)]
    rgb_trans = [float(np.mean(img[trans_ring, c])) for c in range(3)]
    rgb_core = [float(np.mean(img[cheek_core, c])) for c in range(3)]
    rgb_wall = [float(np.mean(img[wall_blown, c])) for c in range(3)]

    luma_core = float(np.mean((img[cheek_core, 0]*77 + img[cheek_core, 1]*150 + img[cheek_core, 2]*29) / 256.0))
    luma_mid = float(np.mean((img[skin_mid, 0]*77 + img[skin_mid, 1]*150 + img[skin_mid, 2]*29) / 256.0))

    rb_skin = rgb_skin[0] - rgb_skin[2]
    rb_trans = rgb_trans[0] - rgb_trans[2]
    rb_core = rgb_core[0] - rgb_core[2]
    rb_wall = abs(rgb_wall[0] - rgb_wall[2])

    print(f"=== {name} Pipeline Telemetry ===")
    print(f"Midtone Skin Luma:  {luma_mid:.2f} ADU | RGB: [{rgb_skin[0]:.1f}, {rgb_skin[1]:.1f}, {rgb_skin[2]:.1f}] | R - B = {rb_skin:.2f}")
    print(f"Transition Ring:     RGB: [{rgb_trans[0]:.1f}, {rgb_trans[1]:.1f}, {rgb_trans[2]:.1f}] | R - B = {rb_trans:.2f}")
    print(f"Cheek Core Luma:    {luma_core:.2f} ADU | RGB: [{rgb_core[0]:.1f}, {rgb_core[1]:.1f}, {rgb_core[2]:.1f}] | R - B = {rb_core:.2f}")
    print(f"Blown Wall:         RGB: [{rgb_wall[0]:.1f}, {rgb_wall[1]:.1f}, {rgb_wall[2]:.1f}] | |R - B| = {rb_wall:.2f}")

    # Gate 1: Cheek Core R - B in [12..22] and R > G > B
    gate1 = (12.0 <= rb_core <= 22.0) and (rgb_core[0] > rgb_core[1]) and (rgb_core[1] > rgb_core[2])
    # Gate 2: Transition Ring R - B >= Cheek Core R - B (Zero Moat)
    gate2 = rb_trans >= rb_core
    # Gate 3: Blown Wall |R - B| < 8.0 ADU
    gate3 = rb_wall < 8.0

    print(f"Gate 1 (Cheek Core R-B in [12..22] & R>G>B): {'PASS' if gate1 else 'FAIL'} (R-B={rb_core:.2f}, R>G: {rgb_core[0]>rgb_core[1]}, G>B: {rgb_core[1]>rgb_core[2]})")
    print(f"Gate 2 (Zero Moat: Trans >= Core):          {'PASS' if gate2 else 'FAIL'} (Trans={rb_trans:.2f} >= Core={rb_core:.2f})")
    print(f"Gate 3 (Blown Wall |R-B| < 8.0):            {'PASS' if gate3 else 'FAIL'} (|R-B|={rb_wall:.2f})")
    print()

    return {
        "luma_mid": luma_mid,
        "luma_core": luma_core,
        "rgb_mid": rgb_skin,
        "rgb_trans": rgb_trans,
        "rgb_core": rgb_core,
        "rgb_wall": rgb_wall,
        "rb_skin": rb_skin,
        "rb_trans": rb_trans,
        "rb_core": rb_core,
        "rb_wall": rb_wall,
        "gate1": gate1,
        "gate2": gate2,
        "gate3": gate3,
    }

print("PORTRAIT EVALUATION:")
desk_res = evaluate_pipeline("Desktop FPU", "output/portrait_desktop.png", "data/HighExposureImages/Portrait_Overexposed_Raw.jpg")
mcu_res = evaluate_pipeline("MCU Streaming", "output/portrait_mcu.png", "data/HighExposureImages/Portrait_Overexposed_Raw.jpg")

print("ROOM WINDOW EVALUATION:")
def evaluate_window(name, img_path, raw_path):
    img = np.array(Image.open(img_path), dtype=np.float32)
    raw = np.array(Image.open(raw_path), dtype=np.float32)
    raw_min = np.minimum(np.minimum(raw[:,:,0], raw[:,:,1]), raw[:,:,2])
    blown = raw_min >= 248.0
    rgb_win = [float(np.mean(img[blown, c])) for c in range(3)]
    rb_win = abs(rgb_win[0] - rgb_win[2])
    gate = rb_win < 8.0
    print(f"{name} Blown Window: RGB: [{rgb_win[0]:.1f}, {rgb_win[1]:.1f}, {rgb_win[2]:.1f}] | |R - B| = {rb_win:.2f} | Gate (<8.0): {'PASS' if gate else 'FAIL'}")
    return rb_win

desk_win = evaluate_window("Desktop FPU", "output/window_recovered_desktop.png", "data/HighExposureImages/Room_Window_Overexposed.jpg")
mcu_win = evaluate_window("MCU Streaming", "output/window_recovered_mcu.png", "data/HighExposureImages/Room_Window_Overexposed.jpg")
