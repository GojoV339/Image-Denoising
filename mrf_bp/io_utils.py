from pathlib import Path
import numpy as np
import cv2

def imread_gray_uint8(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.uint8)

def to_gray_uint8(img_rgb_or_gray_uint8):
    if img_rgb_or_gray_uint8.ndim == 2:
        return img_rgb_or_gray_uint8
    g = cv2.cvtColor(img_rgb_or_gray_uint8, cv2.COLOR_BGR2GRAY)
    return g

def stack_noisy_from_clean(clean_u8, K=1, sigma=15.0, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=(K,)+clean_u8.shape).astype(np.float32)
    Y = np.clip(clean_u8[None, ...].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Y  # (K,H,W) uint8

def save_u8_png(path, u8):
    cv2.imencode(".png", u8)[1].tofile(path)
