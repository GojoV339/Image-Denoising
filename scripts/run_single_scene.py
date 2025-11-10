import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from mrf_bp.io_utils import imread_gray_uint8, stack_noisy_from_clean, save_u8_png
from mrf_bp.model import build_label_space, unary_cost_stack, beliefs_from_messages, map_from_beliefs
from mrf_bp.bp_fft import bp_denoise
from mrf_bp.em import em_update
from mrf_bp.metrics import psnr_u8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True, help="Path to clean grayscale image (uint8)")
    ap.add_argument("--K", type=int, default=1, help="#noisy copies")
    ap.add_argument("--sigma", type=float, default=15.0, help="AWGN std (uint8 scale)")
    ap.add_argument("--iters_bp", type=int, default=200)
    ap.add_argument("--iters_em", type=int, default=30)
    ap.add_argument("--alpha0", type=float, default=0.005)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    clean = imread_gray_uint8(args.clean)  # (H,W) uint8
    Yk = stack_noisy_from_clean(clean, K=args.K, sigma=args.sigma, seed=0)  # (K,H,W) uint8

    # init parameters
    sigma2 = float(args.sigma**2)
    alpha = float(args.alpha0)

    # build unary costs
    psi = unary_cost_stack(Yk.astype(np.float32), L=256, sigma2=sigma2)

    # BP to get beliefs
    beliefs, messages = bp_denoise(psi, alpha, max_bp_iter=args.iters_bp, tol=1e-4)
    # EM to refine alpha, sigma2
    sigma2, alpha = em_update(Yk.astype(np.float32), beliefs, messages, sigma2, alpha, max_em_iter=args.iters_em, tol=1e-4)

    # re-run BP with updated parameters
    psi = unary_cost_stack(Yk.astype(np.float32), L=256, sigma2=sigma2)
    beliefs, messages = bp_denoise(psi, alpha, max_bp_iter=args.iters_bp, tol=1e-4)
    x_map = map_from_beliefs(beliefs)  # uint8

    save_u8_png(str(out/"denoised.png"), x_map)
    save_u8_png(str(out/"clean.png"), clean)
    save_u8_png(str(out/"noisy_example.png"), Yk[0])

    print(f"PSNR noisy vs clean: {psnr_u8(Yk[0], clean):.2f} dB")
    print(f"PSNR denoised vs clean: {psnr_u8(x_map, clean):.2f} dB")
    print(f"alpha={alpha:.6f}, sigma={np.sqrt(sigma2):.3f}")

if __name__ == "__main__":
    main()
