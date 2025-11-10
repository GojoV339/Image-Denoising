import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def psnr_u8(pred_u8, gt_u8):
    return float(peak_signal_noise_ratio(gt_u8, pred_u8, data_range=255))

def ssim_u8(pred_u8, gt_u8):
    return float(structural_similarity(gt_u8, pred_u8, data_range=255))
