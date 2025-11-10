import numpy as np
from numpy.fft import rfft, irfft
from .model import pairwise_kernel, normalize_messages, beliefs_from_messages

def _fft_convolve_label(m_in, ker_fft):
    """
    Convolution over label axis using FFT (real FFT).
    m_in: (..., L) float32 message
    ker_fft: rfft of kernel length L
    Returns: (..., L) convolved result
    """
    L = m_in.shape[-1]
    M = rfft(m_in, axis=-1)
    out = irfft(M * ker_fft, n=L, axis=-1)
    # clip small negatives due to numerical errors
    out = np.maximum(out, 1e-32)
    return out.astype(np.float32)

def bp_denoise(psi_cost, alpha, max_bp_iter=200, tol=1e-4, init_messages=None):
    """
    Loopy BP on 4-neighbor grid with FFT label-convolutions.
    psi_cost: (H, W, L) unary cost (data term)
    alpha: pairwise weight in exp(-alpha*(xi-xj)^2)
    Returns: beliefs (H, W, L) and messages dict
    """
    H, W, L = psi_cost.shape
    ker = pairwise_kernel(L=L, alpha=alpha)
    ker_fft = rfft(ker, n=L)

    # initialize messages as uniform if none
    if init_messages is None:
        uni = np.ones((H, W, L), dtype=np.float32) / L
        messages = {d: uni.copy() for d in ['up', 'down', 'left', 'right']}
    else:
        messages = {k: normalize_messages(v) for k, v in init_messages.items()}

    for t in range(max_bp_iter):
        old = {k: v.copy() for k, v in messages.items()}

        eps = 1e-32
    
        # LOG-domain unary: log_u = -psi_cost
        log_u = -psi_cost.astype(np.float32)
    
        # log of messages
        logM = {d: np.log(np.maximum(messages[d], eps)) for d in ['up', 'down', 'left', 'right']}
    
        # total incoming log-product
        log_prod_all = log_u + logM['up'] + logM['down'] + logM['left'] + logM['right']
    
        # 4-neighbor shifts
        def roll(a, sh):
            return np.roll(a, shift=sh, axis=(0,1))
    
        new = {}
    
        # ---- UP message (sender below) ----
        log_sender = log_prod_all - logM['down']       # remove reverse message
        log_sender = roll(log_sender, (-1, 0))
        log_sender -= log_sender.max(axis=-1, keepdims=True)
        sender = np.exp(log_sender)
        msg_up = _fft_convolve_label(sender, ker_fft)
        new['up'] = normalize_messages(msg_up)
    
        # ---- DOWN message (sender above) ----
        log_sender = log_prod_all - logM['up']
        log_sender = roll(log_sender, (1, 0))
        log_sender -= log_sender.max(axis=-1, keepdims=True)
        sender = np.exp(log_sender)
        msg_down = _fft_convolve_label(sender, ker_fft)
        new['down'] = normalize_messages(msg_down)
    
        # ---- LEFT message (sender right) ----
        log_sender = log_prod_all - logM['right']
        log_sender = roll(log_sender, (0, -1))
        log_sender -= log_sender.max(axis=-1, keepdims=True)
        sender = np.exp(log_sender)
        msg_left = _fft_convolve_label(sender, ker_fft)
        new['left'] = normalize_messages(msg_left)
    
        # ---- RIGHT message (sender left) ----
        log_sender = log_prod_all - logM['left']
        log_sender = roll(log_sender, (0, 1))
        log_sender -= log_sender.max(axis=-1, keepdims=True)
        sender = np.exp(log_sender)
        msg_right = _fft_convolve_label(sender, ker_fft)
        new['right'] = normalize_messages(msg_right)
    
        # ---- DAMPING (CRUCIAL) ----
        rho = 0.5  # damping factor
        for d in ['up','down','left','right']:
            messages[d] = (1 - rho) * old[d] + rho * new[d]
    
        # ---- Convergence check ----
        diff = 0.0
        for d in ['up','down','left','right']:
            diff += np.mean(np.abs(messages[d] - old[d]))
        if diff < tol:
            break


    beliefs = beliefs_from_messages(psi_cost, messages, L=L)
    return beliefs, messages
