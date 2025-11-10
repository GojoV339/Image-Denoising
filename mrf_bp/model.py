import numpy as np

def build_label_space(L=256):
    # labels 0..L-1 (discrete gray values)
    return np.arange(L, dtype=np.int32)

def unary_cost_stack(Yk, L=256, sigma2=15.0**2):
    """
    Yk: (K, H, W) grayscale noisy images with values in [0..255]
    returns unary potentials psi_i(xi) for all labels xi
    shape: (H, W, L), psi_i(x) = (1/(2*sigma2)) * sum_k (x - y_k)^2
    (as costs; paper uses energy in exp(-cost))
    """
    K, H, W = Yk.shape
    labels = np.arange(L, dtype=np.float32)  # (L,)
    # (H, W, L) broadcast compute sum_k (x - y_k)^2
    psi = np.zeros((H, W, L), dtype=np.float32)
    for k in range(K):
        y = Yk[k]  # (H, W)
        # (H, W, L): (x - y)^2
        diff2 = (labels[None, None, :] - y[..., None])**2
        psi += diff2
    psi *= (1.0/(2.0 * sigma2))
    return psi  # this is the "data term" cost at each pixel & label

def pairwise_kernel(L=256, alpha=0.005):
    """
    f(xi - xj; alpha) = exp( -alpha * (xi - xj)^2 ), length L circular (periodic in label domain).
    Used for 1D convolution over label values via FFT.
    """
    xs = np.arange(L, dtype=np.float32)
    # circular distances on label ring (paper assumes values 0..L-1; use min(|d|, L-|d|))
    d = np.minimum(xs, L - xs)
    ker = np.exp(-alpha * (d**2))
    # normalize kernel to sum=1 to improve numerical stability
    s = np.sum(ker)
    if s > 0:
        ker /= s
    return ker.astype(np.float32)

def normalize_messages(m, eps=1e-12):
    """
    Normalize message over labels to sum=1.
    m: (..., L)
    """
    s = np.sum(m, axis=-1, keepdims=True)
    s = np.maximum(s, eps)
    return m / s

import numpy as np

def beliefs_from_messages(psi_cost, messages, L=256, eps=1e-32):
    H, W, _ = psi_cost.shape

    # log-belief = -psi_cost + sum(log messages)
    log_b = -psi_cost.astype(np.float32)
    for d in ['up', 'down', 'left', 'right']:
        log_b += np.log(np.maximum(messages[d], eps))

    # stabilize
    max_log = np.max(log_b, axis=-1, keepdims=True)
    log_b -= max_log

    b = np.exp(log_b)

    # normalize per pixel
    norm = np.sum(b, axis=-1, keepdims=True)
    b = b / np.maximum(norm, eps)

    return b.astype(np.float32)


def map_from_beliefs(beliefs, labels=None):
    if labels is None:
        labels = np.arange(beliefs.shape[-1], dtype=np.int32)
    idx = beliefs.argmax(axis=-1)
    return labels[idx].astype(np.uint8)
