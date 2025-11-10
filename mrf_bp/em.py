import numpy as np
from numpy.fft import rfft, irfft
from .model import pairwise_kernel

def _fft_conv(m, ker_fft):
    L = m.shape[-1]
    return np.maximum(irfft(rfft(m, axis=-1)*ker_fft, n=L, axis=-1), 1e-32)

def _pairwise_expectation_post(messages_i_to_j, messages_j_to_i, alpha, L=256):
    """
    Compute ⟨phi(xi-xj)⟩_post for quadratic phi (x^2) using FFT trick in paper.
    Here phi(d) = d^2 (with circular distances), f(d)=exp(-alpha d^2)
    Returns tuple (G_sum, H_sum) per edge; caller sums over edges.
    """
    xs = np.arange(L, dtype=np.float32)
    d = np.minimum(xs, L - xs)
    phi = (d**2).astype(np.float32)
    ker = np.exp(-alpha*phi)
    ker_fft = rfft(ker, n=L)

    # messages are over labels: m_i(x), m_j(x)
    # g(xi) = sum_xj phi(xi-xj) f(xi-xj) m_i_to_j(xj)
    # h(xi) = sum_xj f(xi-xj)                 m_i_to_j(xj)
    # then G = sum_xi g(xi) m_j_to_i(xi), H = sum_xi h(xi) m_j_to_i(xi)
    # compute via convolution in label domain
    # Build conv(phi * f) efficiently: compute conv with (phi*f) by direct domain multiply then FFT.
    phi_f = phi * ker
    phi_f_fft = rfft(phi_f, n=L)

    h = _fft_conv(messages_i_to_j, ker_fft)           # (.., L)
    g = _fft_conv(messages_i_to_j, phi_f_fft)         # (.., L)

    G = np.sum(g * messages_j_to_i, axis=-1)  # (...,)
    H = np.sum(h * messages_j_to_i, axis=-1)  # (...,)
    return G, H

def em_update(Yk, beliefs, messages, sigma2_curr, alpha_curr, max_em_iter=50, tol=1e-4):
    """
    EM updates for sigma^2 and alpha as in paper.
    Yk: (K,H,W) noisy copies
    beliefs: (H,W,L)
    messages: dict of directional messages (H,W,L)
    Returns: sigma2_new, alpha_new
    """
    K, H, W = Yk.shape
    L = beliefs.shape[-1]
    labels = np.arange(L, dtype=np.float32)

    sigma2 = sigma2_curr
    alpha = alpha_curr

    for _ in range(max_em_iter):
        # E-step expectations (posterior):
        # <psi_i(xi)> = sum_xi [ sum_k (xi - y_k)^2 ] * b_i(xi)
        # First compute mean over k of squared diffs per label
        # We will aggregate per pixel.
        psi_exp = 0.0
        for k in range(K):
            y = Yk[k][..., None]  # (H,W,1)
            diff2 = (labels[None,None,:] - y)**2
            psi_exp += np.sum(diff2 * beliefs, axis=-1)  # (H,W)
        # sigma2_new = 1/(N*K) * sum_i <psi_i(xi)>
        sigma2_new = float(psi_exp.sum() / (H*W*K))
        sigma2_new = max(sigma2_new, 1e-6)

        # <phi(xi-xj)>_post per edge: need G/H for all edges and sum
        # we approximate using directional pairs:
        dirs = [('up','down',( -1, 0)), ('down','up',(1,0)), ('left','right',(0,-1)), ('right','left',(0,1))]
        G_sum = 0.0
        H_sum = 0.0
        for d_send, d_recv, shift in dirs:
            # messages[d_send] are from neighbor into the pixel; to align pairs,
            # roll receiver to reference orientation; periodic boundary as paper.
            m_recv = messages[d_recv]  # receiver-facing messages at target
            m_send = np.roll(messages[d_send], shift=shift, axis=(0,1))
            g, h = _pairwise_expectation_post(m_send, m_recv, alpha, L=L)
            G_sum += g.sum()
            H_sum += h.sum()
        # We need to find alpha_new from equation: sum_ij <phi>_post = sum_ij <phi>_prior(alpha_new)
        # We approximate <phi>_prior(alpha) under translational symmetry (paper Sec.5); here we fit alpha
        # by 1D search (bisection/line-search) that matches ratio G/H on posterior to prior.
        # For practicality, we match the expected value <phi> numerically using label kernel.
        xs = np.arange(L, dtype=np.float32)
        d = np.minimum(xs, L - xs)
        phi = d**2
        def expected_phi_prior(a):
            ker = np.exp(-a*phi)
            ker /= np.maximum(ker.sum(), 1e-12)
            # for 4-neighbor isotropic model, expectation per edge approximates E[phi] under 1D marginal
            return float((phi * ker).sum())

        target = (G_sum / max(H_sum, 1e-12))  # posterior estimate per-edge phi
        # bisection on alpha in [0, 2*alpha_curr] as in paper
        lo, hi = 0.0, max(2.0*alpha, 1e-6)
        for _it in range(40):
            mid = 0.5*(lo+hi)
            val = expected_phi_prior(mid)
            if val > target:
                lo = mid
            else:
                hi = mid
        alpha_new = 0.5*(lo+hi)
        alpha_new = max(alpha_new, 1e-8)

        delta = abs(alpha_new - alpha) + abs(sigma2_new - sigma2)/max(1.0, sigma2)
        alpha, sigma2 = alpha_new, sigma2_new
        if delta < tol:
            break

    return sigma2, alpha
