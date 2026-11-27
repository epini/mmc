#!/usr/bin/env python3
"""Step h: optical scattering tensors of the WM elements (decided 26/11, see CLAUDE.md 2h).

    python optics.py              # conda env pyxoptoenv (generalized_ade); after warp_tensors.py

Mapping, WM elements only (label 4), no FA threshold, OFA = FA:
  D_opt has the eigenvectors and the eigenvalue ratios of the (FMRIB58-rescaled) DTI tensor, scaled
  so that mean(eig D_opt) = D_MEAN, the measured WM value (D_perp 1.69e-2, D_par 2.47e-2 mm^2/ps,
  n = 1.39, g = 0.97). The scattering tensor M with the same eigenvectors is obtained by inverting
  the generalized ADE (generalized_ade.d_tensor_ade, scalar HG g). The map shape(M) -> shape(D) is
  scale-free (M -> kM gives D -> D/k): a lookup table on the principal values M = (1, p, q),
  1 <= p <= q, gives the D ratios (Dx/Dz, Dy/Dz) and the scale; it is inverted by linear
  interpolation on the triangulated (log Dx/Dz, log Dy/Dz) plane; shapes just outside the convex hull
  of the table (nearly uniaxial or nearly isotropic D, on the edges p = 1, p = q) take the nearest
  table node in the same log plane. The smallest mu goes with the largest D (fast axis).

Checks printed (and in OUT/optics_checks.txt):
  1. d_tensor_full (the same series written for a full M in the lab frame) on a rotated M: D has
     the eigenvectors of M, eigenvalues = d_tensor_ade on the principal values;
  2. LUT inversion round trip on random shapes, through d_tensor_ade;
  3. his measured D_par, D_perp -> mu's_par, mu's_perp (paper: 2.6, 4.4 1/mm), LUT and direct root.
Output OUT/optics_wm.npz: M (ne, 6) in 1/mm, MMC packing, mesh frame; for non-WM elements zeros
(the run script puts mu_s I of the label there); musr_par/perp-like principal values per element.
"""

import math
import os
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.optimize import brentq
from scipy.special import gammaln, lpmv

from generalized_ade import d_tensor_ade
from generalized_ade.quadrature import gauss_legendre

from common import OUT, mat_to_sym6, sym6_to_mat

N_WM, G_WM = 1.39, 0.97
D_PERP, D_PAR = 1.69e-2 * 1e3, 2.47e-2 * 1e3          # mm^2/ns (measured, mm^2/ps x 1e3)
D_MEAN = (2 * D_PERP + D_PAR) / 3                       # 19.5 mm^2/ns
MUSR_PAPER = (2.6, 4.4)                                 # mu's par, perp (1/mm), Pini et al. 2025
C0 = 299.792458                                         # mm/ns


def d_tensor_full(n, M, g, nchi=200, nphi=512, lmax=101, tol=1e-7):
    """Generalized-ADE diffusion tensor (mm^2/ns) for a full symmetric scattering tensor M (1/mm) in the
    lab frame: the series of generalized_ade.d_tensor_ade with the cross terms D_ij (frame check only)."""
    v = C0 / n
    chi, wchi = gauss_legendre(nchi)
    phi = np.arange(nphi) * (2 * np.pi / nphi)
    PHI, CHI = np.meshgrid(phi, chi, indexing="xy")
    S = np.sqrt(np.maximum(0.0, 1 - CHI ** 2))
    s = np.stack([S * np.cos(PHI), S * np.sin(PHI), CHI])
    W = wchi[:, None] * (2 * np.pi / nphi)
    invmu = 1.0 / np.einsum("iab,ij,jab->ab", s, M, s)
    lavg = (invmu * W).sum() / (4 * np.pi)
    pref = v / (4 * np.pi * lavg)
    D = pref * np.einsum("iab,jab,ab->ij", s, s, invmu ** 2 * W)
    b = s * invmu * W
    prev = D.copy()
    for ell in range(1, lmax + 1, 2):
        coeff = g ** ell / (1 - g ** ell)
        mpos = np.arange(ell + 1)
        e = np.exp(1j * np.outer(mpos, phi))
        acc = np.einsum("iab,mb->iam", b, e)                          # (3, nchi, m)
        h = np.zeros((3, ell + 1), complex)
        for k, m in enumerate(mpos):
            lognorm = 0.5 * (math.log(2 * ell + 1) - math.log(2.0) + gammaln(ell - m + 1) - gammaln(ell + m + 1))
            ylm = np.exp(lognorm) * lpmv(m, ell, chi) / math.sqrt(2 * np.pi)
            h[:, k] = acc[:, :, k] @ ylm
        cross = np.real(np.outer(h[:, 0], np.conj(h[:, 0])) + 2 * h[:, 1:] @ np.conj(h[:, 1:]).T)
        D = D + pref * coeff * cross
        if ell >= 15 and np.abs(D - prev).max() <= tol * np.abs(D).max():
            break
        prev = D.copy()
    return D


def build_lut(qmax=300.0, n=48):
    """Principal values M = (1, p, q), 1 <= p <= q <= qmax (log grid, boundaries p = 1 and p = q included).
    Returns arrays logp, logq, log(Dx/Dz), log(Dy/Dz), log mean(D) for M = (1, p, q)."""
    lq = np.linspace(0, np.log(qmax), n)
    rows = []
    for i, b in enumerate(lq):
        for a in np.linspace(0, b, i + 1) if i > 0 else [0.0]:
            dx, dy, dz = d_tensor_ade(N_WM, 1.0, float(np.exp(a)), float(np.exp(b)), G_WM)
            rows.append((a, b, np.log(dx / dz), np.log(dy / dz), np.log((dx + dy + dz) / 3)))
    return np.array(rows)


class Inverter:
    """D principal values (any order) -> M principal values (same order), with mean(D) as given."""

    def __init__(self, lut):
        pts = lut[:, 2:4]
        self.f = LinearNDInterpolator(pts, lut[:, [0, 1, 4]])
        self.fnn = NearestNDInterpolator(pts, lut[:, [0, 1, 4]])
        self.amax = pts[:, 0].max()

    def __call__(self, d):
        d = np.atleast_2d(d)
        order = np.argsort(-d, axis=1)                         # largest D first (fast axis)
        ds = np.take_along_axis(d, order, 1)
        la = np.log(ds[:, 0] / ds[:, 2])
        lb = np.log(ds[:, 1] / ds[:, 2])
        clip = la > self.amax
        if clip.any():                                         # beyond the table: keep the shape, cap the ratio
            lb[clip] *= self.amax / la[clip]
            la[clip] = self.amax
        x = np.stack([la, lb], 1)
        r = self.f(x)
        out_hull = np.isnan(r).any(1)                          # just outside the hull: nearest node (log plane)
        if out_hull.any():
            r[out_hull] = self.fnn(x[out_hull])
        self.last_nn = out_hull
        lp, lq, lmd = r[:, 0], r[:, 1], r[:, 2]
        k = np.exp(lmd) / ds.mean(1)                           # M -> kM gives D -> D / k
        ms = k[:, None] * np.stack([np.ones_like(lp), np.exp(lp), np.exp(lq)], 1)
        out = np.empty_like(ms)
        np.put_along_axis(out, order, ms, 1)
        return out, clip


def main():
    t0 = time.time()
    log = []

    def say(s):
        print(s)
        log.append(s)

    # 1. frame check on a rotated, fully anisotropic M
    rng = np.random.default_rng(1)
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    mu = np.array([80.0, 120.0, 190.0])
    Mr = Q @ np.diag(mu) @ Q.T
    Dr = d_tensor_full(N_WM, Mr, G_WM)
    wd, Vd = np.linalg.eigh(Dr)
    dref = np.array(d_tensor_ade(N_WM, *mu, G_WM))
    align = np.abs(np.sum(Vd[:, ::-1] * Q, 0))                 # largest D <-> smallest mu
    say("1. rotated M (eig %s 1/mm): D eigenvalues %s vs d_tensor_ade on the principal values %s mm^2/ns;"
        % (mu, np.round(wd[::-1], 4), np.round(dref, 4)))
    say("   |cos| between D and M eigenvectors (fast axis = smallest mu): %s; off-diagonal of Q^T D Q / max: %.1e"
        % (np.round(align, 8), np.abs((Q.T @ Dr @ Q) - np.diag(np.diag(Q.T @ Dr @ Q))).max() / np.abs(Dr).max()))

    # 2. lookup table and round trip
    lut = build_lut()
    inv = Inverter(lut)
    say("2. LUT: %d shapes M = (1, p, q), q <= %.0f; D ratio Dx/Dz up to %.1f (%.0f s)"
        % (len(lut), np.exp(lut[:, 1].max()), np.exp(lut[:, 2].max()), time.time() - t0))
    err = []
    for _ in range(40):
        m = np.sort(np.exp(rng.uniform(0, np.log(60), 3)))
        d = np.array(d_tensor_ade(N_WM, *m, G_WM))
        mi, _ = inv(d)
        d2 = np.array(d_tensor_ade(N_WM, *mi[0], G_WM))
        err.append(np.abs(d2 / d - 1).max())
    say("   round trip D -> M (LUT) -> D on 40 random shapes: max relative error %.1e, median %.1e"
        % (max(err), np.median(err)))

    # 3. his measured uniaxial D
    dm = np.array([D_PAR, D_PERP, D_PERP])
    mi, _ = inv(dm)
    musr = mi[0] * (1 - G_WM)
    # direct: uniaxial M = (m_par, m_perp, m_perp), ratio from root finding, scale from the mean
    f = lambda r: (lambda d: d[0] / d[1])(d_tensor_ade(N_WM, 1.0, r, r, G_WM)) - D_PAR / D_PERP
    r = brentq(f, 1.0, 50.0)
    dd = np.array(d_tensor_ade(N_WM, 1.0, r, r, G_WM))
    k = dd.mean() / dm.mean()
    say("3. measured D_par %.1f, D_perp %.1f mm^2/ns (n %.2f, g %.2f):" % (D_PAR, D_PERP, N_WM, G_WM))
    say("   LUT:    mu_s par %.2f, perp %.2f 1/mm -> mu's par %.3f, perp %.3f 1/mm" % (mi[0, 0], mi[0, 1], musr[0], musr[1]))
    say("   direct: mu's par %.3f, perp %.3f 1/mm (ratio %.4f)" % (k * (1 - G_WM), k * r * (1 - G_WM), r))
    say("   paper (MC fit): mu's par %.1f, perp %.1f 1/mm; naive v/(3D): par %.3f, perp %.3f 1/mm"
        % (MUSR_PAPER[0], MUSR_PAPER[1], C0 / N_WM / (3 * D_PAR), C0 / N_WM / (3 * D_PERP)))
    say("   isotropic WM with the same mean D: mu's %.3f 1/mm, mu_s %.2f 1/mm"
        % (C0 / N_WM / (3 * D_MEAN), C0 / N_WM / (3 * D_MEAN) / (1 - G_WM)))

    # 4. per WM element
    e = np.load(os.path.join(OUT, "colin27_hcp1065_elem.npz"))
    wm = e["label"] == 4
    Dd = sym6_to_mat(e["D"][wm].astype(float), "mmc")
    w, V = np.linalg.eigh(Dd)
    dopt = w / w.mean(1, keepdims=True) * D_MEAN                 # OFA = FA, mean D fixed
    mus, clip = inv(dopt)
    Mwm = np.einsum("nij,nj,nkj->nik", V, mus, V)
    M6 = np.zeros((len(wm), 6), np.float32)
    M6[wm] = mat_to_sym6(Mwm, "mmc")
    fa = e["fa"][wm]
    nn = inv.last_nn
    ratio = mus.max(1) / mus.min(1)
    say("4. WM elements %d: mu's (= mu_s (1-g)) principal values, median of min / mid / max: %s 1/mm; "
        "max/min ratio median %.2f, 99th pct %.2f; beyond the LUT (capped): %d; outside the hull "
        "(nearest node): %d; NaN elements: %d"
        % (wm.sum(), np.round(np.nanmedian(np.sort(mus, 1), 0) * (1 - G_WM), 3),
           np.nanmedian(ratio), np.nanpercentile(ratio, 99), clip.sum(), nn.sum(),
           np.isnan(mus).any(1).sum()))
    if nn.any():
        dnn = np.array([np.sort(d_tensor_ade(N_WM, *mus[i], G_WM)) for i in np.flatnonzero(nn)])
        dt = np.sort(dopt[nn], 1)
        say("   nearest-node elements: FA %s; target D_opt %s, D(M) %s mm^2/ns; max |D(M) - D_opt| / max(D_opt) "
            "%.1e (mean D exact by construction)"
            % (np.round(fa[nn], 3), np.round(dt, 3).tolist(), np.round(dnn, 3).tolist(),
               (np.abs(dnn - dt).max(1) / dt.max(1)).max()))
    # spot check through the full lab-frame formula on a few elements
    idx = rng.choice(wm.sum(), 5, replace=False)
    rel = [np.abs(np.linalg.eigvalsh(d_tensor_full(N_WM, Mwm[i], G_WM)) / np.sort(dopt[i]) - 1).max() for i in idx]
    say("   5 random WM elements, D(M) by the full lab-frame series vs target D_opt: max rel. error %.1e" % max(rel))
    np.savez_compressed(os.path.join(OUT, "optics_wm.npz"), M=M6, wm=wm, mus_principal=mus.astype(np.float32),
                        fa=fa, d_mean=D_MEAN, g=G_WM, n=N_WM, lut=lut)
    with open(os.path.join(OUT, "optics_checks.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")
    print("%.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
