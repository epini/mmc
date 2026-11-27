#!/usr/bin/env python3
"""Time-domain analysis of the Colin27 head demo (CLAUDE.md step 2i): isotropic vs anisotropic WM.

    python demo_head_td.py        # after demo_head_gpu.py (needs <case>_raw in OUT/demo_head_gpu.npz)

From the exits saved photon by photon (15-45 mm from the source):
  time of flight t = sum_k n_k L_k / c (partial path L_k and refractive index n_k of each tissue);
  weight w = w0 exp(-sum_k mua_k L_k) (no Russian roulette in the run; tend = 5 ns).
Outputs (OUT):
  demo_head_tpsf.png     TPSF R(t) on the rings 20, 30, 40 mm (+-2.5 mm), iso vs aniso, and
                         aniso / iso - 1 vs t (+-2 SE)
  demo_head_gates.png    mean partial path <L_GM>, <L_WM> per 0.5 ns gate (0-5 ns) on the three rings,
                         and their change aniso / iso - 1
  demo_head_azimuth.png  30 mm ring (25-35 mm): change of <L_GM> vs azimuth for the early (0-1 ns) and
                         late (2-5 ns) photons, and per gate x azimuth (RdBu_r, symmetric limits)
  demo_head_td.txt       the numbers
Errors: jackknife over the batches of the run (iso and aniso share the seeds: the ratios are paired).
<L_k> in a gate is the sensitivity of the gated signal to mua in tissue k: d(-ln R_gate)/d(mua_k).
"""

import os

import numpy as np

from common import OUT

C0 = 299.792458                       # mm/ns
FRAC = os.path.join(OUT, "demo_head_gpu")
RINGS = (20.0, 30.0, 40.0)
HALF = 2.5
TB = np.arange(0.0, 5.0 + 1e-9, 0.05)  # TPSF bins, ns
RATIO_REBIN = 5                        # aniso / iso of the TPSF on 0.25 ns bins
GATES = np.arange(0.0, 5.0 + 1e-9, 0.5)
AZ_RING = (25.0, 35.0)
NAZ = 12
EARLY, LATE = (0.0, 1.0), (2.0, 5.0)
GM, WM = 2, 3                         # columns of L (labels 3, 4)


def jack(num, den):
    """num, den (nb, ...) per-batch sums -> ratio of the totals and its jackknife SE."""
    nb = num.shape[0]
    tn, td = num.sum(0), den.sum(0)
    j = (tn - num) / (td - den)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tn / td, np.sqrt((nb - 1) / nb * ((j - j.mean(0)) ** 2).sum(0)), j


def paired(ja, jb):
    """Jackknife replicas of two ratios -> change b / a - 1 and its SE."""
    nb = ja.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        jr = jb / ja - 1
        r = jr.mean(0)
    return r, np.sqrt((nb - 1) / nb * ((jr - r) ** 2).sum(0))


def hist(case, z, nb, edges, cond, key=None):
    """Per-batch histograms over t of sum w, sum w L_k (k = 0..3) and counts for the exits with cond."""
    raw, b = z[case + "_raw"].astype(float), z[case + "_rawb"]
    n, mua = z["prop"][1:, 3], z["prop"][1:, 0]
    L = raw[:, 2:6]
    t = L @ n / C0
    w = raw[:, 6] * np.exp(-L @ mua)
    m = cond(raw) & (t < edges[-1])
    it = np.digitize(t[m], edges) - 1
    k = key(raw[m]) if key is not None else np.zeros(m.sum(), int)
    nk = int(k.max()) + 1 if key is not None else 1
    out = np.zeros((nb, nk, len(edges) - 1, 6))
    q = np.column_stack([w[m], w[m, None] * L[m], np.ones(m.sum())])
    np.add.at(out, (b[m], k, it), q)
    return out, t


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(FRAC + ".npz")
    if "iso_raw" not in z:
        raise SystemExit("demo_head_gpu.npz has no photon-by-photon exits: rerun demo_head_gpu.py")
    nb, batch = int(z["nb"]), float(z["batch"])
    N = nb * batch
    log = []

    def say(s):
        print(s)
        log.append(s)

    say("time domain, Colin27 head, %d x %.3g photons per case; tissue n %s, mua %s 1/mm"
        % (nb, batch, z["prop"][1:, 3].tolist(), z["prop"][1:, 0].tolist()))
    ringkey = lambda r: np.digitize(r[:, 0], [RINGS[0] - HALF, RINGS[0] + HALF, RINGS[1] - HALF, RINGS[1] + HALF,
                                              RINGS[2] - HALF, RINGS[2] + HALF]) - 1
    inring = lambda r: np.isin(ringkey(r), (0, 2, 4))
    key3 = lambda r: ringkey(r) // 2
    H, G = {}, {}
    for case in ("iso", "aniso"):
        H[case], t = hist(case, z, nb, TB, inring, key3)
        G[case], _ = hist(case, z, nb, GATES, inring, key3)
        say("  %s: max time of flight of the saved exits %.3f ns (tend 5 ns)" % (case, t.max()))
    # consistency with the CW tallies of demo_head_gpu.py on the 25-30 mm bin
    for case in ("iso", "aniso"):
        hc, _ = hist(case, z, nb, np.array([0.0, 1e3]), lambda r: (r[:, 0] >= 25) & (r[:, 0] < 30))
        tal = z[case + "_rad"][:, 5].sum(0)
        say("  check %s, 25-30 mm: R from the photon list %.5e, CW tally %.5e (photons with t > 5 ns: none saved)"
            % (case, hc[..., 0].sum() / N, tal[0] / N))

    # ---- TPSF
    tc = 0.5 * (TB[1:] + TB[:-1])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    say("TPSF on the rings (+-%.1f mm): R per launched photon; mean time of flight <t>, change of R in gates" % HALF)
    for j, d in enumerate(RINGS):
        Ri = H["iso"][:, j, :, 0]
        Ra = H["aniso"][:, j, :, 0]
        dt = np.diff(TB)
        for R, sty, lab in ((Ri, "k-", "iso WM"), (Ra, "r--", "aniso WM")):
            ax[0, j].semilogy(tc, R.sum(0) / N / dt, sty, lw=1.2, label=lab)
        # ratio on coarser bins (RATIO_REBIN TPSF bins each)
        cs = lambda X: X.reshape(X.shape[0], -1, RATIO_REBIN).sum(2)
        Rci, Rca = cs(Ri), cs(Ra)
        tcc = cs(np.tile(tc, (1, 1)))[0] / RATIO_REBIN
        ones = np.ones_like(Rci)
        _, _, ji = jack(Rci, ones)
        _, _, ja = jack(Rca, ones)
        r, se = paired(ji, ja)
        ok = cs(H["iso"][:, j, :, 5]).sum(0) >= 100
        ax[1, j].fill_between(tcc[ok], 100 * (r - 2 * se)[ok], 100 * (r + 2 * se)[ok], color="0.75", lw=0, step="mid")
        ax[1, j].step(tcc[ok], 100 * r[ok], "k-", lw=1, where="mid")
        ax[1, j].axhline(0, color="0.5", lw=0.7)
        ax[0, j].set_title("ring %g mm" % d)
        ax[0, j].set_ylabel("R(t), 1/ns per photon")
        ax[1, j].set_ylabel("R aniso / iso − 1, %%  (%.2g ns bins, band ±2 SE)" % (RATIO_REBIN * (TB[1] - TB[0])))
        ax[1, j].set_xlabel("time of flight, ns")
        ax[0, j].legend()
        # mean time of flight
        tw = lambda R: (R * tc).sum(1)
        mi, smi, jmi = jack(tw(Ri)[:, None], Ri.sum(1)[:, None])
        ma, sma, jma = jack(tw(Ra)[:, None], Ra.sum(1)[:, None])
        say("  %g mm: <t> iso %.4f ± %.4f, aniso %.4f ± %.4f ns, difference %+.1f ± %.1f ps; counts %d"
            % (d, mi[0], smi[0], ma[0], sma[0], 1e3 * (ma - mi)[0],
               1e3 * np.sqrt((nb - 1) / nb * (((jma - jmi) - (jma - jmi).mean(0)) ** 2).sum()),
               H["iso"][:, j, :, 5].sum()))
    fig.suptitle("Colin27, source ~C3: time-of-flight distributions, isotropic vs anisotropic white matter")
    fig.savefig(os.path.join(OUT, "demo_head_tpsf.png"), dpi=130)
    plt.close(fig)

    # ---- per-gate mean partial path in GM, WM
    gc = 0.5 * (GATES[1:] + GATES[:-1])
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    say("per gate (%.1f ns): <L_GM>, <L_WM> iso / aniso (mm) and change (%%), +- SE; detected photons iso"
        % (GATES[1] - GATES[0]))
    for j, d in enumerate(RINGS):
        say("  ring %g mm:" % d)
        res = {}
        for k, name in ((GM, "GM"), (WM, "WM")):
            vi, si, ji = jack(G["iso"][:, j, :, 1 + k], G["iso"][:, j, :, 0])
            va, sa, ja = jack(G["aniso"][:, j, :, 1 + k], G["aniso"][:, j, :, 0])
            r, se = paired(ji, ja)
            res[name] = (vi, si, va, sa, r, se)
        cnt = G["iso"][:, j, :, 5].sum(0)
        ok = cnt >= 100
        for g in range(len(gc)):
            say("    %.2f-%.2f ns: GM %7.3f / %7.3f (%+6.2f ± %.2f%%)   WM %7.3f / %7.3f (%+6.2f ± %.2f%%)   n %d"
                % (GATES[g], GATES[g + 1], res["GM"][0][g], res["GM"][2][g], 100 * res["GM"][4][g], 100 * res["GM"][5][g],
                   res["WM"][0][g], res["WM"][2][g], 100 * res["WM"][4][g], 100 * res["WM"][5][g], cnt[g]))
        for name, col, mk in (("GM", "black", "o"), ("WM", "red", "s")):
            vi, si, va, sa, r, se = res[name]
            ax[0, j].errorbar(gc[ok], vi[ok], 2 * si[ok], fmt=mk + "-", color=col, capsize=2, label="<L_%s> iso" % name)
            ax[0, j].errorbar(gc[ok], va[ok], 2 * sa[ok], fmt=mk + "--", color=col, mfc="white", capsize=2,
                              label="<L_%s> aniso" % name)
            ax[1, j].errorbar(gc[ok], 100 * r[ok], 200 * se[ok], fmt=mk + "-", color=col, capsize=3, label="<L_%s>" % name)
        ax[1, j].axhline(0, color="0.5", lw=0.7)
        ax[0, j].set_yscale("log")
        ax[0, j].set_title("ring %g mm" % d)
        ax[0, j].set_ylabel("mean partial path, mm  (±2 SE)")
        ax[1, j].set_ylabel("aniso / iso − 1, %  (±2 SE)")
        ax[1, j].set_xlabel("gate centre, ns")
        ax[0, j].legend(fontsize=8)
        ax[1, j].legend(fontsize=8)
    fig.suptitle("Per-gate mean partial path (= d(−ln R_gate)/dμa) in GM and WM, gates of 0.5 ns")
    fig.savefig(os.path.join(OUT, "demo_head_gates.png"), dpi=130)
    plt.close(fig)

    # ---- 30 mm ring, azimuth
    ring = lambda r: (r[:, 0] >= AZ_RING[0]) & (r[:, 0] < AZ_RING[1])
    azk = lambda r: (r[:, 1] / (2 * np.pi / NAZ)).astype(int) % NAZ
    A = {c: hist(c, z, nb, GATES, ring, azk)[0] for c in ("iso", "aniso")}     # (nb, NAZ, ngate, 6)
    phc = (np.arange(NAZ) + 0.5) * 360 / NAZ
    say("ring %g-%g mm, <L_GM> change vs azimuth (0 = anterior, 90 = e2 = n x anterior, superior/medial):" % AZ_RING)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    for (g0, g1), col, mk in ((EARLY, "black", "o"), (LATE, "red", "s")):
        sel = (GATES[:-1] >= g0) & (GATES[1:] <= g1)
        num = {c: A[c][:, :, sel, 1 + GM].sum(2) for c in A}
        den = {c: A[c][:, :, sel, 0].sum(2) for c in A}
        vi, si, ji = jack(num["iso"], den["iso"])
        va, sa, ja = jack(num["aniso"], den["aniso"])
        r, se = paired(ji, ja)
        Ri, sRi, jRi = jack(den["iso"], np.ones_like(den["iso"]))
        Ra, sRa, jRa = jack(den["aniso"], np.ones_like(den["aniso"]))
        rR, seR = paired(jRi, jRa)
        chi2 = np.sum((r / se) ** 2)
        say("  gates %.1f-%.1f ns: chi2 of the <L_GM> change vs 0 over %d sectors = %.1f; R change chi2 = %.1f"
            % (g0, g1, NAZ, chi2, np.sum((rR / seR) ** 2)))
        for s in range(NAZ):
            say("    %5.1f deg: <L_GM> iso %7.3f ± %.3f, aniso %7.3f ± %.3f mm (%+6.2f ± %.2f%%); R %+6.2f ± %.2f%%; n %d"
                % (phc[s], vi[s], si[s], va[s], sa[s], 100 * r[s], 100 * se[s], 100 * rR[s], 100 * seR[s],
                   A["iso"][:, s, sel, 5].sum()))
        ax[0].errorbar(phc, 100 * r, 200 * se, fmt=mk + "-", color=col, capsize=3, label="%.0f–%.0f ns" % (g0, g1))
        ax[1].errorbar(phc, 100 * rR, 200 * seR, fmt=mk + "-", color=col, capsize=3, label="%.0f–%.0f ns" % (g0, g1))
    for a_, lab in ((ax[0], "<L_GM> aniso / iso − 1, %  (±2 SE)"), (ax[1], "gated R aniso / iso − 1, %  (±2 SE)")):
        a_.axhline(0, color="0.5", lw=0.7)
        a_.set_xticks(np.arange(0, 361, 90))
        a_.set_xlabel("azimuth around the source, deg (0 = anterior)")
        a_.set_ylabel(lab)
        a_.legend()
    # gate x azimuth map of the <L_GM> change
    vi, si, ji = jack(A["iso"][..., 1 + GM], A["iso"][..., 0])
    va, sa, ja = jack(A["aniso"][..., 1 + GM], A["aniso"][..., 0])
    r, se = paired(ji, ja)
    cnt = np.minimum(A["iso"][..., 5].sum(0), A["aniso"][..., 5].sum(0))
    okm = (cnt >= 100) & np.isfinite(r) & np.isfinite(se) & (se > 0)
    img = np.where(okm, 100 * r, np.nan).T                                  # (gate, azimuth)
    lim = max(np.nanpercentile(np.abs(img), 98), 1.0)
    ax[2].set_facecolor("0.85")
    im = ax[2].pcolormesh(np.linspace(0, 360, NAZ + 1), GATES, img, cmap="RdBu_r", vmin=-lim, vmax=lim)
    sig = np.where(okm, np.abs(r) / np.where(okm, se, 1.0), 0).T
    ax[2].contour(phc, 0.5 * (GATES[1:] + GATES[:-1]), sig, levels=[3.0], colors="black", linewidths=1.6)
    ax[2].set_xticks(np.arange(0, 361, 90))
    ax[2].set_xlabel("azimuth, deg")
    ax[2].set_ylabel("gate, ns")
    fig.colorbar(im, ax=ax[2], label="<L_GM> aniso / iso − 1, %")
    ax[2].set_title("per gate (grey: < 100 photons; black: 3 SE)")
    fig.suptitle("Ring %g–%g mm: GM sensitivity vs azimuth, early vs late photons" % AZ_RING)
    fig.savefig(os.path.join(OUT, "demo_head_azimuth.png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(OUT, "demo_head_td.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
