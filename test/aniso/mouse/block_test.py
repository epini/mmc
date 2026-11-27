#!/usr/bin/env python3
"""Mouse optogenetics block test (CLAUDE.md step 2b): optical fibre near an anisotropic CC slab.

    python block_test.py --wl 473 --tip above [--nphoton 1e8 --nbatch 10]   # one wavelength, one tip
    python block_test.py --wl 473 --all-tips                                 # the three tips
    python block_test.py --wl 473 --tip above --plot-only                    # figures from the saved npz
    python block_test.py --check                                             # quick checks, small runs

Parameters and placeholders: config.py. Geometry: GM block 6 x 6 x 6 mm (z = depth, surface z = 0, air
outside), CC slab 0.25 mm thick (normal z) at depth CC_TOP, scattering tensor uniaxial along x
(axons medio-lateral), mus_par / mus_perp = r with (mus_par + 2 mus_perp)/3 fixed; r = 1 is the
isotropic reference (also run through elemmus, same seeds batch by batch: the ratios are paired).

Fibre: 200 um core, NA 0.37, pointing +z. MMC has no source that is both a disk and a cone, so each run
is a point 'cone' source at the centre of the tip (half-angle asin(NA/n), uniform in solid angle), and
the 3D fluence of each batch is convolved in x, y with the uniform core disk (voxel area fractions).
The medium is laterally uniform (homogeneous block and slab, uniform tensor), so a source shifted by
(dx, dy) gives the shifted field and the convolution is the disk source exactly, up to the block edges
3 mm away (shifts <= 0.1 mm). Fibre body not meshed (v1): the tissue is continuous around the tip.

Output: dual-grid MMC (method 'grid') on GRID voxels; fluence per launched photon = flux x TEND (one
CW gate), irradiance for POWER_MW out of the fibre = POWER_MW x fluence (mW/mm^2). Exits are saved
(issaveexit, one detector covering the block) for the power leaving the block on each side of the slab;
weights w0 exp(-mua L) (Russian roulette at 1e-6 of the weight: negligible here).
Per case (wavelength, tip, r), with jackknife errors over the batches:
  planes xz (y = 0), yz (x = 0), xy (z = tip) of the irradiance and of aniso/iso - 1;
  profiles along x and y in the slab mid-plane and BELOW_DZ under the slab;
  volume and x/y/z extents above each threshold; fraction absorbed in the slab, above, below;
  net power through the slab faces: P_down = power into z > CC bottom (absorbed + escaped there,
  minus 1 if the tip is there), P_up = the same for z < CC top; energy balance absorbed + escaped = 1
  (exact on the raw grid; the disk convolution loses ~0.1% through the lateral faces of the array).
Files (OUT = MOUSE_DATA/out, default <repo>/../mouse_data/out): block_<wl>_<tip>.npz (summaries only,
no 3D fields), block_<wl>_<tip>_{fluence,ratio,profiles,metrics}.png, block_<wl>_<tip>.txt.
"""

import argparse
import os
import sys
import time

import numpy as np

import config as C

LABEL_GM, LABEL_CC = 1, 2
C0 = 299.792458                                      # mm/ns


# ------------------------------------------------------------------------------------------ mesh
def graded(a, b, h0, ratio, hmax):
    """Node coordinates from a to b, first step h0 at a, growing by ratio up to hmax; exact end b."""
    L, steps = abs(b - a), []
    h = h0
    while sum(steps) < L - 1e-9:
        steps.append(min(h, hmax))
        h *= ratio
    steps = np.array(steps) * L / sum(steps)
    return a + np.sign(b - a) * np.concatenate([[0.0], np.cumsum(steps)])


def build_mesh():
    """Structured tetrahedral mesh: node (nn, 3), elem (ne, 4) 1-based positive volume, label (ne,)."""
    h = C.MESH_FINE
    xs = np.linspace(-C.BLOCK / 2, C.BLOCK / 2, int(round(C.BLOCK / h)) + 1)
    z0, z1 = C.FINE_Z
    zf = np.linspace(z0, z1, int(round((z1 - z0) / h)) + 1)
    zs = np.unique(np.round(np.concatenate([graded(z0, 0.0, h * C.MESH_GROW, C.MESH_GROW, C.MESH_COARSE),
                                            zf, graded(z1, C.BLOCK, h * C.MESH_GROW, C.MESH_GROW,
                                                       C.MESH_COARSE)]), 9))
    for zc in (C.CC_TOP, C.CC_TOP + C.CC_THICK):
        assert np.min(np.abs(zs - zc)) < 1e-9, "the slab faces must be mesh planes"
    nx, ny, nz = len(xs), len(xs), len(zs)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    node = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    # Kuhn split: 6 tetrahedra per cube along the 000-111 diagonal (conforming between cubes)
    corner = {}
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                corner[(bx, by, bz)] = idx[bx:nx - 1 + bx, by:ny - 1 + by, bz:nz - 1 + bz].ravel()
    tets = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        v, path = [0, 0, 0], [(0, 0, 0)]
        for ax in perm:
            v[ax] = 1
            path.append(tuple(v))
        tets.append(np.column_stack([corner[p] for p in path]))
    elem = np.concatenate(tets)
    p = node[elem]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    neg = vol < 0
    elem[neg, 2], elem[neg, 3] = elem[neg, 3], elem[neg, 2].copy()
    zc = p[:, :, 2].mean(1)
    label = np.where((zc > C.CC_TOP) & (zc < C.CC_TOP + C.CC_THICK), LABEL_CC, LABEL_GM)
    return node, elem + 1, label


# ------------------------------------------------------------------------------------------ source
def disk_kernel(sub=20):
    """Area fraction of the core disk in each GRID x GRID cell, offsets centred on 0; sums to 1."""
    m = int(np.ceil(C.CORE_RADIUS / C.GRID - 0.5)) + 1
    off = np.arange(-m, m + 1) * C.GRID
    s = (np.arange(sub) + 0.5) / sub * C.GRID - C.GRID / 2
    u = off[:, None] + s[None, :]                       # sub-sample coordinates per cell
    inside = (u[:, None, :, None] ** 2 + u[None, :, None, :] ** 2) <= C.CORE_RADIUS ** 2
    K = inside.mean((2, 3))
    return K / K.sum()


def half_angle():
    return float(np.arcsin(C.NA / C.N_TISSUE))


# ------------------------------------------------------------------------------------------ runs
def quiet(fn, verbose):
    if verbose:
        return fn()
    sys.stdout.flush()
    fd, keep = sys.stdout.fileno(), os.dup(sys.stdout.fileno())
    os.dup2(os.open(os.devnull, os.O_WRONLY), fd)
    try:
        return fn()
    finally:
        os.dup2(keep, fd)
        os.close(keep)


def base_cfg(mesh, wl, tip_z, nphoton, seed, compute, gpuid):
    node, elem, label = mesh
    o = C.WAVELENGTHS[wl]
    mus_cc = C.MUS_CC_OVER_GM * o["mus_gm"]
    eps = 1e-4                                          # keep the point source off mesh faces and edges
    return dict(nphoton=int(nphoton), seed=int(seed), node=node, elem=elem, elemprop=label,
                prop=[[0, 0, 1, 1], [o["mua"], o["mus_gm"], C.G, C.N_TISSUE], [o["mua"], mus_cc, C.G, C.N_TISSUE]],
                srctype="cone", srcparam1=[half_angle(), 0, 0, 0], srcparam2=[0, 0, 0, 0],
                srcpos=[eps, eps, tip_z + eps], srcdir=list(C.FIBRE_DIR),
                tstart=0, tend=C.TEND, tstep=C.TEND, method="grid", steps=[C.GRID] * 3,
                compute=compute, gpuid=gpuid, issaveexit=1,
                detpos=[[0.0, 0.0, C.BLOCK / 2, 2 * C.BLOCK]], maxdetphoton=int(nphoton))


def elemmus_for(label, wl, r):
    E = np.zeros((len(label), 6), np.float32)
    E[:, :3] = C.WAVELENGTHS[wl]["mus_gm"]
    par, perp = C.cc_tensor(wl, r)
    E[label == LABEL_CC, :3] = [par, perp, perp]
    return E


def run_once(cfg, verbose):
    import pmmc
    r = quiet(lambda: pmmc.mmclab(cfg), verbose)
    # fluence per launched photon, 1/mm^2: on the dual grid pmmc's normalized flux leaves out the voxel
    # volume (invisible with the default 1 mm voxels), so divide by it; checked by the energy balance
    # (absorbed from the grid + escaped = 1, and = the absorbed fraction printed by MMC)
    phi = (np.asarray(r["flux"])[..., 0] * (C.TEND / C.GRID ** 3)).astype(np.float32)
    d = r["detp"]
    p = np.asarray(d["p"], np.float32)
    L = np.asarray(d["ppath"], np.float64).reshape(len(p), -1).sum(1)
    mua = cfg["prop"][1][0]                                                   # same mua in GM and CC
    w = np.asarray(d["w0"], np.float64) * np.exp(-mua * L)
    return phi, p, w


class Grid:
    """Voxel centres of the dual grid (origin at the mesh bounding-box minimum)."""

    def __init__(self, shape):
        self.shape = shape
        self.x = -C.BLOCK / 2 + (np.arange(shape[0]) + 0.5) * C.GRID
        self.y = -C.BLOCK / 2 + (np.arange(shape[1]) + 0.5) * C.GRID
        self.z = (np.arange(shape[2]) + 0.5) * C.GRID
        zt, zb = C.CC_TOP, C.CC_TOP + C.CC_THICK
        lo, hi = self.z - C.GRID / 2, self.z + C.GRID / 2
        self.w_slab = np.clip(np.minimum(hi, zb) - np.maximum(lo, zt), 0, None) / C.GRID     # z weights
        self.w_above = np.clip(np.minimum(hi, zt) - lo, 0, C.GRID) / C.GRID
        self.w_below = np.clip(hi - np.maximum(lo, zb), 0, C.GRID) / C.GRID

    def interp(self, axis_coords, c):
        """Indices and weights for linear interpolation at coordinate c between voxel centres."""
        f = np.clip((c - axis_coords[0]) / C.GRID, 0, len(axis_coords) - 1.000001)
        i = int(np.floor(f))
        return i, f - i

    def plane(self, F, axis, c):
        ax = (self.x, self.y, self.z)[axis]
        i, t = self.interp(ax, c)
        a, b = np.take(F, i, axis), np.take(F, i + 1, axis)
        return (1 - t) * a + t * b

    def line(self, F, along, c1, c2):
        """Line along x (along = 0) or y (1) at the other lateral coordinate c1 and depth c2."""
        P = self.plane(F, 2, c2)                          # (nx, ny)
        other = self.y if along == 0 else self.x
        i, t = self.interp(other, c1)
        return (1 - t) * P[:, i] + t * P[:, i + 1] if along == 0 else (1 - t) * P[i, :] + t * P[i + 1, :]


def field_metrics(F, g, mua):
    """Scalar metrics of one (normalized, per photon) fluence field: threshold volumes/extents and
    absorbed fractions (above / in / below the slab)."""
    out = {}
    v = C.GRID ** 3
    for thr in C.THRESHOLDS_MW_MM2:
        m = F >= thr / C.POWER_MW
        out["V_%g" % thr] = m.sum() * v
        for k, name in enumerate("xyz"):
            ix = np.flatnonzero(m.any(tuple(j for j in range(3) if j != k)))
            out["ext%s_%g" % (name, thr)] = (ix[-1] - ix[0] + 1) * C.GRID if ix.size else 0.0
    S = F.sum((0, 1), dtype=np.float64) * v * mua                          # absorbed per z voxel layer
    out["abs_slab"] = float(S @ g.w_slab)
    out["abs_above"] = float(S @ g.w_above)
    out["abs_below"] = float(S @ g.w_below)
    return out


def simulate(wl, tipname, nphoton, nbatch, compute, gpuid, verbose):
    from scipy.signal import fftconvolve
    os.makedirs(C.OUT, exist_ok=True)
    t00 = time.time()
    mesh = build_mesh()
    node, elem, label = mesh
    print("mesh: %d nodes, %d tetrahedra (%d in the CC); %.0f s" % (len(node), len(elem), (label == LABEL_CC).sum(),
                                                                     time.time() - t00), flush=True)
    tip_z = C.TIPS[tipname]
    K = disk_kernel()[:, :, None].astype(np.float32)
    mua = C.WAVELENGTHS[wl]["mua"]
    batch = nphoton / nbatch
    zt, zb = C.CC_TOP, C.CC_TOP + C.CC_THICK
    in_below, in_above = tip_z > zb, tip_z < zt
    keep = {}
    iso_planes = None
    for r in C.R_LIST:
        E = elemmus_for(label, wl, r)
        t0 = time.time()
        fields, planes, lines, esc = [], [], [], []
        for b in range(nbatch):
            cfg = base_cfg(mesh, wl, tip_z, batch, C.SEED + 7919 * b, compute, gpuid)
            cfg["elemmus"] = E
            phi, p, w = run_once(cfg, verbose)
            F = fftconvolve(phi, K, mode="same", axes=(0, 1)).astype(np.float32)   # disk source
            np.maximum(F, 0, out=F)                                                  # FFT round-off
            if b == 0:
                g = Grid(F.shape)
            fields.append(F)
            planes.append((g.plane(F, 1, 0.0).T, g.plane(F, 0, 0.0).T, g.plane(F, 2, tip_z).T))  # xz, yz, xy
            lines.append([g.line(F, 0, 0.0, zt + C.CC_THICK / 2), g.line(F, 1, 0.0, zt + C.CC_THICK / 2),
                          g.line(F, 0, 0.0, zb + C.BELOW_DZ), g.line(F, 1, 0.0, zb + C.BELOW_DZ)])
            esc.append([w.sum() / batch, w[p[:, 2] > zb + 1e-6].sum() / batch, w[p[:, 2] < zt - 1e-6].sum() / batch])
            print("  %d nm, tip %s, r = %g: batch %d/%d, %d exits, %.0f s"
                  % (wl, tipname, r, b + 1, nbatch, len(w), time.time() - t0), flush=True)
        # jackknife: metrics of the leave-one-out mean fields (threshold metrics are not linear)
        S = np.sum(fields, 0, dtype=np.float64)
        full = field_metrics((S / nbatch).astype(np.float32), g, mua)
        loo = {k: np.zeros(nbatch) for k in full}
        for b in range(nbatch):
            mb = field_metrics(((S - fields[b]) / (nbatch - 1)).astype(np.float32), g, mua)
            for k in mb:
                loo[k][b] = mb[k]
        esc = np.array(esc)                                  # (nb, 3): total, below, above
        escl = (esc.sum(0) - esc) / (nbatch - 1)
        for name, col in (("esc_total", 0), ("esc_below", 1), ("esc_above", 2)):
            full[name] = esc[:, col].mean()
            loo[name] = escl[:, col]
        for d in (full, loo):
            d["P_down"] = d["abs_below"] + d["esc_below"] - float(in_below)
            d["P_up"] = d["abs_above"] + d["esc_above"] - float(in_above)
            d["esc_slab"] = d["esc_total"] - d["esc_below"] - d["esc_above"]       # side faces at CC depth
            d["balance_total"] = d["abs_slab"] + d["abs_above"] + d["abs_below"] + d["esc_total"] - 1.0
        bp = [np.array([pl[k] for pl in planes]) for k in range(3)]                  # (nb, ...) each
        rec = dict(mean_planes=[q.mean(0) for q in bp], lines=np.array(lines),
                   full=full, loo=loo, seconds=time.time() - t0)
        if r == 1.0:
            iso_planes = bp
        else:
            ratio, se = [], []
            for qa, qi in zip(bp, iso_planes):
                Sa, Si = qa.sum(0, dtype=np.float64), qi.sum(0, dtype=np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    jr = (Sa - qa) / (Si - qi) - 1
                    ratio.append(Sa / Si - 1)
                se.append(np.sqrt((nbatch - 1) / nbatch * ((jr - jr.mean(0)) ** 2).sum(0)))
            rec["ratio"], rec["ratio_se"] = ratio, se
        keep[r] = rec
        del fields, S
        print("  r = %g done: %.0f s; energy balance %+.1e" % (r, rec["seconds"], full["balance_total"]), flush=True)
    save(wl, tipname, keep, g, nphoton, nbatch, compute, time.time() - t00)


# ------------------------------------------------------------------------------------------ save / load
def fname(wl, tip, ext):
    return os.path.join(C.OUT, "block_%d_%s%s" % (wl, tip, ext))


def save(wl, tip, keep, g, nphoton, nbatch, compute, seconds):
    z = dict(x=g.x, y=g.y, z=g.z, r_list=np.array(C.R_LIST), nphoton=nphoton, nbatch=nbatch, compute=compute,
             seconds=seconds, tip_z=C.TIPS[tip])
    for r, rec in keep.items():
        t = "r%g_" % r
        for name, q in zip(("xz", "yz", "xy"), rec["mean_planes"]):
            z[t + name] = q.astype(np.float32)
            if "ratio" in rec:
                i = ("xz", "yz", "xy").index(name)
                z[t + "ratio_" + name] = rec["ratio"][i].astype(np.float32)
                z[t + "ratiose_" + name] = rec["ratio_se"][i].astype(np.float32)
        z[t + "lines"] = rec["lines"].astype(np.float32)
        z[t + "seconds"] = rec["seconds"]
        for k, v in rec["full"].items():
            z[t + "full_" + k] = v
            z[t + "loo_" + k] = rec["loo"][k]
    np.savez_compressed(fname(wl, tip, ".npz"), **z)


def jk_se(loo):
    n = len(loo)
    return float(np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum()))


# ------------------------------------------------------------------------------------------ figures
def plot(wl, tip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(fname(wl, tip, ".npz"))
    rl = list(z["r_list"])
    x, y, zz = z["x"], z["y"], z["z"]
    tip_z = float(z["tip_z"])
    zt, zb = C.CC_TOP, C.CC_TOP + C.CC_THICK
    P = C.POWER_MW
    log = []

    def say(s):
        print(s)
        log.append(s)

    o = C.WAVELENGTHS[wl]
    say("Mouse block test, %d nm, tip %s (z = %.3f mm), %s, %.3g photons per case in %d batches; total %.0f s"
        % (wl, tip, tip_z, str(z["compute"]), float(z["nphoton"]), int(z["nbatch"]), float(z["seconds"])))
    say("PLACEHOLDER optics: GM mus %.2f /mm, g %.2f, mua %.3f /mm, n %.2f; CC mean mus %.1f /mm; fibre core %g um, "
        "NA %.2f (half-angle %.1f deg in tissue); %g mW" % (o["mus_gm"], C.G, o["mua"], C.N_TISSUE,
                                                          C.MUS_CC_OVER_GM * o["mus_gm"], 2e3 * C.CORE_RADIUS, C.NA,
                                                          np.degrees(half_angle()), P))
    for r in rl:
        par, perp = C.cc_tensor(wl, r)
        say("  r = %.1f: CC mus_par %.1f, mus_perp %.1f /mm" % (r, par, perp))

    lat = 2.5
    zmax = min(4.0, zz[-1])
    win = dict(xz=(x, zz, lat, zmax), yz=(y, zz, lat, zmax), xy=(x, y, lat, lat))
    labels = dict(xz=("x (along axons), mm", "depth z, mm"), yz=("y (across axons), mm", "depth z, mm"),
                  xy=("x (along axons), mm", "y, mm"))
    titles = dict(xz="xz (y = 0)", yz="yz (x = 0)", xy="xy (z = tip, %.3f mm)" % tip_z)

    def crop(name, A):
        u, v, lu, lv = win[name]
        iu = np.abs(u) <= lu
        iv = (np.abs(v) <= lv) if name == "xy" else (v <= lv)
        return u[iu], v[iv], A[np.ix_(iv, iu)]

    def deco(a, name, colour):
        if name != "xy":
            for zc in (zt, zb):
                a.axhline(zc, color=colour, lw=0.8)
            a.invert_yaxis()
            a.plot(0, tip_z, "v", ms=7, color=colour, mec="white" if colour != "white" else "black")
        else:
            a.add_patch(matplotlib.patches.Circle((0, 0), C.CORE_RADIUS, fill=False, color=colour, lw=0.8))
        a.set_aspect("equal")

    # ---- fluence maps
    ref = np.log10(P * np.maximum(z["r1_xz"], 1e-30))
    vmax = np.ceil(ref.max())
    vmin = vmax - 6
    fig, ax = plt.subplots(len(rl), 3, figsize=(14, 3.7 * len(rl)), constrained_layout=True)
    for i, r in enumerate(rl):
        for j, name in enumerate(("xz", "yz", "xy")):
            u, v, A = crop(name, P * z["r%g_%s" % (r, name)])
            a = ax[i, j]
            im = a.pcolormesh(u, v, np.log10(np.maximum(A, 1e-30)), cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
            cs = a.contour(u, v, A, levels=sorted(C.THRESHOLDS_MW_MM2), colors="white",
                           linestyles=["dashed", "solid", "dotted"][:len(C.THRESHOLDS_MW_MM2)], linewidths=0.9)
            deco(a, name, "red")
            a.set_title("r = %.1f, %s" % (r, titles[name]))
            a.set_xlabel(labels[name][0])
            a.set_ylabel(labels[name][1])
        fig.colorbar(im, ax=ax[i, :], shrink=0.85, label="log10 irradiance, mW/mm² (%g mW)" % P)
    fig.suptitle("%d nm, tip %s: irradiance for %g mW out of the fibre (PLACEHOLDER optics)\nwhite contours: "
                 "0.1 (dashed), 1 (solid), 10 (dotted) mW/mm²; red lines: CC slab; triangle / circle: fibre tip"
                 % (wl, tip, P))
    fig.savefig(fname(wl, tip, "_fluence.png"), dpi=110)
    plt.close(fig)

    # ---- ratio maps
    ra = [r for r in rl if r != 1.0]
    fig, ax = plt.subplots(len(ra), 3, figsize=(14, 3.7 * len(ra)), constrained_layout=True)
    vals = []
    for r in ra:
        for name in ("xz", "yz", "xy"):
            _, _, R = crop(name, z["r%g_ratio_%s" % (r, name)])
            _, _, S = crop(name, z["r%g_ratiose_%s" % (r, name)])
            _, _, I = crop(name, P * z["r1_%s" % name])
            ok = (I >= 1e-3) & (S < 0.05)
            vals.append(np.abs(R[ok]))
    lim = max(5.0, float(np.ceil(100 * np.percentile(np.concatenate(vals), 99))))
    for i, r in enumerate(ra):
        for j, name in enumerate(("xz", "yz", "xy")):
            u, v, R = crop(name, z["r%g_ratio_%s" % (r, name)])
            _, _, S = crop(name, z["r%g_ratiose_%s" % (r, name)])
            _, _, I = crop(name, P * z["r1_%s" % name])
            ok = (I >= 1e-3) & (S < 0.05)
            a = ax[i, j]
            a.set_facecolor("0.85")
            im = a.pcolormesh(u, v, np.where(ok, 100 * R, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
            sig = np.where(ok, np.abs(R) / np.maximum(S, 1e-12), 0)
            if (sig > 3).any() and (sig < 3).any():
                a.contour(u, v, sig, levels=[3.0], colors="black", linewidths=0.5, linestyles="dashed")
            deco(a, name, "black")
            a.set_title("r = %.1f vs r = 1, %s" % (r, titles[name]))
            a.set_xlabel(labels[name][0])
            a.set_ylabel(labels[name][1])
        fig.colorbar(im, ax=ax[i, :], shrink=0.85, label="aniso / iso − 1, %")
    fig.suptitle("%d nm, tip %s: change of the fluence, anisotropic CC (r) vs isotropic CC (same mean mus)\n"
                 "grey: irradiance < 1e-3 mW/mm² or SE > 5%%; black dashed: |Δ| = 3 SE; black lines: CC slab"
                 % (wl, tip))
    fig.savefig(fname(wl, tip, "_ratio.png"), dpi=110)
    plt.close(fig)

    # ---- profiles
    cols = ["black", "tab:blue", "tab:green", "tab:orange", "red", "tab:purple", "tab:brown"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    names = ["along x, CC mid-plane", "along y, CC mid-plane", "along x, %g mm below the CC" % C.BELOW_DZ,
             "along y, %g mm below the CC" % C.BELOW_DZ]
    for i, r in enumerate(rl):
        Lb = P * z["r%g_lines" % r]                              # (nb, 4, n)
        for k in range(4):
            a = ax.flat[k]
            u = x if k in (0, 2) else y
            m = Lb[:, k].mean(0)
            se = Lb[:, k].std(0, ddof=1) / np.sqrt(Lb.shape[0])
            sel = np.abs(u) <= 2.0
            a.semilogy(u[sel], m[sel], "-", color=cols[i % len(cols)], label="r = %.1f" % r)
            a.fill_between(u[sel], np.maximum(m - 2 * se, 1e-30)[sel], (m + 2 * se)[sel], color=cols[i % len(cols)],
                           alpha=0.2, lw=0)
    for k in range(4):
        a = ax.flat[k]
        a.axhline(C.MAIN_THRESHOLD, color="gray", lw=0.7, ls="--")
        a.set_xlabel(("x" if k in (0, 2) else "y") + ", mm")
        a.set_ylabel("irradiance, mW/mm² (%g mW)" % P)
        a.set_title(names[k])
        a.legend(fontsize=8)
    fig.suptitle("%d nm, tip %s: lateral profiles (band ±2 SE); grey dashed: %g mW/mm²" % (wl, tip, C.MAIN_THRESHOLD))
    fig.savefig(fname(wl, tip, "_profiles.png"), dpi=110)
    plt.close(fig)

    # ---- metrics vs r
    def get(r, k):
        if k == "esc_slab" and "r%g_full_esc_slab" % r not in z:      # npz written before esc_slab existed
            f = [get(r, q) for q in ("esc_total", "esc_below", "esc_above")]
            return f[0][0] - f[1][0] - f[2][0], 0.0
        return float(z["r%g_full_%s" % (r, k)]), jk_se(z["r%g_loo_%s" % (r, k)])

    panels = ([("V_%g" % t, "volume > %g mW/mm², mm³" % t) for t in C.THRESHOLDS_MW_MM2]
              + [("ext%s_%g" % (a_, C.MAIN_THRESHOLD), "%s extent > %g mW/mm², mm" % (a_, C.MAIN_THRESHOLD))
                 for a_ in "xyz"]
              + [("abs_slab", "fraction absorbed in the CC"), ("P_down", "net power into z > CC bottom"),
                 ("P_up", "net power into z < CC top")])
    fig, ax = plt.subplots(3, 3, figsize=(13, 10.5), constrained_layout=True)
    rr = np.array(rl)
    for a, (k, lab) in zip(ax.flat, panels):
        v = np.array([get(r, k) for r in rl])
        a.errorbar(rr, v[:, 0], 2 * v[:, 1], fmt="o-", color="black", capsize=3)
        a.set_xlabel("r = mus_par / mus_perp in the CC")
        a.set_ylabel(lab)
        a.set_xlim(0, 1.05)
    fig.suptitle("%d nm, tip %s: metrics vs anisotropy of the CC (mean mus fixed; ±2 SE, jackknife)" % (wl, tip))
    fig.savefig(fname(wl, tip, "_metrics.png"), dpi=110)
    plt.close(fig)

    say("metrics (value ± SE; change vs r = 1, paired jackknife):")
    for k, lab in panels:
        line = "  %-12s" % k
        i1 = z["r1_loo_%s" % k]
        f1 = float(z["r1_full_%s" % k])
        for r in rl:
            f, s = get(r, k)
            if r == 1.0 or f1 == 0:
                line += " | r=%.1f %.4g ± %.2g" % (r, f, s)
            else:
                lo = z["r%g_loo_%s" % (r, k)]
                with np.errstate(invalid="ignore", divide="ignore"):
                    d = jk_se(lo / i1 - 1)
                line += " | r=%.1f %.4g (%+.2f%% ± %.2f)" % (r, f, 100 * (f / f1 - 1), 100 * d)
        say(line)
    for r in rl:
        say("  r = %.1f: absorbed slab/above/below %.4f/%.4f/%.4f, escaped %.4f (below the CC %.4f, above %.4f, "
            "sides at CC depth %.4f); energy balance absorbed + escaped - 1 = %+.1e"
            % (r, get(r, "abs_slab")[0], get(r, "abs_above")[0], get(r, "abs_below")[0], get(r, "esc_total")[0],
               get(r, "esc_below")[0], get(r, "esc_above")[0], get(r, "esc_slab")[0], get(r, "balance_total")[0]))
    say("  (energy balance: 1.00000 on the raw grid; the disk convolution loses ~0.1% at the lateral faces)")
    with open(fname(wl, tip, ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(log) + "\n")


# ------------------------------------------------------------------------------------------ checks
def check(compute, gpuid):
    """Small runs: disk kernel, energy balance, isotropic tensor = vanilla on the grid output."""
    K = disk_kernel()
    a = (K > 0).sum() * C.GRID ** 2
    print("disk kernel %s, cells with any core: %d (area of the covered cells %.4f, core %.4f mm^2)"
          % (K.shape, (K > 0).sum(), a, np.pi * C.CORE_RADIUS ** 2))
    mesh = build_mesh()
    node, elem, label = mesh
    print("mesh: %d nodes, %d tetrahedra, CC %d; volume %.3f mm^3 (block %.3f)"
          % (len(node), len(elem), (label == 2).sum(),
             np.abs(np.einsum("ij,ij->i", np.cross(node[elem[:, 1] - 1] - node[elem[:, 0] - 1],
                                                   node[elem[:, 2] - 1] - node[elem[:, 0] - 1]),
                              node[elem[:, 3] - 1] - node[elem[:, 0] - 1])).sum() / 6, C.BLOCK ** 3))
    wl, tz = 473, C.TIPS["inside"]
    cfg = base_cfg(mesh, wl, tz, 1e6, 12345, compute, gpuid)
    cfg_iso = dict(cfg, elemmus=elemmus_for(label, wl, 1.0))
    t0 = time.time()
    phi_t, p_t, w_t = run_once(cfg_iso, False)
    t1 = time.time()
    phi_v, p_v, w_v = run_once(dict(cfg, seed=54321), False)
    g = Grid(phi_t.shape)
    mua = C.WAVELENGTHS[wl]["mua"]
    for name, phi, w in (("tensor mus*I", phi_t, w_t), ("vanilla", phi_v, w_v)):
        m = field_metrics(phi, g, mua)
        tot = m["abs_slab"] + m["abs_above"] + m["abs_below"]
        print("%-13s grid %s: absorbed %.5f + escaped %.5f = %.5f; abs in CC %.5f; V(1 mW/mm2, point source) %.4f mm^3"
              % (name, phi.shape, tot, w.sum() / 1e6, tot + w.sum() / 1e6, m["abs_slab"], m["V_1"]))
    print("first run %.1f s (1e6 photons)" % (t1 - t0))
    ix = np.argmin(np.abs(g.x))
    prof = g.line(phi_t, 0, 0.0, tz)
    sel = np.abs(g.x) < 0.5
    print("grid origin: fluence maximum at x = %.3f, z = %.3f mm (tip 0, %.3f)"
          % (g.x[np.argmax(prof)], g.z[np.argmax(phi_t[ix].max(0))], tz))
    print("symmetry x -> -x of the profile along x at the tip depth: sum over |x| < 0.5 mm %.4g vs %.4g"
          % (prof[sel & (g.x > 0)].sum(), prof[sel & (g.x < 0)].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wl", type=int, default=473, choices=sorted(C.WAVELENGTHS))
    ap.add_argument("--tip", default="above", choices=list(C.TIPS))
    ap.add_argument("--all-tips", action="store_true")
    ap.add_argument("--nphoton", type=float, default=C.NPHOTON)
    ap.add_argument("--nbatch", type=int, default=C.NBATCH)
    ap.add_argument("--compute", default="cuda", choices=["opencl", "cuda"])
    ap.add_argument("--gpuid", type=int, default=1)
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.check:
        check(a.compute, a.gpuid)
        return
    for tip in (list(C.TIPS) if a.all_tips else [a.tip]):
        if not a.plot_only:
            simulate(a.wl, tip, a.nphoton, a.nbatch, a.compute, a.gpuid, a.verbose)
        plot(a.wl, tip)


if __name__ == "__main__":
    main()
