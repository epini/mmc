#!/usr/bin/env python3
"""Step e: check figures of the HCP1065 tensors on the Colin27 mesh.

    python checks.py               # after warp_tensors.py, conda env headenv

check_fa.png         FA per tissue on the mesh elements (volume weighted), fraction of WM with FA > 0.2;
                     FA per JHU tract: template in MNI vs Colin27 elements (registration check)
check_dec.png        direction-coloured FA (red L-R, green A-P, blue S-I), axial / coronal / sagittal:
                     voxel field on the Colin27 grid and the piecewise-constant value per element (MMC)
check_cc_cst.png     corpus callosum (JHU genu/body/splenium) |v1.x| and corticospinal path (JHU CST,
                     cerebral peduncle, PLIC) |v1.z| on the elements, midsagittal and coronal maps
Numbers printed and written to OUT/checks.txt.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

from common import COLIN_MASK, HCP_FA, JHU, JHU_LABELS, MESH_ORIGIN, OUT, TISSUE, fa_v1, load_mesh, spd_fun, sym6_to_mat

HERE = os.path.dirname(os.path.abspath(__file__))
FA_SCALE = 0.5     # DEC brightness = FA / FA_SCALE (HCP1065 template FA <= 0.54)
SLICES = {"axial z = +20 mm": (2, 20.0), "coronal y = -20 mm": (1, -20.0), "sagittal x = 0 mm": (0, 0.0)}


def dec(fa, v1):
    return np.clip(np.abs(v1) * np.clip(fa / FA_SCALE, 0, 1)[..., None], 0, 1)


def voxel_slice(Lc, brain, axis, w):
    """FA and v1 on a Colin27 grid slice at world coordinate w (mm) along axis."""
    idx = int(round(w + MESH_ORIGIN[axis]))
    sl = [slice(None)] * 3
    sl[axis] = idx
    L, m = Lc[tuple(sl)], brain[tuple(sl)]
    fa, v1 = np.zeros(m.shape), np.zeros(m.shape + (3,))
    f, _, v = fa_v1(spd_fun(sym6_to_mat(L[m].astype(float), "fsl"), np.exp))
    fa[m], v1[m] = f, v
    return fa, v1


def locate(pts, node, elem, tree, k=40):
    """Index of the tetrahedron containing each point (-1 if none among the k nearest centroids)."""
    _, cand = tree.query(pts, k=k)
    out = np.full(len(pts), -1)
    T = node[elem - 1]                                  # (ne, 4, 3)
    for j in range(k):
        c = cand[:, j]
        todo = out < 0
        if not todo.any():
            break
        A = np.swapaxes(T[c[todo], 1:] - T[c[todo], :1], 1, 2)   # (n, 3, 3) columns = edges
        lam = np.linalg.solve(A, (pts[todo] - T[c[todo], 0])[..., None])[..., 0]
        inside = (lam >= -1e-9).all(1) & (lam.sum(1) <= 1 + 1e-9)
        out[np.flatnonzero(todo)[inside]] = c[todo][inside]
    return out


def element_slice(e, node, elem, tree, axis, w, shape):
    """Per-pixel element index on the slice plane (1 mm pixels of the Colin27 grid)."""
    other = [a for a in range(3) if a != axis]
    g = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    pts = np.zeros((g[0].size, 3))
    pts[:, other[0]], pts[:, other[1]] = g[0].ravel(), g[1].ravel()
    pts[:, axis] = w + MESH_ORIGIN[axis]
    return locate(pts, node, elem, tree).reshape(shape)


def show(ax, img, axis, title):
    other = [a for a in range(3) if a != axis]
    n0, n1 = img.shape[:2]
    ext = [-MESH_ORIGIN[other[0]], n0 - MESH_ORIGIN[other[0]], -MESH_ORIGIN[other[1]], n1 - MESH_ORIGIN[other[1]]]
    ax.imshow(np.swapaxes(img, 0, 1), origin="lower", extent=ext, interpolation="nearest")
    ax.set_xlabel("xyz"[other[0]] + " (mm)"); ax.set_ylabel("xyz"[other[1]] + " (mm)")
    ax.set_title(title, fontsize=9)


def main():
    e = np.load(os.path.join(OUT, "colin27_hcp1065_elem.npz"))
    node, elem, label = load_mesh()
    fa, v1, jhu = e["fa"], e["v1"], e["jhu"]
    T = node[elem - 1]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), T[:, 3] - T[:, 0])) / 6
    log = []

    def say(s):
        print(s)
        log.append(s)

    # ---- FA per tissue and per JHU tract
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.linspace(0, 0.6, 61)
    cols = {2: "#1f77b4", 3: "#2ca02c", 4: "#d62728"}
    for k in (2, 3, 4):
        s = label == k
        ax[0].hist(fa[s], bins, weights=vol[s] / vol[s].sum(), histtype="step", lw=1.5, color=cols[k],
                   label="%s (%d el., median %.3f)" % (TISSUE[k], s.sum(), np.median(fa[s])))
        say("%-12s %6d elements, %8.0f mm^3, FA median %.3f, mean (vol.) %.3f, FA > 0.2: %.1f%% (vol. %.1f%%)"
            % (TISSUE[k], s.sum(), vol[s].sum(), np.median(fa[s]), np.average(fa[s], weights=vol[s]),
               100 * np.mean(fa[s] > 0.2), 100 * vol[s][fa[s] > 0.2].sum() / vol[s].sum()))
    ax[0].set_xlabel("FA (HCP1065 template)"); ax[0].set_ylabel("volume fraction per bin")
    ax[0].set_title("FA on the Colin27 mesh elements, per tissue label", fontsize=10); ax[0].legend(fontsize=8)

    fam = nib.load(HCP_FA).get_fdata()
    jm = nib.load(JHU_LABELS).get_fdata()
    names = list(JHU)
    mni = [np.median(fam[jm == JHU[n]]) for n in names]
    col = [np.median(fa[(jhu == JHU[n]) & (label == 4)]) if ((jhu == JHU[n]) & (label == 4)).any() else np.nan for n in names]
    nwm = [((jhu == JHU[n]) & (label == 4)).sum() for n in names]
    frac_wm = [np.mean(label[jhu == JHU[n]] == 4) if (jhu == JHU[n]).any() else np.nan for n in names]
    xb = np.arange(len(names))
    ax[1].bar(xb - 0.2, mni, 0.4, label="template, MNI voxels", color="#7f7f7f")
    ax[1].bar(xb + 0.2, col, 0.4, label="Colin27 WM elements", color="#d62728")
    ax[1].set_xticks(xb); ax[1].set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("median FA"); ax[1].legend(fontsize=8)
    ax[1].set_title("JHU tracts: FA before / after registration", fontsize=10)
    say("\nJHU label             MNI FA   Colin27 WM-element FA  (n WM elements, fraction of the label's elements in WM)")
    for n, a, b, c, d in zip(names, mni, col, nwm, frac_wm):
        say("%-20s  %.3f    %.3f   (%d, %.2f)" % (n, a, b, c, d))
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "check_fa.png"), dpi=110); plt.close(fig)

    # ---- direction-coloured maps, voxel field and per element
    Lc = nib.load(os.path.join(OUT, "logtensor_in_colin.nii.gz")).get_fdata(dtype=np.float32)
    brain = nib.load(COLIN_MASK).get_fdata() > 0.5
    tree = cKDTree(e["centroid"])
    rgb_el = dec(fa, v1)
    rgb_el[label != 4] *= 0.35                         # non-WM elements dimmed
    rgb_el[label == 1] = 0
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))
    eslices = {}
    for c, (name, (axis, w)) in enumerate(SLICES.items()):
        f_, v_ = voxel_slice(Lc, brain, axis, w)
        show(ax[0, c], dec(f_, v_), axis, "voxel field, " + name)
        idx = element_slice(e, node, elem, tree, axis, w, f_.shape)
        eslices[name] = idx
        img = np.where(idx[..., None] >= 0, rgb_el[np.maximum(idx, 0)], 0)
        show(ax[1, c], img, axis, "mesh elements (WM bright, CSF/GM dimmed), " + name)
    fig.suptitle("HCP1065 principal direction on Colin27 — red L–R, green A–P, blue S–I, brightness FA/%.1f" % FA_SCALE)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "check_dec.png"), dpi=110); plt.close(fig)

    # ---- corpus callosum and corticospinal path
    wm = label == 4
    cc = wm & np.isin(jhu, [JHU["genu CC"], JHU["body CC"], JHU["splenium CC"]])
    cst = wm & np.isin(jhu, [JHU[k] for k in ("CST R", "CST L", "cerebral peduncle R", "cerebral peduncle L", "PLIC R", "PLIC L")])
    xw = e["centroid"][:, 0] - MESH_ORIGIN[0]
    say("\ncorpus callosum (JHU genu+body+splenium, WM elements): n = %d, median |v1.x| = %.3f, "
        "|v1.x| > 0.8 in %.1f%%; midline |x| < 5 mm: n = %d, |v1.x| > 0.8 in %.1f%%"
        % (cc.sum(), np.median(np.abs(v1[cc, 0])), 100 * np.mean(np.abs(v1[cc, 0]) > 0.8),
           (cc & (np.abs(xw) < 5)).sum(), 100 * np.mean(np.abs(v1[cc & (np.abs(xw) < 5), 0]) > 0.8)))
    for k in ("CST R", "CST L", "cerebral peduncle R", "cerebral peduncle L", "PLIC R", "PLIC L"):
        s = wm & (jhu == JHU[k])
        if s.any():
            say("%-20s n = %4d, median |v1.z| = %.3f, |v1.z| > 0.7 in %.1f%%"
                % (k, s.sum(), np.median(np.abs(v1[s, 2])), 100 * np.mean(np.abs(v1[s, 2]) > 0.7)))
        else:
            say("%-20s no WM elements" % k)
    # x-flip check: forceps minor (genu) runs lateral + anterior -> sign(v1x v1y) = sign(x);
    # forceps major (splenium) lateral + posterior -> sign(v1x v1y) = -sign(x)
    for k, sgn in (("genu CC", 1), ("splenium CC", -1)):
        s = wm & (jhu == JHU[k]) & (np.abs(xw) > 8)
        p = v1[s, 0] * v1[s, 1] * np.sign(xw[s]) * sgn
        say("x-flip check, %s lateral (|x| > 8 mm, n = %d): fraction with the expected sign of v1x*v1y = %.2f"
            % (k, s.sum(), np.mean(p > 0)))
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    for c, (name, which) in enumerate((("sagittal x = 0 mm", "cc"), ("coronal y = -20 mm", "cst"))):
        axis, w = SLICES[name]
        idx = eslices[name]
        sel = cc if which == "cc" else cst
        rgb = dec(fa, v1) * (0.25 + 0.75 * sel[:, None])
        rgb[~wm] = 0
        img = np.where(idx[..., None] >= 0, rgb[np.maximum(idx, 0)], 0)
        show(ax[0, c], img, axis, "WM elements, %s (%s bright)" % (name, "JHU corpus callosum" if which == "cc" else "JHU CST path"))
    b = np.linspace(0, 1, 41)
    ax[1, 0].hist(np.abs(v1[cc, 0]), b, color="#d62728", alpha=0.8, label="|v1·x| (L–R)")
    ax[1, 0].hist(np.abs(v1[cc, 2]), b, histtype="step", color="k", label="|v1·z| (S–I)")
    ax[1, 0].set_title("corpus callosum WM elements (n = %d)" % cc.sum(), fontsize=10); ax[1, 0].legend()
    ax[1, 1].hist(np.abs(v1[cst, 2]), b, color="#1f77b4", alpha=0.8, label="|v1·z| (S–I)")
    ax[1, 1].hist(np.abs(v1[cst, 0]), b, histtype="step", color="k", label="|v1·x| (L–R)")
    ax[1, 1].set_title("CST + cerebral peduncle + PLIC WM elements (n = %d)" % cst.sum(), fontsize=10); ax[1, 1].legend()
    for a in ax[1]:
        a.set_xlabel("|component of the principal direction|"); a.set_ylabel("elements")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "check_cc_cst.png"), dpi=110); plt.close(fig)
    with open(os.path.join(OUT, "checks.txt"), "w") as f:
        f.write("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
