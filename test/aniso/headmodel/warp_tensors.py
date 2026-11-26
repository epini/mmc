#!/usr/bin/env python3
"""Steps c, d: HCP1065 tensors -> Colin27 grid (with reorientation) -> Colin27 mesh elements.

    python warp_tensors.py            # after register.py, conda env headenv

1. HCP1065 tensor (FSL voxel axes, LAS) -> world RAS: D = P D_fsl P, P = diag(-1, 1, 1).
   Eigenvalues clamped to >= 1e-5 mm^2/s; outside the MNI brain mask an isotropic 3e-3 mm^2/s
   (free water), so that interpolation at the brain edge stays SPD.
2. Log-Euclidean: L = log(D) per voxel; the six components of L are resampled on the Colin27
   grid (linear) with the forward transforms of register.py: L(phi(x)), phi = Colin27 -> MNI.
3. phi itself is obtained by resampling three images holding the MNI world coordinates (exact for
   linear interpolation of a linear function, no LPS/RAS ambiguity). J = d phi / dx (central
   differences, 1 mm Colin27 grid with RAS axes). The local map MNI -> Colin27 is F = J^-1;
   finite-strain reorientation R = F (F^T F)^-1/2 (polar part, R = U V^T from the SVD of F):
   L'(x) = R L(phi(x)) R^T. Never the components as scalars.
4. Mesh: L' interpolated (trilinear, Log-Euclidean) at the centroid of each tetrahedron, D = exp(L').
   Saved for every element; meaningful for the brain labels (CSF, GM, WM).

Output OUT/colin27_hcp1065_elem.npz:
  D (ne, 6) mm^2/s, MMC packing [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] in the mesh (RAS) frame;
  fa (ne,), evals (ne, 3) ascending, v1 (ne, 3) principal direction, label (ne,), centroid (ne, 3)
  mesh frame, inbrain (ne,) warped MNI brain mask at the centroid, jhu (ne,) JHU label (checks),
  detJ (ne,) Jacobian determinant of phi. Also OUT/logtensor_in_colin.nii.gz (6 comps, FSL order).
"""

import os
import time

import ants
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

from common import (COLIN_T1, COLIN_MASK, MNI_MASK, HCP_TENSOR, JHU_LABELS, OUT, load_mesh,
                    sym6_to_mat, mat_to_sym6, spd_fun, fa_v1)

XFM = [os.path.join(OUT, "mni2colin_1Warp.nii.gz"), os.path.join(OUT, "mni2colin_0GenericAffine.mat")]
LAM_MIN, LAM_BG = 1e-5, 3e-3


def to_colin(ref_colin, ref_mni, arr, interp="linear"):
    img = ref_mni.new_image_like(np.ascontiguousarray(arr, dtype=np.float32))
    return ants.apply_transforms(ref_colin, img, transformlist=XFM, interpolator=interp).numpy()


def main():
    t0 = time.time()
    tn = nib.load(HCP_TENSOR)
    ref_mni = ants.image_read(MNI_MASK)
    mask_n = nib.load(MNI_MASK).get_fdata() > 0.5
    assert np.array_equal(ref_mni.numpy() > 0.5, mask_n), "ANTs and nibabel voxel order differ"
    ref_colin = ants.image_read(COLIN_T1)
    cn = nib.load(COLIN_T1)
    assert np.allclose(cn.affine[:3, :3], np.eye(3)), "Colin27 grid expected RAS, 1 mm"

    # 1. world-RAS tensors and their logarithm in MNI space
    P = np.diag([-1.0, 1.0, 1.0])
    D = sym6_to_mat(tn.get_fdata(dtype=np.float32)[mask_n].astype(np.float64), "fsl")
    D = P @ D @ P
    L = np.zeros(mask_n.shape + (6,), np.float32)
    L[...] = mat_to_sym6(np.log(LAM_BG) * np.eye(3), "fsl")
    L[mask_n] = mat_to_sym6(spd_fun(D, lambda w: np.log(np.maximum(w, LAM_MIN))), "fsl")
    del D
    print("log-tensors in MNI space: %d brain voxels, %.0f s" % (mask_n.sum(), time.time() - t0))

    # 2. resample the log components and the MNI world coordinates on the Colin27 grid
    Lc = np.stack([to_colin(ref_colin, ref_mni, L[..., k]) for k in range(6)], -1)
    del L
    ijk = np.indices(mask_n.shape, dtype=np.float64)
    xyz = np.einsum("ab,b...->a...", tn.affine[:3, :3], ijk) + tn.affine[:3, 3, None, None, None]
    phi = np.stack([to_colin(ref_colin, ref_mni, xyz[a]) for a in range(3)], 0)
    del ijk, xyz
    inbrain_img = to_colin(ref_colin, ref_mni, mask_n.astype(np.float32))
    jhu_img = to_colin(ref_colin, ants.image_read(JHU_LABELS), nib.load(JHU_LABELS).get_fdata(), "nearestNeighbor")
    print("resampled on Colin27 grid, %.0f s" % (time.time() - t0))

    # 3. Jacobian of phi and finite-strain rotation, inside the (dilated) Colin27 brain mask
    roi = binary_dilation(nib.load(COLIN_MASK).get_fdata() > 0.5, iterations=3)
    J = np.empty((roi.sum(), 3, 3))
    for a in range(3):
        for b, g in enumerate(np.gradient(phi[a], axis=(0, 1, 2))):
            J[:, a, b] = g[roi]
    del phi
    F = np.linalg.inv(J)
    U, _, Vt = np.linalg.svd(F)
    R = U @ Vt
    Lr = sym6_to_mat(Lc[roi].astype(np.float64), "fsl")
    Lc[roi] = mat_to_sym6(R @ Lr @ np.swapaxes(R, -1, -2), "fsl")
    detj = np.ones(roi.shape, np.float32)
    detj[roi] = np.linalg.det(J)
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    inb = inbrain_img[roi] > 0.5
    print("reorientation: rotation angle median %.1f deg, 95th pct %.1f deg; det J median %.3f (brain)"
          % (np.median(ang[inb]), np.percentile(ang[inb], 95), np.median(detj[roi][inb])))
    del J, F, U, Vt, R, Lr
    nib.save(nib.Nifti1Image(Lc, cn.affine), os.path.join(OUT, "logtensor_in_colin.nii.gz"))

    # 4. sample on the element centroids (mesh frame = Colin27 0-based voxel coordinates)
    node, elem, label = load_mesh()
    cen = node[elem - 1].mean(1)
    Le = np.stack([map_coordinates(Lc[..., k], cen.T, order=1, mode="nearest") for k in range(6)], -1)
    De = spd_fun(sym6_to_mat(Le.astype(np.float64), "fsl"), np.exp)
    fa, w, v1 = fa_v1(De)
    out = dict(D=mat_to_sym6(De, "mmc").astype(np.float32), fa=fa.astype(np.float32), evals=w.astype(np.float32),
               v1=v1.astype(np.float32), label=label, centroid=cen.astype(np.float32),
               inbrain=map_coordinates(inbrain_img, cen.T, order=1).astype(np.float32),
               jhu=map_coordinates(jhu_img, cen.T, order=0).astype(np.int16),
               detJ=map_coordinates(detj, cen.T, order=1).astype(np.float32))
    np.savez_compressed(os.path.join(OUT, "colin27_hcp1065_elem.npz"), **out)
    wm = label == 4
    print("%d elements, %d WM; WM FA median %.3f, FA > 0.2 in %.1f%% of WM elements; %.0f s"
          % (len(label), wm.sum(), np.median(fa[wm]), 100 * np.mean(fa[wm] > 0.2), time.time() - t0))


if __name__ == "__main__":
    main()
