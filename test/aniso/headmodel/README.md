# Head model: Colin27 mesh + HCP1065 DTI tensors

Python pipeline (conda env `headenv`: antspyx 0.6.3, nibabel, scipy, matplotlib; separate from
`pyxoptoenv` because antspyx pins an older scipy). Data are not in git: see `common.py` for the
sources and the folder layout (`HEADMODEL_DATA`, default `<repo>/../../headmodel_data`).

| step | script | what |
|---|---|---|
| a | (download) | Colin27 mesh V2L (fangq/Colin27BrainMesh, 70 226 nodes, 423 375 tetrahedra, labels 1 scalp+skull, 2 CSF, 3 GM, 4 WM); MNI BIC Colin27 1998 T1; FSL `fsl-data_standard` 2208.0 (MNI152 T1 1 mm, HCP1065 tensor/FA, FMRIB58_FA); JHU ICBM-DTI-81 labels (checks only) |
| b | `register.py` | ANTsPy affine + SyN, MNI152 brain -> Colin27 brain (~5 min); correlation with the Colin27 T1 0.927 (affine) -> 0.977 (SyN) |
| c, d | `warp_tensors.py` | FSL LAS -> RAS (P D P), Log-Euclidean resampling, finite-strain reorientation from the Jacobian of the warp, tensor at each element centroid -> `out/colin27_hcp1065_elem.npz` (~50 s) |
| g | `warp_tensors.py` | anisotropy magnitude from FMRIB58_FA (same registration, scalar): per element, HCP1065 eigenvectors, MD and (λ2−λ3)/(λ1−λ3) kept, eigenvalue spread rescaled to FA = FMRIB58 (`common.rescale_fa`); both FAs kept in the npz |
| e | `checks.py` | `check_fa.png`, `check_dec.png`, `check_cc_cst.png`, numbers in `out/checks.txt` |
| f | — | FA -> OFA -> (mus_par, mus_perp): **not done, to be decided** |

Mesh frame: nodes are 0-based voxel coordinates of `colin27_t1_tal_lin.nii` (RAS, 1 mm, no flip;
found from the T1 white/grey contrast of the element centroids, the x-flipped mesh halves it).
World RAS = node − (90, 126, 72) mm. Tensors are stored in the mesh frame (x L→R, y P→A, z I→S).

## Results of the checks (26/11/2026)

- Registration: median local rotation of the warp 8.4°, 95th percentile 19.5°; JHU tract FA on the
  Colin27 WM elements close to the template value in MNI (`check_fa.png`, right; genu 0.52 vs 0.60,
  the thin genu loses most to the coarse elements).
- HCP1065 FA is the FA of the mean tensor of 1065 subjects (CC 0.28–0.37, max 0.54): orientations
  only. FMRIB58 FA in MNI: genu 0.60, body 0.60, splenium 0.73, CST 0.43–0.45, PLIC 0.60.
- FA rescaling: FA = FMRIB58 and MD, (λ2−λ3)/(λ1−λ3), v1 unchanged to 1e-13; no element needed the
  positivity cap (FA bound of the shape). FMRIB58 is 0 outside its brain mask: there the HCP1065
  tensor is kept (flag `hcp_fallback`: WM 2.3%, GM 23%, CSF 54% of the elements).
- FA on the elements, median (HCP1065 before): CSF 0.025 (0.015), GM 0.100 (0.031), WM 0.223 (0.103);
  WM volume with FA > 0.2: 70% (26% before). Colin27 WM elements per JHU tract: genu 0.52, body 0.60,
  splenium 0.72, CST 0.44–0.45, cerebral peduncle 0.66, PLIC 0.59–0.60. The Colin27 WM label
  (FreeSurfer white surface) includes subcortical WM and the deep nuclei: low-FA tail.
- Corpus callosum (JHU genu + body + splenium, 3404 WM elements): |v1·x| > 0.8 in 97% of the
  midline elements (|x| < 5 mm), 58% overall (the lateral parts fan out).
- Corticospinal path (JHU CST, cerebral peduncle, PLIC, 602 WM elements): |v1·z| > 0.7 in 63–94%
  per label, median |v1·z| 0.76–0.92.
- x-flip check (sign of v1x·v1y): forceps minor lateral and anterior, 100% of 316 genu elements;
  forceps major lateral and posterior, 90% of 909 splenium elements. With the wrong LAS -> RAS
  conversion these fractions would be ~0 and ~0.1.

Licences: HCP open-access data use terms (acknowledge HCP; derived data under the same terms),
FSL licence (non-commercial), Colin27 mesh public domain, MNI Colin27 (McConnell Brain Imaging Centre).
