"""Shared paths and conventions of the Colin27 + HCP1065 head model.

Data (not in git) live in HEADMODEL_DATA, default <repo>/../../headmodel_data:
  colin27/MMC_Collins_Atlas_Mesh_Version_2L.mat      github.com/fangq/Colin27BrainMesh (public domain)
  colin27/mni_colin27_1998/colin27_t1_tal_lin*.nii   MNI BIC Colin27 1998 T1 (+ brain/head masks)
  fsl/data/standard/MNI152_T1_1mm*.nii.gz            FSL fsl-data_standard 2208.0 (FSL licence)
  fsl/data/standard/FSL_HCP1065_*_1mm.nii.gz         HCP1065 DTI template (HCP open-access terms)
  fsl/data/atlases/JHU/JHU-ICBM-labels-1mm.nii.gz    JHU ICBM-DTI-81 labels, only for the checks
Outputs go to HEADMODEL_DATA/out.

Frames. The mesh nodes are 0-based voxel coordinates of colin27_t1_tal_lin.nii (181x217x181,
1 mm, RAS, no flip: checked on the T1 white/grey contrast); world RAS = node - (90, 126, 72) mm.
MMC runs in the mesh frame, whose axes are the RAS axes: x left->right, y posterior->anterior,
z inferior->superior. Tensors are stored in that frame. HCP1065 (FSL, LAS storage): FSL tensor
order Dxx, Dxy, Dxz, Dyy, Dyz, Dzz in voxel axes; voxel axis i points to world -x, so the world
(RAS) tensor is P D P with P = diag(-1, 1, 1) (Dxy, Dxz change sign).
"""

import os

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.environ.get("HEADMODEL_DATA", os.path.join(REPO, "..", "..", "headmodel_data"))
OUT = os.path.join(DATA, "out")

COLIN_T1 = os.path.join(DATA, "colin27", "mni_colin27_1998", "colin27_t1_tal_lin.nii")
COLIN_MASK = os.path.join(DATA, "colin27", "mni_colin27_1998", "colin27_t1_tal_lin_mask.nii")
COLIN_MESH = os.path.join(DATA, "colin27", "MMC_Collins_Atlas_Mesh_Version_2L.mat")
STD = os.path.join(DATA, "fsl", "data", "standard")
MNI_T1 = os.path.join(STD, "MNI152_T1_1mm.nii.gz")
MNI_BRAIN = os.path.join(STD, "MNI152_T1_1mm_brain.nii.gz")
MNI_MASK = os.path.join(STD, "MNI152_T1_1mm_brain_mask.nii.gz")
HCP_TENSOR = os.path.join(STD, "FSL_HCP1065_tensor_1mm.nii.gz")
HCP_FA = os.path.join(STD, "FSL_HCP1065_FA_1mm.nii.gz")
JHU_LABELS = os.path.join(DATA, "fsl", "data", "atlases", "JHU", "JHU-ICBM-labels-1mm.nii.gz")

TISSUE = {1: "scalp+skull", 2: "CSF", 3: "gray matter", 4: "white matter"}
MESH_ORIGIN = np.array([90.0, 126.0, 72.0])     # world RAS = node - MESH_ORIGIN

# JHU ICBM-DTI-81 label values used in the checks (JHU-labels.xml index = image value)
JHU = {"genu CC": 3, "body CC": 4, "splenium CC": 5, "CST R": 7, "CST L": 8,
       "cerebral peduncle R": 15, "cerebral peduncle L": 16, "PLIC R": 19, "PLIC L": 20}


def load_mesh():
    """node (nn, 3) in mesh frame (mm), elem (ne, 4) 1-based, label (ne,)."""
    import scipy.io as sio
    m = sio.loadmat(COLIN_MESH)
    return m["node"].astype(float), m["elem"][:, :4].astype(np.int64), m["elem"][:, 4].astype(np.int64)


def sym6_to_mat(t, order="fsl"):
    """(..., 6) -> (..., 3, 3). order 'fsl': xx xy xz yy yz zz; 'mmc': xx yy zz xy xz yz."""
    if order == "fsl":
        xx, xy, xz, yy, yz, zz = np.moveaxis(t, -1, 0)
    else:
        xx, yy, zz, xy, xz, yz = np.moveaxis(t, -1, 0)
    return np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1), np.stack([xz, yz, zz], -1)], -2)


def mat_to_sym6(M, order="mmc"):
    xx, yy, zz = M[..., 0, 0], M[..., 1, 1], M[..., 2, 2]
    xy, xz, yz = M[..., 0, 1], M[..., 0, 2], M[..., 1, 2]
    if order == "fsl":
        return np.stack([xx, xy, xz, yy, yz, zz], -1)
    return np.stack([xx, yy, zz, xy, xz, yz], -1)


def spd_fun(M, f):
    """f applied to the eigenvalues of symmetric matrices M (..., 3, 3)."""
    w, V = np.linalg.eigh(M)
    return np.einsum("...ij,...j,...kj->...ik", V, f(w), V)


def fa_v1(M):
    """FA, eigenvalues (ascending) and principal eigenvector of symmetric tensors (..., 3, 3)."""
    w, V = np.linalg.eigh(M)
    md = w.mean(-1, keepdims=True)
    fa = np.sqrt(1.5 * ((w - md) ** 2).sum(-1) / np.maximum((w ** 2).sum(-1), 1e-30))
    return fa, w, V[..., :, 2]
