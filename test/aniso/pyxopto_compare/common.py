"""Shared definition of the MMC vs PyXOpto comparison cases.

Slab of thickness L between two half-spaces of index 1, pencil beam along +z at the
origin of the top face, scattering tensor with three distinct eigenvalues and principal
axes rotated so that the tensor has all off-diagonal components (Mxy, Mxz, Myz != 0).
Units here: mm, mm^-1. PyXOpto works in m: the conversion is done in run_pyxopto.py.

Both codes produce, for the top (reflectance) and bottom (transmittance) surfaces, the
weight per launched photon binned in (optical path length, y, x), with out-of-range
values clipped into the edge bins (as the PyXOpto CartesianPl detector does).
"""

import numpy as np

CASES = {
    # mua = 0: geometry and tensor only
    "A": dict(L=10.0, n=1.4, mua=0.0, eig=(5.0, 10.0, 20.0), euler_zyz=(30.0, 40.0, 50.0), g=0.8),
    # same, with absorption
    "B": dict(L=10.0, n=1.4, mua=0.01, eig=(5.0, 10.0, 20.0), euler_zyz=(30.0, 40.0, 50.0), g=0.8),
}

# detector grid, mm (bin edges = linspace(start, stop, n+1))
XY_AXIS = (-50.0, 50.0, 100)
PL_AXIS = (0.0, 3000.0, 300)     # optical path length n*geometric length, mm


def rotation_zyz(a, b, c):
    a, b, c = np.deg2rad([a, b, c])
    Rz1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz2 = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2


def mus_tensor(case):
    """Full symmetric 3x3 scattering tensor in mm^-1."""
    c = CASES[case]
    R = rotation_zyz(*c["euler_zyz"])
    return R @ np.diag(c["eig"]) @ R.T


def edges(ax):
    return np.linspace(ax[0], ax[1], ax[2] + 1)


def bin_clipped(values, ax):
    """Bin index with clipping into the edge bins (PyXOpto convention)."""
    idx = np.floor((values - ax[0]) / (ax[1] - ax[0]) * ax[2]).astype(np.int64)
    return np.clip(idx, 0, ax[2] - 1)
