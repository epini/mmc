"""Shared definition of the MMC vs PyXOpto comparison cases.

Slab of thickness L between two half-spaces of index 1, pencil beam along +z at the
origin of the top face, scattering tensor with three distinct eigenvalues and principal
axes rotated so that the tensor has all off-diagonal components (Mxy, Mxz, Myz != 0).
Units here: mm, mm^-1. PyXOpto works in m: the conversion is done in run_pyxopto.py.

Cases C and D (test 4) stack two layers with the same mua, g, n and orthogonal fibres
(uniaxial tensors, fibres along x in the top layer and along y in the bottom one): the photons
cross the face between two tensor domains, with no Fresnel reflection there.

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
    # two layers of 5 mm, mus_par = 5, mus_perp = 20 1/mm, fibres along x (top) and y (bottom)
    "C": dict(L=10.0, n=1.4, mua=0.0, g=0.8, mus_par=5.0, mus_perp=20.0,
              layers=((5.0, 0.0), (5.0, 90.0))),
    "D": dict(L=10.0, n=1.4, mua=0.01, g=0.8, mus_par=5.0, mus_perp=20.0,
              layers=((5.0, 0.0), (5.0, 90.0))),
}

# Test 5: four-layer head of Pini et al., Neurophotonics 12, 045003 (2025), Table 1 (830 nm).
# (thickness mm, n, g, mua 1/mm, reduced scattering mus' 1/mm); mus = mus' / (1 - g). White matter:
# uniaxial, mus'_par = 2.6 along the fibres, mus'_perp = 4.4. HX: fibres along x, HY: along y.
HEAD = (("scalp/skull", 10.0, 1.39, 0.89, 0.019, 0.86),
        ("CSF", 2.5, 1.34, 0.89, 0.0026, 0.16),
        ("gray matter", 3.0, 1.39, 0.90, 0.030, 0.70),
        ("white matter", 10.0, 1.39, 0.97, 0.043, (2.6, 4.4)))
for _k, _phi in (("HX", 0.0), ("HY", 90.0)):
    CASES[_k] = dict(L=sum(l[1] for l in HEAD), head=HEAD, wm_phi=_phi,
                     xy=(-50.0, 50.0, 20), pl=(0.0, 1800.0, 360))   # 5 mm pixels; opl 5 mm = 0.0167 ns

# detector grid, mm (bin edges = linspace(start, stop, n+1)); a case can override with "xy", "pl"
XY_AXIS = (-50.0, 50.0, 100)
PL_AXIS = (0.0, 3000.0, 300)     # optical path length n*geometric length, mm
C_MM_NS = 299.792458             # mm/ns: time = optical path length / c


def axes(case):
    c = CASES[case]
    return c.get("xy", XY_AXIS), c.get("pl", PL_AXIS)


def rotation_zyz(a, b, c):
    a, b, c = np.deg2rad([a, b, c])
    Rz1 = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz2 = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz1 @ Ry @ Rz2


def mus_tensor(case):
    """Full symmetric 3x3 scattering tensor in mm^-1 (single-layer cases)."""
    c = CASES[case]
    R = rotation_zyz(*c["euler_zyz"])
    return R @ np.diag(c["eig"]) @ R.T


def uniaxial(mus_par, mus_perp, phi_deg):
    """Fibre along (cos phi, sin phi, 0): M = mus_perp I + (mus_par - mus_perp) f f^T, 1/mm."""
    f = np.array([np.cos(np.deg2rad(phi_deg)), np.sin(np.deg2rad(phi_deg)), 0.0])
    return mus_perp * np.eye(3) + (mus_par - mus_perp) * np.outer(f, f)


def layers(case):
    """List of (thickness in mm, 3x3 tensor in 1/mm), from the top (z = 0) down."""
    c = CASES[case]
    if "layers" not in c:
        return [(c["L"], mus_tensor(case))]
    if "head" in c:
        return [(l["d"], l["M"]) for l in stack(case)]
    return [(d, uniaxial(c["mus_par"], c["mus_perp"], phi)) for d, phi in c["layers"]]


def stack(case):
    """List of layer dicts (d, M, mua, g, n, name) from the top down; scalars per layer."""
    c = CASES[case]
    if "head" not in c:
        return [dict(d=d, M=M, mua=c["mua"], g=c["g"], n=c["n"], name="layer %d" % (k + 1))
                for k, (d, M) in enumerate(layers(case))]
    out = []
    for name, d, n, g, mua, musr in c["head"]:
        if np.isscalar(musr):
            M = musr / (1 - g) * np.eye(3)
        else:
            M = uniaxial(musr[0] / (1 - g), musr[1] / (1 - g), c["wm_phi"])
        out.append(dict(d=d, M=M, mua=mua, g=g, n=n, name=name))
    return out


def edges(ax):
    return np.linspace(ax[0], ax[1], ax[2] + 1)


def bin_clipped(values, ax):
    """Bin index with clipping into the edge bins (PyXOpto convention)."""
    idx = np.floor((values - ax[0]) / (ax[1] - ax[0]) * ax[2]).astype(np.int64)
    return np.clip(idx, 0, ax[2] - 1)
