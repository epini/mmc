#!/usr/bin/env python3
"""First look at anisotropic scattering in pmmc (branch aniso).

    python demo_aniso.py

60 mm cube, pencil beam along +z at the centre of the top face. Two runs:
  - isotropic:   mus = 10 mm^-1 in every direction
  - anisotropic: fibres along x, mus_par = 5 mm^-1 along x, mus_perp = 15 mm^-1 across
Same g = 0.9, mua = 0.005 mm^-1, n = 1.37. The plot shows the CW fluence on the plane z = 10 mm
seen from above: isotropic gives circles, the fibres stretch the spot along x.
Runs on the CPU, about a minute.
"""

import numpy as np
import matplotlib.pyplot as plt
import iso2mesh as i2m
import pmmc

# mesh: 60 mm cube split into tetrahedra on a 2 mm grid (pure Python, no external mesher)
g = np.arange(0, 61, 2.0)
node, elem = i2m.meshgrid5(g, g, g)[:2]
ne = elem.shape[0]
print("mesh: %d nodes, %d elements" % (node.shape[0], ne))

cfg = {
    "nphoton": 1e6,
    "node": node,
    "elem": elem[:, :4],
    "elemprop": np.ones(ne, int),
    "prop": [[0, 0, 1, 1], [0.005, 10.0, 0.9, 1.37]],   # [mua mus g n]; mus is ignored with elemmus
    "srcpos": [30, 30, 0],
    "srcdir": [0, 0, 1],
    "tstart": 0, "tend": 5e-9, "tstep": 5e-9,            # one time gate = CW
    "compute": "sse",                                    # CPU (the anisotropic GPU kernel is not ready yet)
}


def fluence(elemmus):
    """CW fluence on the default 1 mm output grid of pmmc, shape (61, 61, 61) as [x, y, z]."""
    c = dict(cfg)
    c["elemmus"] = elemmus
    return np.asarray(pmmc.mmclab(c)["flux"])[..., 0]


# isotropic: mus * I on every element
iso = np.tile([10.0, 10.0, 10.0, 0, 0, 0], (ne, 1))
# anisotropic: fibres along x -> M = mus_perp*I + (mus_par - mus_perp) * x x^T
mus_par, mus_perp = 5.0, 15.0
ani = np.tile([mus_par, mus_perp, mus_perp, 0, 0, 0], (ne, 1))

f_iso = fluence(iso)
f_ani = fluence(ani)

z0 = 10   # mm (= grid index, 1 mm voxels from z = 0)
fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
for a, f, title in ((ax[0], f_iso, "isotropic, $\\mu_s$ = 10 mm$^{-1}$"),
                    (ax[1], f_ani, "fibres along x, $\\mu_{s\\parallel}$ = 5, $\\mu_{s\\perp}$ = 15 mm$^{-1}$")):
    sl = f[:, :, z0].T                       # rows = y, columns = x
    v = np.log10(np.maximum(sl, sl.max() * 1e-6))
    im = a.imshow(v, origin="lower", extent=[0, 60, 0, 60], cmap="magma")
    a.contour(np.arange(61), np.arange(61), v, levels=8, colors="w", linewidths=0.5)
    a.set_title(title)
    a.set_xlabel("x (mm)")
ax[0].set_ylabel("y (mm)")
fig.colorbar(im, ax=ax, label="log$_{10}$ fluence (a.u.)")
fig.suptitle("pmmc aniso: CW fluence on the plane z = %d mm" % z0)
fig.savefig("demo_aniso.png", dpi=120)
print("saved demo_aniso.png")
plt.show()
