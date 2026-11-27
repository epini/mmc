"""Mouse optogenetics block test (CLAUDE.md step 2b): every parameter in one place.

ALL OPTICAL VALUES ARE PLACEHOLDERS, to be replaced by his choice (CLAUDE.md step 2b).
Units: mm, 1/mm, ns. z points down into the tissue (depth), the top face z = 0 is the tissue surface.
"""

import os

# ---------------------------------------------------------------- optics (PLACEHOLDERS)
N_TISSUE = 1.36
G = 0.88
MUS_CC_OVER_GM = 6.0                     # CC mean mus = 6 x GM mus
R_LIST = [1.0, 0.7, 0.5, 0.3, 0.2]       # r = mus_par / mus_perp in the CC; r = 1 is the isotropic reference
WAVELENGTHS = {
    # nm: GM mus (1/mm), mua (1/mm, GM and CC); g, n as above
    473: dict(mus_gm=21.0, mua=0.10),                             # Yona 2016 (~47 um scattering length)
    590: dict(mus_gm=21.0 * (590.0 / 473.0) ** -1.3, mua=0.05),   # mus ~ lambda^-1.3
}


def cc_tensor(wl, r):
    """CC scattering tensor, uniaxial along x (axons medio-lateral), (mus_par + 2 mus_perp)/3 fixed:
    returns (mus_par, mus_perp)."""
    m = MUS_CC_OVER_GM * WAVELENGTHS[wl]["mus_gm"]
    perp = 3.0 * m / (r + 2.0)
    return r * perp, perp


# ---------------------------------------------------------------- geometry
BLOCK = 6.0                              # GM block, x, y in [-3, 3], z in [0, 6]
CC_TOP, CC_THICK = 1.00, 0.25            # CC slab: z in [1.00, 1.25], normal z
TIPS = {"above": CC_TOP - 0.30,          # fibre tip positions (depth of the tip centre)
        "inside": CC_TOP + CC_THICK / 2,
        "below": CC_TOP + CC_THICK + 0.30}
# mesh: structured tetrahedra; edges MESH_FINE in x, y everywhere and in z inside FINE_Z, growing
# geometrically (ratio MESH_GROW) up to MESH_COARSE in z outside; nodes on the slab planes.
# With the dual-grid output the mesh only has to carry the geometry (the slab planes): 0.05 and 0.125 mm
# gave the same fields and metrics (27/11, r = 0.2, tip inside, 5e6 photons each: all metrics |z| < 1.6,
# voxelwise mean z -0.01) and 0.125 has 11x fewer tetrahedra (4.6x faster). 0.05 was the plan's value.
MESH_FINE, MESH_COARSE, MESH_GROW = 0.125, 0.25, 1.25
FINE_Z = (0.50, 2.00)
GRID = 0.02                              # output voxel (dual-grid MMC, method 'grid')

# ---------------------------------------------------------------- fibre
CORE_RADIUS = 0.100                      # 200 um core, flat cleaved
NA = 0.37                                # half-angle in tissue asin(NA / N_TISSUE), uniform in solid angle
FIBRE_DIR = (0.0, 0.0, 1.0)              # pointing +z (into depth), perpendicular to the axons (x)
POWER_MW = 10.0                          # power out of the fibre

# ---------------------------------------------------------------- simulation and metrics
NPHOTON = 1e8                            # per case (one r, one tip, one wavelength)
NBATCH = 10                              # batches for the error bars (jackknife)
SEED = 1648335518
TEND = 5e-9                              # s; CW = one gate, all photons are dead long before
THRESHOLDS_MW_MM2 = [0.1, 1.0, 10.0]     # irradiance thresholds for the activated volume; 1 is the main one
MAIN_THRESHOLD = 1.0
BELOW_DZ = 0.05                          # "just below the slab": profile at CC bottom + BELOW_DZ

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.environ.get("MOUSE_DATA", os.path.join(REPO, "..", "mouse_data"))
OUT = os.path.join(DATA, "out")
