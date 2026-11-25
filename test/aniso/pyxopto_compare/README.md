# MMC aniso vs PyXOpto — full scattering tensor

Slab L = 10 mm, n = 1.4 in air, HG g = 0.8, pencil beam along +z.
Tensor with eigenvalues (5, 10, 20) mm⁻¹ and principal axes rotated by ZYZ Euler angles (30°, 40°, 50°):
all off-diagonal components are non-zero, z is not a principal axis.
Case A: μa = 0. Case B: μa = 0.01 mm⁻¹.

Both codes give R and T binned in (optical path length, y, x) with the same grid
(`common.py`) and the same clipping into the edge bins (PyXOpto `CartesianPl` convention).
Same tensor definition in both: μs(ŝ) = ŝᵀ M ŝ (PyXOpto `tensor3f_project`).

## Running

MMC (CPU, from this branch):
```
python3 run_mmc.py --case A --nphoton 1e7
```

PyXOpto (OpenCL GPU, or CPU as fallback):
```
python run_pyxopto.py --case A --nphoton 1e8
```

Comparison and plot:
```
python3 compare.py --case A --plot compare_A.png
```

`compare.py` scales MMC by (1 − R_sp): MMC launches the photon inside the medium,
PyXOpto removes the specular reflection at the source and reports it separately.

Note: with recent SciPy (≥ 1.14) and pyopencl (≥ 2024.2) the public PyXOpto (Jan 2024) needs two
one-line fixes (`scipy.integrate.simps` → `simpson`; `CommandQueue(..., properties=None)` → `0`).
Not needed on an environment where PyXOpto already works.

## Results (25/11/2026, 10⁷ photons per code per case, both on CPU; PyXOpto via pocl)

| | Case A (μa = 0) | Case B (μa = 0.01 mm⁻¹) |
|---|---|---|
| R: MMC / PyXOpto | 0.85736 / 0.85712 (1.6σ) | 0.68022 / 0.68005 (0.8σ) |
| T: MMC / PyXOpto | 0.11486 / 0.11510 (1.7σ) | 0.03120 / 0.03123 (0.4σ) |
| centroid T (x, y), mm | (−2.286, −4.089) / (−2.283, −4.077) | (−2.284, −4.075) / (−2.277, −4.068) |
| cov T (xx, yy, xy), mm² | (44.30, 73.62, −7.43) / (44.12, 73.68, −7.50) | (28.12, 46.84, −4.69) / (28.12, 46.93, −4.74) |
| T(x,y) map, reduced χ² | 0.92 (1195 bins) | 0.35 (546 bins)* |
| T(optical path), reduced χ² | 1.23 (104 bins) | 0.26 (49 bins)* |

The transmitted spot is displaced from the beam axis by (−2.3, −4.1) mm because of Mxz, Myz ≠ 0:
both codes reproduce it. Plots: `compare_A.png`, `compare_B.png`.

\* With absorption the weights are < 1 and the counting-statistics error used by `compare.py` is an
overestimate, hence χ² < 1.

Note on `run_mmc.py`: the MMC `.mch` stores the launch weight of detected photons; absorption is applied
afterwards as exp(−μa · partial path), as in MCX/MMC.
