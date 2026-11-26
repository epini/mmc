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

## Test 4 — two layers with orthogonal fibres (cases C, D)

Same slab, cut into two 5 mm layers with the same n = 1.4, g = 0.8, μa (C: 0, D: 0.01 mm⁻¹) and
uniaxial tensors M = μs⊥ I + (μs∥ − μs⊥) f̂f̂ᵀ, μs∥ = 5, μs⊥ = 20 mm⁻¹: fibres f̂ = x̂ in the top layer,
ŷ in the bottom one. Only the tensor changes at z = 5 mm (no Fresnel there): the photons cross the face
between two tensor domains. PyXOpto: two `AnisotropicLayer`s. MMC: two stacked 6-tetrahedron boxes with a
conforming common face and element labels 1, 2 (`run_pmmc.py`, pmmc on the GPU, 10⁸ photons in ~1 min).

```
python run_pyxopto.py --case C --nphoton 1e8
python run_pmmc.py    --case C --nphoton 1e8      # compute="opencl", gpuid=1; --compute sse for the CPU
python compare.py     --case C --plot compare_C.png
```

Results (26/11/2026, 10⁸ photons per code, both on the RTX 3050):

| | Case C (μa = 0) | Case D (μa = 0.01 mm⁻¹) |
|---|---|---|
| R: MMC / PyXOpto | 0.88476 / 0.88479 (−0.7σ) | 0.70087 / 0.70087 (0.0σ) |
| T: MMC / PyXOpto | 0.08746 / 0.08743 (0.8σ) | 0.01466 / 0.01466 (0.0σ) |
| var R x, y (mm²) | (18.32, 9.11) / (18.33, 9.11) | (7.44, 2.85) / (7.44, 2.85) |
| var T x, y (mm²) | (82.34, 82.46) / (82.41, 82.35) | (46.94, 47.01) / (46.99, 46.95) |
| R(x,y), T(x,y) map, reduced χ² | 0.94, 0.97 (≈3100 bins) | 0.34, 0.22* |
| R, T(optical path), reduced χ² | 0.97, 1.01 | 1.07, 0.88 (exact variance, opl < 1285 mm)** |

R(x,y) is elongated along the fibres of the top layer (var x / var y ≈ 2); T(x,y) is nearly round,
the two orthogonal layers compensate each other. Plots: `compare_C.png`, `compare_D.png`.

\* Counting-statistics errors, an overestimate with absorption (see above).
\** With absorption `compare.py` also uses the exact MMC variance (Σw², saved by `run_pmmc.py`) for both
codes, in the bins where exp(−μa·opl/n) > 10⁻⁴. Below that weight PyXOpto terminates packets by lottery
(`McMinimumPacketWeight` = 10⁻⁴, survival 1%, weight ×100): its variance there is ~100× larger, and over all
bins the χ² with the MMC variance would be ≈ 10, driven only by the opl > 1500 mm tail.

### Consistency check: one tensor, two element regions

`split_check.py`: the single-tensor slab of case A or B meshed as two boxes (3 + 7 mm, labels 1 and 2, same
properties and tensor) vs the 6-element slab, 10⁷ photons each, independent seeds, GPU.

| | R, z | T, z | χ²/bin: R opl, R map, T opl, T map |
|---|---|---|---|
| A (μa = 0) | −0.34 | +0.93 | 1.05, 0.99, 1.18, 1.00 |
| B (μa = 0.01 mm⁻¹) | −0.18 | +0.12 | 1.03, 0.97, 1.16, 1.00 |

With the same seed the two meshes give R, T equal to |z| < 0.1 (correlated runs, for information).
