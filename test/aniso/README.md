# Test anisotropic MMC (branch `aniso`)

## Using anisotropic scattering from Python (pmmc)

Build pmmc from this branch (needs a C/C++ compiler and CMake; on Windows the Visual Studio Build Tools):
```
git submodule update --init src/pybind11
cd pmmc && pip install .
```

Then add `cfg["elemmus"]` to an ordinary pmmc configuration, one tensor per tetrahedron, in 1/mm:
```python
import numpy as np, pmmc
cfg = {...}                          # node, elem, elemprop, prop, srcpos, srcdir, tstart/tend/tstep, ...
cfg["elemmus"] = T                   # (ne, 6) as [Mxx, Myy, Mzz, Mxy, Mxz, Myz]  or  (ne, 3, 3)
cfg["compute"] = "opencl"            # GPU (cfg["gpuid"] = 1), or "cuda"; "sse" / gpuid = -1 for the CPU
res = pmmc.mmclab(cfg)
```
- mus(s) = sᵀ M s along the direction s; the `mus` column of `prop` is ignored, `mua`, `g`, `n` still come from `prop`.
- Every element needs a tensor: isotropic tissues get mus·I, i.e. `[mus, mus, mus, 0, 0, 0]`.
- A (ne,3,3) input is symmetrized, (M + Mᵀ)/2: the antisymmetric part does not contribute to sᵀ M s.
- Wrong shape, wrong number of rows, a tensor that is not positive semi-definite, OptiX, iMMC or MCML mode: explicit error.

`test_pmmc.py` (run with `OMP_NUM_THREADS=1`) checks: bitwise isotropic regression, (ne,6) = (ne,3,3) and
non-symmetric = symmetric part (bitwise), agreement with the command line, and the error cases.
`gpu_vs_cpu.py` compares the GPU kernel with the CPU (Test 6 below).

All on CPU (`-G -1`), binary built from this branch.

## Test 1 — isotropic regression

`cube_iso_ref.json` (vanilla) vs `cube_iso_aniso.json` (`Mesh.ElemMusFile` = μs·**I** on all 6 elements),
cube2 from `examples/validation`, 10⁶ photons, `OMP_NUM_THREADS=1`.

```
export OMP_NUM_THREADS=1
../../bin/mmc -f cube_iso_ref.json   -G -1 -F mc2 -d 1
../../bin/mmc -f cube_iso_aniso.json -G -1 -F mc2 -d 1
cmp iso_ref.mc2 iso_aniso.mc2 && cmp iso_ref.mch iso_aniso.mch
```

Result: fluence and detected photons **bit-identical** for all tracers (`-M P`, `H`, `B`, `S`, default `G`).
With more than one thread, vanilla MMC is not deterministic either (photon order across threads).

## Test 2–3 — anisotropic slab vs generalized ADE

`slab_vs_ade.py`: 200×200×10 mm slab, n = 1.3 in air, μa = 0, pencil beam along z.
Observables on transmitted photons: T(t) vs `tt_ade`, and the slope of the variance of the transmitted profile
(= 2 D_i) vs `d_tensor_ade` ([epini/ADE](https://github.com/epini/ADE)).

| Case | Photons | D_x MC/ADE | D_y MC/ADE | T(t), median rel. err. |
|---|---|---|---|---|
| μs = (1, 3, 4) mm⁻¹, g = 0 | 10⁷ | −0.60% | −0.49% | 1.1% |
| same, tensor rotated 30° around z | 10⁷ | +0.89% | +0.27% | 1.1% |
| μs = (10, 30, 40) mm⁻¹, g = 0.9 | 3·10⁶ | −0.89% | +0.38% | 2.3% |

In the rotated case the correlation x′y′ in the tensor frame is ≈ 0 (±0.002): the profile rotates rigidly with the tensor.
For comparison, the "simplistic" formula v/(3 μs,x (1−g)) would give D_x = 76.9 mm²/ns in the g = 0 case
instead of 64.1: the test resolves the difference clearly.

```
python3 slab_vs_ade.py --mus 1 3 4 --g 0 --nphoton 1e7 --rot 30 --plot slab_g0_rot30.png
python3 slab_vs_ade.py --mus 10 30 40 --g 0.9 --nphoton 3e6 --plot slab_g09_rot0.png
```

## Test 4 — two layers with orthogonal fibres vs PyXOpto

Slab of two 5 mm layers, fibres along x on top and along y below (μs∥ = 5, μs⊥ = 20 mm⁻¹), MMC on the GPU
vs PyXOpto with two `AnisotropicLayer`s, plus the consistency check "one tensor, two element regions =
whole slab": see `pyxopto_compare/README.md`. R, T, maps and optical path agree within statistics.

## Test 6 — GPU kernel vs CPU

`gpu_vs_cpu.py`: slab of `pyxopto_compare` (full tensor, eigenvalues (5, 10, 20) mm⁻¹, all off-diagonals,
g = 0.8, n = 1.4), pmmc OpenCL on an RTX 3050 Laptop vs pmmc CPU, same photon count, independent streams.
Detected photons on the top (R) and bottom (T) faces; z = (GPU − CPU)/σ, χ² on T(optical path) and T(x, y).
It also checks, on the GPU, elemmus = μs·**I** against vanilla MMC (statistically: atomic float sums make
the GPU not bitwise reproducible).

```
python gpu_vs_cpu.py --case A --nphoton 1e6     # compute="opencl", gpuid=1
python gpu_vs_cpu.py --case B --nphoton 1e6
```

10⁷ photons per backend, case B (μa = 0.01 mm⁻¹), GPU / CPU:

| | GPU | CPU | z |
|---|---|---|---|
| R | 0.699322 | 0.699339 | −0.05 |
| T | 0.032163 | 0.032191 | −0.55 |
| ⟨x⟩_T, ⟨y⟩_T (mm) | −2.280, −4.076 | −2.282, −4.083 | 0.36, 0.76 |
| ⟨opl⟩_T (mm) | 149.448 | 149.454 | −0.07 |
| cov_T xx, yy, xy (mm²) | 28.03, 46.84, −4.726 | 28.01, 46.89, −4.735 | 0.37, −0.47, 0.18 |

At 10⁶ photons (cases A and B) all |z| < 2.5 and the χ² tests pass. GPU ≈ 45× faster than the CPU (all cores).
Vanilla MMC (isotropic μs = 10 mm⁻¹, case B, 10⁷ photons) gives the same GPU–CPU agreement (⟨opl⟩_T z = 0.1).

## Test 7 — CUDA vs OpenCL vs CPU

`backends_compare.py` (pmmc built with `CMAKE_ARGS=-DBUILD_CUDA=ON`): same slab and observables as Test 6,
`compute="cuda"`, `"opencl"` and `"sse"`, independent random streams, every pair compared; plus on CUDA the
isotropic tensor against vanilla MMC, and the anisotropic run against the isotropic one (the tensor reaches
the CUDA kernel). Case B, 10⁷ photons per backend, Quadro RTX 6000 (CUDA 11.5, Linux), 128-core CPU:

| | CUDA | OpenCL | CPU | z CUDA−OpenCL | z CUDA−CPU | z OpenCL−CPU |
|---|---|---|---|---|---|---|
| R | 0.699719 | 0.699802 | 0.699550 | −0.24 | 0.48 | 0.72 |
| T | 0.0321391 | 0.0320757 | 0.0321141 | 1.27 | 0.50 | −0.77 |
| ⟨x⟩_T, ⟨y⟩_T (mm) | −2.278, −4.077 | −2.273, −4.086 | −2.276, −4.073 | −0.60, 0.93 | −0.26, −0.43 | 0.35, −1.35 |
| ⟨opl⟩_T (mm) | 149.344 | 149.451 | 149.465 | −1.09 | −1.23 | −0.15 |
| cov_T xx, yy, xy (mm²) | 28.02, 46.82, −4.759 | 28.06, 46.91, −4.766 | 28.02, 46.80, −4.786 | −0.77, −0.99, 0.15 | −0.09, 0.14, 0.55 | 0.68, 1.14, 0.40 |

T(opl) and T(x, y) χ²: p ≥ 0.04 for all pairs. CUDA isotropic tensor vs vanilla: |z| ≤ 1.9.
Time for 10⁷ photons: CUDA 1.9 s, OpenCL 5.9 s (of which 3.9 s JIT compilation), CPU 526 s.
`gpu_vs_cpu.py --compute cuda` (cases A, B, 10⁶ photons) passes as well.

```
python backends_compare.py --case B --nphoton 1e7
```

## To do

- Test 5: 4-layer head (Fig. 5, Pini et al., Neurophotonics 2025).
