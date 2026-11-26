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
The CUDA path (`mmc_cu_host.cu`) is written but not yet compiled or tested.

## To do

- Test 4: two half-spaces with orthogonal fibre directions (face crossing).
- Test 5: 4-layer head (Fig. 5, Pini et al., Neurophotonics 2025).
