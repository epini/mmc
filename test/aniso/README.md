# Test anisotropic MMC (branch `aniso`)

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

## To do

- Test 4: two half-spaces with orthogonal fibre directions (face crossing).
- Test 5: 4-layer head (Fig. 5, Pini et al., Neurophotonics 2025).
- Test 6: GPU = CPU (after the OpenCL kernel).
