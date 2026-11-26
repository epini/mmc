# CLAUDE.md — anisotropic MMC (fork epini/mmc, branch `aniso`)

Owner: Ernesto Pini (postdoc LENS/INRiM, Complex Photonics). **Reply to him in Italian.**
He is a physicist, expert in anisotropic radiative transfer, not a C/CUDA programmer: you write and
build the code, he decides the physics. Explain build/tooling steps plainly; he is used to MCXlab.

This file is for the fork only: drop it from any pull request to fangq/mmc.

## Goal

Monte Carlo of light transport in the whole human head with **tensor scattering** in white matter,
the tensor derived from DTI tractography (FA, principal directions) mapped to an Optical Fractional
Anisotropy (OFA). Plan: 1) tensor scattering in MMC (CPU ✓, GPU next); 2) head mesh + DTI template
(MNI152 + HCP1065 proposed); 3) FA → OFA → (μs∥, μs⊥) mapping via his generalized diffusion theory;
4) GPU production runs; 5) TD-fNIRS / whole-head / PBM studies. Software paper + application paper.

References (in his claude.ai project, not in this repo):
- Pini et al., Neurophotonics 12, 045003 (2025): human WM, μ̃′s⊥ = 4.4, μ̃′s∥ = 2.6 mm⁻¹, g ≈ 0.97,
  OFA = 0.23 (DTI FA there ≈ 0.3–0.4), μa = 0.043 mm⁻¹ at 820 nm; 4-layer head MC (Table 1, Fig. 5).
- Pini, Giusfredi, Pattelli, Adv. Theory Simul. 9, e70430 (2026): generalized ADE for a full scattering
  tensor + scalar HG g. Code: github.com/epini/ADE (`generalized_ade`, installed in his env).
- Lewis & Fang, Neurophotonics 12, 015009 (2025): Colin27 MMC workflow, CSF properties.

## The model as implemented

μs(ŝ) = ŝᵀ M ŝ / |ŝ|², M symmetric 3×3 per tetrahedron, packed `[Mxx, Myy, Mzz, Mxy, Mxz, Myz]` in 1/mm.
μa, g (HG), n stay per tissue label (`prop`); the `mus` column of `prop` is ignored when a tensor is given.
Exact because MMC keeps the remaining path `r->slen` in optical depth and converts it per element with
the local μs; the direction is constant between events. Same definition as PyXOpto `tensor3f_project`.
Only the symmetric part of M matters (an (ne,3,3) input is symmetrized). Uniaxial fibre:
M = μs⊥ I + (μs∥ − μs⊥) f̂f̂ᵀ; crossing fibres: sum of such tensors.

## What is done (all on branch `aniso`, base fangq/mmc 84c5626)

Source changes (small, keep them small):
- `src/mmc_utils.h/.c`: cfg fields `elemmus` (float*, 6·ne), `isanisomus`, `elemmuslen`, `elemmusfile`;
  JSON key `Mesh.ElemMusFile` (raw float32 file, 6·ne, element-major).
- `src/mmc_mesh.c`: `mesh_loadelemmus()` — outside the `#ifndef MCX_CONTAINER` block (pmmc/mmclab must
  link it); file reading only in the CLI; checks row count vs ne, unit scaling (unitinmm), PSD check.
- `src/mmc_raytrace.c`: `mmc_dir_mus()` / `mmc_step_mus()`; all CPU tracers (P/H/B/S/G) use them.
- `src/mmc_host.c` (`mmc_prep`): rejects GPU, iMMC and MCML mode (`-m 1`) when anisotropic.
  MCML is excluded on purpose: `albedoweight()` runs after the new direction is drawn.
- `src/pmmc.cpp`: `cfg["elemmus"]` as (ne,6) or (ne,3,3).
- `pmmc/pmmc/__init__.py`: on Windows the `sparse_numba` libgomp preload is optional (upstream bug that
  made a locally MSVC-built `_pmmc` look "not compiled").

Validation (`test/aniso/`, see its README):
- isotropic tensor = vanilla MMC bit for bit (CLI, all tracers; pmmc), single thread;
- slab vs generalized ADE: D within 1%, T(t) 1–2%, g = 0 and 0.9, rotated tensor (`slab_vs_ade.py`);
- full tensor (all off-diagonals) vs PyXOpto `AnisotropicLayer`: R, T, maps, optical path within noise
  (`pyxopto_compare/`);
- `test_pmmc.py` (pmmc interface + error cases), `demo_aniso.py` (visual example).

## Next steps (agreed order may change: ask him)

1. **GPU kernel** (`src/mmc_core.cl`, shared by OpenCL and CUDA): in `branchless_badouel_raytet`,
   right after the NODAL_USE_MUA/MUSP block (~line 1154–1163) and before `r->Lmove = ... r->slen / prop.mus`,
   set `prop.mus` from a new `__global float* gelemmus` (6 per element) with the same formula as
   `mmc_dir_mus`. Host: `src/mmc_cl_host.c` (buffer, kernel arg, `-DANISO_MUS` JIT option, like
   NODAL_USE_MUA), `src/mmc_cu_host.cu`. Then lift the GPU rejection in `mmc_host.c` and the 4d test.
   Validate GPU vs CPU statistically (not bitwise) on the slab and PyXOpto cases. He has an NVIDIA GPU.
2. Test 4: two domains with orthogonal fibres (face crossing). Test 5: 4-layer head of the 2025 paper.
3. DTI → mesh pipeline (Python): template, Brain2Mesh/iso2mesh, tensors on element centroids in
   Log-Euclidean metric, rotation from image to mesh frame (check: corpus callosum must be left–right).
4. MMClab (`cfg.elemmus` in `src/mmclab.cpp`) only if he asks: Python is the main interface.

## His machine (Windows 11, NVIDIA GPU)

- Repo: `C:\Users\ernes\Desktop\PostDoc\aniso-mmc\mmc_full\mmc` (origin = epini/mmc, upstream = fangq/mmc).
- Python: conda env **`pyxoptoenv`** (Python 3.11): numpy, scipy, pyopencl, PyXOpto 0.2.3,
  generalized_ade, iso2mesh, pmmc (built from this branch). PostDoc is inside Dropbox.
- Build pmmc: `PostDoc\aniso-mmc\build_pmmc.bat` (git pull + build + tests, log in `build_log.txt`).
  What it works around, keep it in mind for any native build on this PC:
  - pip's `cmake` is a Python launcher that fails inside pip builds → use the Build Tools CMake
    (`...\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin`) and `--no-build-isolation`;
  - VS 18 2026 generator does not find cl.exe → `vcvars64.bat` + `CMAKE_GENERATOR=Ninja`, `CC=CXX=cl`;
  - OpenCL headers/lib come from the conda env: `OpenCL_ROOT=%CONDA_PREFIX%\Library`;
  - the build embeds `mmc_core.cl` with `bash`+`xxd`: `bash` is the WSL stub (no distro) → put
    `C:\Program Files\Git\bin` and `...\Git\usr\bin` first in PATH;
  - Windows SDK was missing (installed 25/11/2026 via VS Installer).
  Equivalent manual sequence, from an Anaconda Prompt with the env active:
  `call vcvars64.bat`, set PATH/CMAKE_GENERATOR/CC/CXX/OpenCL_ROOT as above, then
  `cd pmmc && pip install --no-build-isolation .`
- The command-line `mmc.exe` is not built on his PC (test 3 of `test_pmmc.py` is skipped there).

## Gotchas learned

- Bitwise comparisons only with `OMP_NUM_THREADS=1` (test_pmmc.py sets it itself).
- pmmc and the CLI seed differently: vanilla pmmc vs CLI already differ ~0.5% → compare statistically.
- `.mch` detected photons store the launch weight: apply `exp(-mua * ppath)` yourself.
- MMC launches photons inside the medium (no specular loss); PyXOpto subtracts it and reports it.
- pmmc `mmclab()` default output is a 1 mm voxel grid (`flux` shape (nx, ny, nz, ngates)).
- `iso2mesh.meshabox` downloads tetgen at runtime; `meshgrid5` needs no download.

## Working rules

- Work on branch `aniso`; commit small, explain in the message; never commit build artifacts
  (`bin/`, `src/built/`, `src/lib/`, `*.o`, `*.a`, `src/mmc_core.clh`, `*.mch`, `*.mc2`, `*.jnii`).
- Never force-push or rewrite pushed history; fix with a new commit.
- Keep the diff against upstream minimal and in upstream style, so it can become a PR to fangq/mmc.
- After any change to the transport: rerun the isotropic regression and at least a quick ADE slab check.
