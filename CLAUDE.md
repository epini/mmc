# CLAUDE.md — anisotropic MMC (fork epini/mmc, branch `aniso`)

Owner: Ernesto Pini (postdoc LENS/INRiM, Complex Photonics). **Reply to him in Italian.**
He is a physicist, expert in anisotropic radiative transfer, not a C/CUDA programmer: you write and
build the code, he decides the physics. Explain build/tooling steps plainly; he is used to MCXlab.

This file is for the fork only: drop it from any pull request to fangq/mmc.

## Goal

Monte Carlo of light transport in the whole human head with **tensor scattering** in white matter,
the tensor derived from DTI tractography (FA, principal directions) mapped to an Optical Fractional
Anisotropy (OFA). Plan: 1) tensor scattering in MMC (CPU ✓, GPU OpenCL ✓); 2) head mesh + DTI template
(MNI152 + HCP1065 proposed); 3) FA → OFA → (μs∥, μs⊥) mapping via his generalized diffusion theory;
4) GPU production runs; 5) TD-fNIRS / whole-head / PBM studies. Software paper + application paper.

References (PDFs in `docs/aniso/papers/` on his PC, git-ignored: read them there when needed):
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
- `src/mmc_core.cl` (OpenCL + CUDA kernel): `gelemmus` (6 floats/element) is the last argument of
  `mmc_main_loop` → `onephoton` → `branchless_badouel_raytet`; there, after the NODAL_USE_MUA/MUSP
  block, `prop.mus` = same formula as `mmc_dir_mus`. Switch `ANISO_USE_MUS`: JIT `-DMCX_ANISO_MUS`
  on OpenCL, `gelemmus != NULL` on CUDA (no MCXParam change).
  Host: `src/mmc_cl_host.c` (buffer, kernel arg 31, JIT flag), `src/mmc_cu_host.cu` (buffer, launch arg).
- `src/mmc_host.c` (`mmc_prep`): rejects OptiX, iMMC and MCML mode (`-m 1`) when anisotropic.
  MCML is excluded on purpose: `albedoweight()` runs after the new direction is drawn.
- `src/pmmc.cpp`: `cfg["elemmus"]` as (ne,6) or (ne,3,3).
- `pmmc/pmmc/__init__.py`: on Windows the `sparse_numba` libgomp preload is optional (upstream bug that
  made a locally MSVC-built `_pmmc` look "not compiled").

Validation (`test/aniso/`, see its README):
- isotropic tensor = vanilla MMC bit for bit (CLI, all tracers; pmmc), single thread;
- slab vs generalized ADE: D within 1%, T(t) 1–2%, g = 0 and 0.9, rotated tensor (`slab_vs_ade.py`);
- full tensor (all off-diagonals) vs PyXOpto `AnisotropicLayer`: R, T, maps, optical path within noise
  (`pyxopto_compare/`);
- `test_pmmc.py` (pmmc interface + error cases), `demo_aniso.py` (visual example);
- GPU (OpenCL, RTX 3050) vs CPU on the PyXOpto slab, cases A/B, R, T, centroid, covariance,
  T(opl), T(x,y) (`gpu_vs_cpu.py`): agreement within statistics; ~45× faster than CPU (all cores).
- CUDA (`compute="cuda"`) tested 27/11 on maxwell (Quadro RTX 6000, CUDA 11.5): CUDA vs OpenCL vs CPU
  on the same slab, case B, 1e7 photons each (`backends_compare.py`): all |z| < 1.4, χ² pass; isotropic
  tensor = vanilla on CUDA. 1e7 photons: CUDA 1.9 s, OpenCL 5.9 s (3.9 s of it JIT), CPU 526 s (128 cores).
- Test 4: two 5 mm layers, fibres x / y (μs∥ 5, μs⊥ 20), MMC GPU vs PyXOpto (cases C, D in
  `pyxopto_compare/`, `run_pmmc.py` + `compare.py`): R, T, maps, optical path within statistics;
  one tensor split into two element regions = whole slab (`split_check.py`).

## Next steps (agreed order may change: ask him)

1. Test 5: 4-layer head of the 2025 paper (Table 1, Fig. 5), MMC aniso vs his PyXOpto result.
2. **Head model, decided 26/11/2026: Colin27 + HCP1065 DTI template registered onto it**
   (comparable with Lewis & Fang 2025, same Colin27 mesh). Steps:
   a. Data: Colin27 T1 + the Colin27 tissue labels / mesh used by Lewis & Fang (Brain2Mesh, 4 layers:
      scalp+skull, CSF, GM, WM; check MCX/MMC examples and the neurojson brain mesh library);
      FSL `MNI152_T1_1mm` and the HCP1065 templates (`FSL_HCP1065_tensor_1mm`, `_FA_`, `_V1_`),
      available without a full FSL install (FSL conda channel data packages). FSL itself does not run
      on native Windows: prefer Python tools (ANTsPy `antspyx` or DIPY SyN, nibabel).
   b. Nonlinear registration MNI152 T1 -> Colin27 T1.
   c. Warp the tensor field with **reorientation** (PPD or finite strain from the local Jacobian);
      never resample tensor components as scalars. Interpolate tensors in Log-Euclidean metric.
   d. Tensor at the centroid of each Colin27 WM tetrahedron; store per element the diffusion tensor,
      FA and principal direction (keep them: the optical mapping comes later).
   e. Checks with figures: FA over the Colin27 WM/GM labels (FA histogram per tissue, fraction of WM
      elements with FA > 0.2), direction-coloured map (red L-R, green A-P, blue S-I), corpus callosum
      L-R and corticospinal tract S-I, slices in the three planes.
   f. **Stop before the FA -> OFA -> (mus_par, mus_perp) mapping**: that is his decision (calibration
      on template FA vs his OFA 0.23, generalized ADE inversion). Only a placeholder mapping, clearly
      labelled, for a first demo run if useful.
   g. **Done 26/11** (a–e, `test/aniso/headmodel/`). Finding: HCP1065 FA is the FA of the *mean tensor*
      over 1065 subjects, so inter-subject orientation dispersion lowers it (CC median 0.28–0.37,
      max 0.54 in the whole brain; individual CC is 0.6–0.8). Its orientations are fine, its FA is not.
      **Decided 26/11: orientation from HCP1065, anisotropy magnitude from FMRIB58_FA** (FSL, mean of
      58 FA maps, MNI152 space, same registration). Per element keep the HCP1065 eigenvectors and
      rescale the eigenvalues so that FA = FMRIB58 FA: keep the mean diffusivity and the ratio
      (λ2−λ3)/(λ1−λ3) (planar/linear shape of crossings), solve for the spread (clamp eigenvalues > 0). Keep both FAs in the npz.
      First instance for the optics: OFA = FA (his choice), to be confirmed on the new numbers.
   h. **Optical mapping, decided 26/11** (his choices): anisotropy **only in WM elements** (label 4),
      GM/CSF/scalp+skull isotropic; **no FA threshold**; **OFA = FA**. Per WM element the light
      diffusion tensor D_opt has the eigenvectors and the eigenvalue *ratios* of the DTI tensor
      (FMRIB58-rescaled, so its FA = OFA, crossings kept), scaled so that **mean(eig D_opt) = fixed**
      = his measured WM value (D⊥ 1.69e-2, D∥ 2.47e-2 mm²/ps, mean 1.95e-2 mm²/ps; n = 1.39,
      g = 0.97, μa = 0.043 mm⁻¹, 820 nm). Then invert the generalized ADE (`generalized_ade`, scalar HG g)
      to get the scattering tensor M (same eigenvectors: check numerically on a rotated case that D and
      M share the principal frame). The map D-shape -> M-shape is scale-free: build a lookup table on the
      two eigenvalue ratios and invert it (no per-element root finding). Validate: inverting his
      measured D∥, D⊥ must give reduced coefficients close to μ̃′s∥ 2.6, μ̃′s⊥ 4.4 mm⁻¹ (paper values come
      from an MC fit, so expect agreement within their errors, not exactly). **Gotcha:** with a tensor
      given, MMC ignores the `mus` column for ALL elements: non-WM elements need μs·I from their label.
   i. **First head demo done 27/11** (`demo_head_gpu.py`, source ~C3, 5e7 photons/case): aniso vs iso WM
      (same mean D) changes the fluence inside WM by about ±10% (significant), but CW reflectance and
      GM partial path at the scalp by < 1–2% up to 60 mm. Next (agreed 27/11): **time domain**, same
      run: TPSF and per-gate mean partial path in GM/WM vs distance and azimuth, late gates.
   Licences: HCP open-access terms (derived data redistributable under the same terms, acknowledge HCP),
   FSL licence (non-commercial). IIT atlas rejected: its licence forbids adapting/building upon the data.
   MIDA (single subject, DTI + 116 structures) is the later "real subject" option.
3. ~~CUDA build test~~ done 27/11 on maxwell (see "What is done" and the maxwell section).
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

## maxwell (Linux workstation: Ubuntu 22.04, 128 cores, Quadro RTX 6000 24 GB, CUDA 11.5)

- Repo `~/aniso-mmc` (same origin). Disk ~90% full: keep large outputs small, delete `.mch`/`.mc2`.
- Python: venv **`~/venvs/aniso`** (system Python 3.10, no conda), `source ~/venvs/aniso/bin/activate`.
  Made with: `python3 -m venv ~/venvs/aniso; pip install numpy scipy matplotlib pyopencl iso2mesh
  nibabel pmcx ninja`; PyXOpto is not on PyPI: `pip install` of a copy of `~/Desktop/PyXOpto` (his
  epini/pyxopto fork with local uncommitted changes; copy it first, the in-tree build writes there);
  `pip install "git+https://github.com/epini/ADE#subdirectory=python"` (generalized_ade).
  The older venv `~/venvs/pyxopto` (Jan 2026, PyXOpto only) was left untouched.
- pmmc with CUDA + OpenCL (~1 min):
  `git submodule update --init src/pybind11; cd pmmc;
  CMAKE_ARGS=$'-DBUILD_CUDA=ON\t-DCUDA_ARCHS=75\t-DBUILD_MEX=OFF' pip install --no-build-isolation .`
  (tab-separated CMAKE_ARGS; sm_75 = RTX 6000; the module is the top-level `_pmmc`).
- CLI: `cd src && make cuda -j16` → `bin/mmc` with `-c opencl|cuda|sse` (all CUDA archs, 55 MB).
- OpenCL lists only the NVIDIA (no iGPU), still `gpuid=1`. pmmc CPU uses all 128 cores.
- Builds leave untracked `bin/ lib/ src/built/ src/lib/ src/*.o src/zmat/**/*.o src/mmc_core.clh
  pmmc/build/ pmmc/*.egg-info`: listed in `.git/info/exclude` on maxwell, never commit them.
- Status 27/11: test_pmmc.py all PASS (incl. pmmc vs CLI), CLI Test 1 bitwise for P/H/B/S/G,
  slab_vs_ade within statistics (g = 0.9 at 3e6 photons: ±1.5% seed-to-seed spread on D),
  gpu_vs_cpu.py A/B on opencl and cuda PASS, backends_compare.py PASS.

## Gotchas learned

- Bitwise comparisons only with `OMP_NUM_THREADS=1` (test_pmmc.py sets it itself).
- pmmc and the CLI seed differently: vanilla pmmc vs CLI already differ ~0.5% → compare statistically.
- pmmc on the GPU: `compute="opencl", gpuid=1` (device 1 = the NVIDIA; the Intel iGPU is also listed).
- pmmc `res["detp"]` is a dict (`detid, nscat, ppath, p, v, w0, ...`), not the raw array.
- Upstream bug (not ours, not fixed): `mmc_cl_host.c` respin>1 resets the seed buffer on kernel arg 15
  (gdetphoton) instead of 17 (gseed).
- `.mch` detected photons store the launch weight: apply `exp(-mua * ppath)` yourself.
- PyXOpto roulette: packets below weight 1e-4 survive with p = 0.01 (×100): with absorption its
  optical-path tail beyond exp(-mua L) < 1e-4 is ~100× noisier; compare only above that weight.
- PyXOpto runs on his NVIDIA through pyopencl (`mc.clinfo.gpus()[0]`), 1e8 photons in ~1 min.
- MMC launches photons inside the medium (no specular loss); PyXOpto subtracts it and reports it.
- pmmc `mmclab()` default output is a 1 mm voxel grid (`flux` shape (nx, ny, nz, ngates)).
- `iso2mesh.meshabox` downloads tetgen at runtime; `meshgrid5` needs no download.

## Working rules

- Plots: colour maps (imshow, pcolormesh, contourf) always in **viridis**; pick contour/line colours that
  stay visible on it (e.g. red, white, black), not cyan.
  Exception (his request 27/11): **signed difference maps** (e.g. aniso − iso) use a diverging red–blue map,
  `RdBu_r` (blue negative, white zero, red positive), symmetric limits; contours on it in black/grey.
- Work on branch `aniso`; commit small, explain in the message; never commit build artifacts
  (`bin/`, `src/built/`, `src/lib/`, `*.o`, `*.a`, `src/mmc_core.clh`, `*.mch`, `*.mc2`, `*.jnii`).
- Never force-push or rewrite pushed history; fix with a new commit.
- Keep the diff against upstream minimal and in upstream style, so it can become a PR to fangq/mmc.
- After any change to the transport: rerun the isotropic regression and at least a quick ADE slab check.
