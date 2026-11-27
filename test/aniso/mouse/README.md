# Mouse optogenetics: block test (CLAUDE.md step 2b)

Optical fibre in grey matter near a corpus-callosum slab with tensor scattering (axons along x),
parametric in r = μs∥/μs⊥ at fixed mean μs. **All optical values are placeholders**: they live in
`config.py` (one block per wavelength), together with geometry, fibre, photons and thresholds.

```
export MOUSE_DATA=~/mouse_data                          # outputs in $MOUSE_DATA/out
python block_test.py --check                            # mesh, disk kernel, energy balance, tensor = vanilla
python block_test.py --wl 473 --all-tips                # 3 tips x 5 r, 1e8 photons per case, 10 batches (CUDA)
python block_test.py --wl 590 --all-tips
python block_test.py --wl 473 --tip inside --plot-only  # figures again from the saved npz
```

## Model

- GM block 6×6×6 mm, x, y ∈ [−3, 3], z ∈ [0, 6] = depth (top face = tissue surface, air outside on all
  faces); CC slab z ∈ [1.00, 1.25] mm, normal z. CC tensor uniaxial along x:
  M = diag(μs∥, μs⊥, μs⊥), (μs∥ + 2μs⊥)/3 = 6 μs(GM); r ∈ {1, 0.7, 0.5, 0.3, 0.2}, r = 1 the isotropic
  reference. g, μa, n equal in GM and CC. Every case goes through `elemmus` (GM gets μs·I), and all r use
  the same seeds batch by batch, so the aniso/iso ratios are paired (jackknife over the 10 batches).
- Fibre: 200 µm core, NA 0.37, pointing +z (across the axons), tips 0.3 mm above the CC (z = 0.70),
  in its mid-plane (1.125) and 0.3 mm below it (1.55). MMC has no source that is both a disk and a cone:
  each run is a point `cone` source (half-angle asin(NA/n) = 15.8°, uniform in solid angle) at the centre
  of the tip, and the 3D fluence of each batch is convolved in x, y with the core disk (area fraction of
  each 20 µm cell). The medium is laterally uniform, so a shifted source gives the shifted field: the
  convolution *is* the disk source, up to the block faces 3 mm away. Fibre body not meshed (v1).
- Output: dual-grid MMC (`method="grid"`, 20 µm voxels, 301³). **pmmc's normalized grid output leaves out
  the voxel volume** (invisible with the default 1 mm voxels): fluence per photon = flux · tend / V_vox.
  Checked by energy: Σ μa φ V_vox = the absorbed fraction MMC prints (0.7387), and absorbed + escaped =
  1.00000. (Not fixed in MMC: report upstream?)
- Mesh: structured tetrahedra, nodes on the slab planes. With grid output the mesh only carries the
  geometry: 0.05 mm (the plan's value, 4.75M tetrahedra) and 0.125 mm (442k) gave the same fields and
  metrics (r = 0.2, tip inside, 5·10⁶ photons each: all metrics |z| < 1.6, voxelwise mean z −0.01), and
  0.125 is 4.6× faster: `MESH_FINE = 0.125` in `config.py`.
- Exits (issaveexit, one detector covering the block): power leaving the block below / above / beside the
  slab, for P_down = net power into z > CC bottom (absorbed + escaped there, minus 1 if the tip is there)
  and P_up likewise for z < CC top. Almost every packet exits (absorption only lowers its weight).

## Metrics (per wavelength, tip, r; ± jackknife SE)

Irradiance for 10 mW out of the fibre; volume and x/y/z extents above 0.1, 1 (main) and 10 mW/mm²;
fraction absorbed in the CC; P_down, P_up; profiles along x and y in the CC mid-plane and 0.05 mm under
it. Figures `block_<wl>_<tip>_{fluence,ratio,profiles,metrics}.png`, numbers in `block_<wl>_<tip>.txt`.
Caveat with the placeholder optics (μa 0.1 /mm at 473 nm): the 0.1 mW/mm² isosurface reaches the block
faces, so V(0.1) depends on the block size; V(1) and V(10) do not.
