# Nota tecnica — scattering tensoriale in MMC

Riferimento codice: fork `epini/mmc`, branch `aniso`, base `fangq/mmc` commit `84c5626` (16/11/2026).
I numeri di riga si riferiscono a quel commit.

## 1. Modello da implementare

Per ogni elemento tetraedrico *e* un tensore simmetrico 3×3 **M**ₑ (mm⁻¹):

    μs(ŝ) = ŝᵀ Mₑ ŝ
          = Mxx sx² + Myy sy² + Mzz sz² + 2(Mxy sx sy + Mxz sx sz + Myz sy sz)

- Caso uniassiale (fibra con direzione f̂): **M** = μs⊥ **I** + (μs∥ − μs⊥) f̂ f̂ᵀ.
- Fibre incrociate: **M** = Σₖ wₖ **M**ₖ. La somma di forme quadratiche è ancora una forma quadratica,
  quindi basta un solo tensore per elemento.
- g, μa e n restano scalari, presi dalla tabella dei mezzi come oggi. La phase function HG non cambia.

## 2. Perché la modifica è esatta e locale

MMC conserva il cammino libero residuo `r->slen` in unità adimensionali (profondità ottica, cioè numero di mfp).
A ogni tratto dentro un elemento lo converte in lunghezza fisica con il μs locale:

    dlen     = r->slen / mus           // distanza fisica fino al prossimo scattering
    r->slen -= r->Lmove * mus          // profondità ottica consumata nel tratto

Tra due eventi (scattering, riflessione o attraversamento di faccia) la direzione ŝ è costante,
quindi dentro un tratto μs(ŝ) è costante. Sostituire `mus` con ŝᵀMₑŝ, valutato a inizio tratto,
dà un campionamento **esatto** della distanza libera con tasso direzionale:
- l'esponenziale è senza memoria, per cui la profondità ottica residua resta Exp(1)
  anche dopo una riflessione di Fresnel o il passaggio in un elemento con tensore diverso;
- è lo stesso meccanismo che MMC usa già per i mezzi eterogenei.

Non serve toccare: `mc_next_scatter` (HG con g scalare), le condizioni al contorno, Fresnel, il ray-tracing.

## 3. Punti di intervento

### 3a. Dati e input (host)

| File | Cosa |
|---|---|
| `src/mmc_utils.h` (struct `mcconfig`) | nuovi campi `float* elemmus` (ne×6: xx,yy,zz,xy,xz,yz) e flag `int isanisomus` |
| `src/mmc_utils.c` ~299, ~473 | init a NULL e free, come `nodemua`/`nodemusp` |
| `src/mmc_utils.c` (parser JSON) | chiave `Domain.ElemMus` (array ne×6), o file binario esterno |
| `src/mmclab.cpp` ~1633 | `cfg.elemmus` per MATLAB/Octave, copiando il blocco `nodemua` |
| `pmmc/` | stesso campo per Python (pmmc non supporta nemmeno `nodemua`: va aggiunto) |

Per gli elementi isotropi (scalpo, cranio, CSF, GM) l'host scrive μs·**I** prendendo μs dalla tabella dei mezzi.
Il kernel resta così senza rami per tipo di tessuto. Memoria: 24 B/elemento, circa 24 MB per 10⁶ elementi.

### 3b. Trasporto CPU — `src/mmc_raytrace.c`

Ovunque si legga `mus = (cfg->mcmethod == mmMCX) ? prop->mus : (prop->mua + prop->mus)`,
si usa una funzione inline:

```c
static inline float aniso_mus(const float* M, const float3* v) {
    return M[0]*v->x*v->x + M[1]*v->y*v->y + M[2]*v->z*v->z
         + 2.f*(M[3]*v->x*v->y + M[4]*v->x*v->z + M[5]*v->y*v->z);
}
```

| Riga | Funzione | Nota |
|---|---|---|
| 253 | `roi_step` | modalità implicita (iMMC); per ora si esclude: aniso + iMMC = errore |
| 315, 377, 434 | `plucker_raytet` | calcolo di `mus`, di `dlen` e consumo di `slen` |
| 656, 667, 710 | `havel_raytet` | idem |
| 923, 925, 954 | `badouel_raytet` | idem |
| 1557, 1566, 1606 | `branchless_badouel_raytet` (CPU) | idem: stesso algoritmo del kernel GPU, il default delle build GPU è `rtBLBadouelGrid` |
| 2921 | `albedoweight` | peso di albedo μs/(μa+μs) → μs(ŝ)/(μa+μs(ŝ)) (solo `mmMCML`) |
| 440, 716, 960, 1612 | output `otWP` (replay) | `Lmove*mus/slen0` → usare μs(ŝ) |

Con `mmMCX` (default) `mus` = μs; con `mmMCML` vale μa+μs, e si diventa μa + μs(ŝ).

### 3c. Kernel GPU — `src/mmc_core.cl` (OpenCL, anche CUDA)

Modello già pronto da copiare: l'override per-nodo `NODAL_USE_MUA/MUSP` alle righe 1154–1163
di `branchless_badouel_raytet` (riga 1061). Lì `prop.mus` viene sostituito per elemento prima del calcolo di `r->Lmove`.
Si aggiunge un blocco analogo:

```c
if (ANISO_MUS) {
    __global float* M = gelemmus + (r->eid - 1) * 6;
    prop.mus = aniso_mus(M, r->vec);
}
```

subito dopo il blocco nodale e prima di riga 1167 (`r->Lmove = ... r->slen / prop.mus`).
Il resto (riga 1217 `otWP`, riga 1225 consumo `slen`) usa già `prop.mus`, quindi eredita la modifica.
Da fare anche il ricalcolo nel ramo iMMC a riga 1179, oppure escluderlo come su CPU.

Host:
- `src/mmc_cl_host.c` ~110, ~246, ~412: nuovo buffer `gelemmus_cl` (`clCreateBuffer`, ne×6 float),
  nuovo argomento del kernel, `#define ANISO_MUS` nelle opzioni di build JIT, come `NODAL_USE_MUA`.
- `src/mmc_cu_host.cu`: stesso schema per CUDA, flag runtime in `gcfg`.
- OptiX (`mmc_optix_*`): fuori scopo per ora.

## 4. Coerenza dei sistemi di riferimento (rischio principale lato dati)

Il tensore va espresso nel sistema di coordinate della mesh (mm, stessi assi dei nodi).
I tensori DTI (FSL `dtifit`, template HCP1065) sono nello spazio immagine: possono esserci flip
(convenzione radiologica FSL) e la matrice affine/qform va applicata con la sola parte di rotazione:
**M**ₘₑₛₕ = R **M**ᵥₒₓ Rᵀ. Test obbligatorio: corpo calloso, dove f̂ deve essere sinistra–destra.

## 5. Piano di test (in `docs/aniso/tests/`)

1. **Regressione**: `elemmus` = μs·I su tutti gli elementi; `examples/validation/cube2.json`
   deve riprodurre MMC originale a parità di seed, a meno dell'arrotondamento float
   (ŝᵀ(μs·I)ŝ = μs·|ŝ|², con |ŝ|² ≈ 1 entro 1e-7). Riferimento: assorbimento 27.21835% con 10⁶ fotoni su CPU.
2. **Slab anisotropo**: confronto di R(x,y,t), T(t) e MSD con il codice ADE (`epini/ADE`) e con PyXOpto,
   per g = 0 e g = 0.9, tensore diagonale.
3. **Invarianza per rotazione**: stesso slab con tensore ruotato di 30° attorno a z; i profili devono ruotare di 30°.
4. **Interfacce**: due semispazi con f̂ ortogonali, per controllare l'attraversamento delle facce.
5. **Testa a 4 strati**: riproduzione di Fig. 5 di Pini et al., Neurophotonics 2025 (PyXOpto, 10¹¹ traiettorie).
6. **GPU = CPU** entro il rumore statistico su 1 e 2.

## 6. Ordine di lavoro proposto

1. Struttura dati e lettura JSON + implementazione CPU (`branchless_badouel_raytet`, poi gli altri tracer) → test 1–4 su CPU.
2. Kernel OpenCL + host → test 6.
3. `mmclab` e `pmmc` → pipeline DTI → mesh.
4. CUDA host.
5. Contatto con Fang per il merge upstream (PR da `epini/mmc:aniso`).
