#!/usr/bin/env python3
"""Step b: nonlinear registration MNI152 T1 (moving) -> Colin27 T1 (fixed), ANTsPy.

    python register.py            # conda env headenv (antspyx, nibabel)

Brain-extracted images on both sides (MNI152_T1_1mm_brain; Colin27 T1 x its brain mask), affine +
SyN (Mattes MI, then CC-free default ANTsPy SyN). Writes to OUT:
  mni2colin_0GenericAffine.mat, mni2colin_1Warp.nii.gz (forward: MNI image -> Colin27 grid)
  mni152_in_colin.nii.gz (QC), hcp_fa_in_colin.nii.gz (scalar FA, QC only: the tensors are
  warped with reorientation in warp_tensors.py)
"""

import os
import shutil
import time

import ants

from common import COLIN_T1, COLIN_MASK, MNI_BRAIN, HCP_FA, OUT


def main():
    os.makedirs(OUT, exist_ok=True)
    fixed = ants.image_read(COLIN_T1) * ants.image_read(COLIN_MASK)
    moving = ants.image_read(MNI_BRAIN)
    t0 = time.time()
    reg = ants.registration(fixed, moving, type_of_transform="SyN", reg_iterations=(100, 70, 50, 20),
                            outprefix=os.path.join(OUT, "tmp_mni2colin_"), verbose=False)
    print("registration %.0f s" % (time.time() - t0))
    names = []
    for k, f in enumerate(reg["fwdtransforms"]):
        dst = os.path.join(OUT, "mni2colin_" + ("1Warp.nii.gz" if f.endswith(".nii.gz") else "0GenericAffine.mat"))
        shutil.copy(f, dst)
        names.append(dst)
    print("forward transforms:", names)
    ants.image_write(reg["warpedmovout"], os.path.join(OUT, "mni152_in_colin.nii.gz"))
    fa = ants.apply_transforms(fixed, ants.image_read(HCP_FA), transformlist=reg["fwdtransforms"], interpolator="linear")
    ants.image_write(fa, os.path.join(OUT, "hcp_fa_in_colin.nii.gz"))
    # similarity before / after, for the record
    lin = ants.registration(fixed, moving, type_of_transform="Affine")
    for name, img in (("affine only", lin["warpedmovout"]), ("affine + SyN", reg["warpedmovout"])):
        print("%-13s normalized correlation with Colin27 T1: %.4f"
              % (name, -ants.image_similarity(fixed, img, metric_type="Correlation")))


if __name__ == "__main__":
    main()
