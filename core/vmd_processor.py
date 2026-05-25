import numpy as np
from vmdpy import VMD
from config import VMD_ALPHA, VMD_TAU, VMD_DC, VMD_INIT, VMD_TOL


def apply_vmd(signal, K=3):
    signal = np.asarray(signal, dtype=float).ravel()

    modes, _, _ = VMD(
        signal,
        VMD_ALPHA,
        VMD_TAU,
        K,
        VMD_DC,
        VMD_INIT,
        VMD_TOL
    )

    return modes