##Function of rate constant increase of c.
import numpy as np
from numba import njit, jit
@njit
def alpha_c(V):
    return 0.01 * (V + 20.0) / (1.0 - np.exp(-(V + 20.0) / 2.5))
