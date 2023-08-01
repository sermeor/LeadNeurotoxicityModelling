##Function of rate constant increase of c.
import numpy as np
from numba import njit, jit
@njit
def beta_c(V):
    return 0.125 * np.exp(-(V + 50.0) / 80.0)