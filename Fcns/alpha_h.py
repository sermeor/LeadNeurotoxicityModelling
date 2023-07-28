##Function of rate constant increase of h.
import numpy as np
from numba import njit, jit
@njit
def alpha_h(Vm):
   return 0.07 * np.exp( - (Vm + 65.0)/20.0)