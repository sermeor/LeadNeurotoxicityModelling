#HH Model
#Function of rate constant increase of m.
import numpy as np
from numba import njit, jit
@njit
def alpha_m(Vm):
   return 0.1*(Vm + 40.0)/(1.0 - np.exp( - (Vm + 40.0)/10.0))