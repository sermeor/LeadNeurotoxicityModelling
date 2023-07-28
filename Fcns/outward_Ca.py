from numba import njit, jit
##Function of outward calcium rate (uM/ms).
@njit
def outward_Ca(Cai):
  Cai_eq = 0
  c = 0.1 #Rate of calcium pump buffering (ms^-1).
  return + c * (Cai - Cai_eq)