from numba import njit, jit
#define whether neuron is firing (1) and not firing (0) at a certain time point
@njit
def spike_boolean(Vm):
  Vth = 0
  return 1 if Vm >= Vth else 0