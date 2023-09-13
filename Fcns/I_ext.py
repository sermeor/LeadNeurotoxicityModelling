from numba import njit, jit
##Function of external current (mA/cm^2)
@njit

def I_ext(t):
  time = t * 1
  if time>= 5000/3600000 and time<=7000/3600000: #stimulation period: 2000ms
    I=6 #350uA emitted by the probe, but bcs of surface area, cell should experience more current than that
  else:
    I = 0
  return I

#5 higher than 10,8