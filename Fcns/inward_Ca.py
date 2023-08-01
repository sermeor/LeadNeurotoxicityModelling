from numba import njit, jit
from Fcns.I_NMDA_Ca import *
from Fcns.I_VGCC import *
##Function of inward calcium rate (uM/ms).
@njit
def inward_Ca(g_NMDA, Vm, c):
    F = 96485 # Faraday Constant (mA*ms/umol).
    d = 8.4e-6 #Distance of membrane shell where calcium ions enter (cm).
    s = 1000 #conversor umol/(cm^3 * ms) to uM/ms.
    w = 5
    return - s * (I_VGCC(c, Vm))/(2*F*d) + w*g_NMDA









#def inward_Ca(g_NMDA):
  #w = 5
  #return w*g_NMDA
  # F = 96485 # Faraday Constant (mA*ms/umol).
  # d = 8.4e-6 #Distance of membrane shell where calcium ions enter (cm).
  # c = 1000 #conversor umol/(cm^3 * ms) to uM/ms.
  # return - c * I_NMDA_Ca(g_NMDA, Vm)/(2*F*d)


