##Function of fraction CaMKIV bound to Ca2+. 
#F: fraction of CaMKII subunits bound to Ca+ /CaM.
import numpy as np
@njit
def CaMKIV(Cai, Cao):
    return 1/(1 + np.exp(-(Cai-Cao)))