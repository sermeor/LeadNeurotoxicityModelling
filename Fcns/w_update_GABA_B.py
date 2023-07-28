from numba import njit
# slow component of neurotransmitters for GABA B
@njit
def w_update_GABA_B(wsi, wsi_decay, spike, w_slow_GABA_B):
    alpha_w_GABA_B = wsi * spike - wsi_decay * w_slow_GABA_B
    return alpha_w_GABA_B