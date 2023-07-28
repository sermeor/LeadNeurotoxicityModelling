from numba import njit
# slow component of neurotransmitters for NMDA
@njit
def w_update_NMDA(wse, wse_decay, spike, w_slow_NMDA):
    alpha_w_NMDA = wse * spike - wse_decay * w_slow_NMDA
    return alpha_w_NMDA