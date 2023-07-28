from numba import njit
#fast component of neurotransmitters for AMPA
@njit
def w_update_AMPA(wfe, wfe_decay, spike, w_fast_AMPA):
    alpha_w_AMPA = wfe * spike - wfe_decay * w_fast_AMPA
    return alpha_w_AMPA