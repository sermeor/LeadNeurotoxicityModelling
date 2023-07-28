from numba import njit
# fast component of neurotransmitters for GABA A
@njit
def w_update_GABA_A(wfi, wfi_decay, spike, w_fast_GABA_A):
    alpha_w_GABA_A = wfi * spike - wfi_decay * w_fast_GABA_A
    return alpha_w_GABA_A