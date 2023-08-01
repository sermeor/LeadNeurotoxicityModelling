import numpy as np
from Fcns.activR5HTtoHA import *
from Fcns.allo_ssri_ki import *
from Fcns.alpha_h import *
from Fcns.alpha_m import *
from Fcns.alpha_n import *
from Fcns.beta_h import *
from Fcns.beta_m import *
from Fcns.beta_n import *
from Fcns.degran_ha_mc import *
from Fcns.fireha import *
from Fcns.fireht import *
from Fcns.FMH_inj import *
from Fcns.H1ha import *
from Fcns.H1ht import *
from Fcns.HTDCin import *
from Fcns.I_AMPA import *
from Fcns.I_ext import *
from Fcns.I_GABA_A import *
from Fcns.I_GABA_B import *
from Fcns.I_K import *
from Fcns.I_L import *
from Fcns.I_Na import *
from Fcns.I_NMDA_Ca import *
from Fcns.I_NMDA_K import *
from Fcns.I_NMDA_Na import *
from Fcns.I_NMDA import *
from Fcns.inhibR5HTto5HT import *
from Fcns.inhibRHAto5HT import *
from Fcns.inhibRHAtoHA import *
from Fcns.inhibsyn5HTto5HT import *
from Fcns.inhibsynHAtoHA import *
from Fcns.inward_Ca import *
from Fcns.k_5ht1ab_rel_ps import *
from Fcns.k_5ht1ab_rel_sp import *
from Fcns.k_fmh_inh import *
from Fcns.k_ssri_reupt import *
from Fcns.mc_activation import *
from Fcns.MgB import *
from Fcns.outward_Ca import *
from Fcns.spike_boolean import *
from Fcns.SSRI_inj import *
from Fcns.TCcatab import *
from Fcns.VAADC import *
from Fcns.VDRR import *
from Fcns.vha_trafficking import *
from Fcns.VHAT import *
from Fcns.VHATg import *
from Fcns.VHATmc import *
from Fcns.VHNMT import *
from Fcns.VHNMTg import *
from Fcns.VHNMTmc import *
from Fcns.vht_trafficking import *
from Fcns.VHTDC import *
from Fcns.VHTDCg import *
from Fcns.VHTDCmc import *
from Fcns.VHTL import *
from Fcns.VHTLg import *
from Fcns.VHTLmc import *
from Fcns.VMAT import *
from Fcns.VMATH import *
from Fcns.VMATHmc import *
from Fcns.VPOOL import *
from Fcns.VSERT import *
from Fcns.VTPH import *
from Fcns.VTRPin import *
from Fcns.VUP2 import *
from Fcns.w_update_AMPA import *
from Fcns.w_update_GABA_A import *
from Fcns.w_update_GABA_B import *
from Fcns.w_update_NMDA import *
from Fcns.I_VGCC import *
from Fcns.alpha_c import *
from Fcns.beta_c import *



def comp_model(t, y, v2, ssri_molecular_weight, SSRI_start_time, SSRI_repeat_time, SSRI_q_inj, 
    fmh_molecular_weight, FMH_start_time, FMH_repeat_time, FMH_q_inj, mc_switch, 
    mc_start_time, btrp0, eht_basal, gstar_5ht_basal, gstar_ha_basal, bht0, 
    vht_basal, vha_basal):

    dy = np.zeros_like(y)
    # Serotonin Terminal Model
    NADP = 26  # NADP concentration in uM.
    NADPH = 330    # NADPH concentration in uM.
    TRPin = 157.6 # addition of tryptophan into blood (uM/h).
    a1 = 5 # Strength of btrp stabilization.  
    a2 = 20    # bound 5ht to autoreceptors produce G5ht*                
    a3 = 200  # T5ht* reverses G5ht* to G5ht.                  
    a4 = 30   # G5ht* produces T5ht*                     
    a5 = 200  # decay of T5ht*.                
    a6 = 36 # rate eht bounding to autoreceptors.                       
    a7 = 20 # rate of 5ht unbonding from autoreceptors.   
    a8 = 1 # 5-HT leakage diffusion from cytosol to extracellular space.
    a9 = 1  # 5-HT leakage diffusion from glia to extracellular space.
    a10 = 1  # catabolism of hiaa
    a11 = 0.01   # UP2 multiplier
    a12 = 2  # removal of trp
    a13 = 1   # removal of pool
    a14 = 40  # eht removal rate.
    a15 = 1 # rate of vht_reserve moving to vht.
    a16 = 1.89 # Factor of release per firing event. 
    a17 = 100  # bound eha to heteroreceptors produce Gha*.
    a18 = 961.094   # Tha* reverses Gha* to Gha.
    a19 = 20  # Gha* produces THA*.
    a20 = 66.2992  # decay of Tha*.
    a21 = 5  # eha binds to heteroreceptors. 
    a22 = 65.6179   # eha dissociates from heteroreceptors.
    g0 = 10 # total g-protein of serotonin autoreceptors.                         
    t0 = 10 # total T protein of serotonin autoreceptors.                         
    b0 = 10 # total serotonin autoreceptors.
    gh0 = 10 # total g-protein of histamine heteroreceptors. 
    th0 =  10  # total T regulary protein of histamine heteroreceptors. 
    bh0 =  10 # total histamine heteroreceptors
    ssri = (y[24]/v2)*1000/(ssri_molecular_weight) # SSRI concentration from compartmental model in uM -> umol/L. 

    # Parameters for SERT membrane, inactivity and pool transport.
    k_ps = 10 * k_5ht1ab_rel_ps(y[11], gstar_5ht_basal)
    k_sp = 10 * k_5ht1ab_rel_sp(y[11], gstar_5ht_basal)
    k_si = 7.5 * k_ssri_reupt(ssri)
    k_is = 0.75

    # Equation parameters.
    # y[0] = btrp
    # y[1] = bh2
    # y[2] = bh4
    # y[3] = trp
    # y[4] = htp
    # y[5] = cht
    # y[6] = vht (needs fitting)
    # y[7] = vht_reserve (needs fitting)
    # y[8] = eht
    # y[9] = hia (needs fitting)
    # y[10] = trppool
    # y[11] = gstar
    # y[12] = tstar
    # y[13] = bound
    # y[14] = glialht (needs fitting)
    # y[15] = Gha*
    # y[16] = Tha*
    # y[17] = bound ha
    # y[18] = SERT_surface_phospho
    # y[19] = SERTs_surface
    # y[20] = SERT_pool
    # y[21] = SERT_inactive

    dy[0] = TRPin - VTRPin(y[0]) - a1 * (y[0] - btrp0)
    dy[1] = inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VDRR(y[1], NADPH, y[2], NADP)
    dy[2] = VDRR(y[1], NADPH, y[2], NADP) - inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) 
    dy[3] = VTRPin(y[0]) - inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VPOOL(y[3], y[10]) - a12 * y[3]
    dy[4] = inhibsyn5HTto5HT(y[11], gstar_5ht_basal) * VTPH(y[3], y[2]) - VAADC(y[4])
    dy[5] = VAADC(y[4]) - VMAT(y[5], y[6]) - VMAT(y[5], y[7]) + VSERT(y[8], y[19], ssri, allo_ssri_ki(ssri)) - TCcatab(y[5]) - a15 * (y[5] - y[8])
    dy[6] = VMAT(y[5], y[6]) - a16 * fireht(y[57], inhibR5HTto5HT(y[11], gstar_5ht_basal) * inhibRHAto5HT(y[15], gstar_ha_basal)) * y[6] + vht_trafficking(y[6], vht_basal)
    dy[7] = VMAT(y[5], y[7]) - a15 * vht_trafficking(y[6], vht_basal)
    dy[8] = a16 * fireht(y[57], inhibR5HTto5HT(y[11], gstar_5ht_basal) * inhibRHAto5HT(y[15], gstar_ha_basal)) * y[6] - VSERT(y[8], y[19], ssri, allo_ssri_ki(ssri)) - a11 * H1ht(y[8], eht_basal) * VUP2(y[8]) - a14 * y[8] + a8 * (y[5] - y[8]) + a9 * (y[14] - y[8])
    dy[9] = TCcatab(y[5]) + TCcatab(y[14]) - a10 * y[9] 
    dy[10] = VPOOL(y[3], y[10]) - a13 * y[10]
    dy[11] = a2 * y[13]**2 * (g0 - y[11]) - a3 * y[12] * y[11]  
    dy[12] = a4 * y[11]**2 * (t0 - y[12]) - a5 * y[12]
    dy[13] = a6 * y[8] * (b0 - y[13]) - a7 * y[13] 
    dy[14] = a11 * H1ht(y[8], eht_basal) * VUP2(y[8]) - TCcatab(y[14]) - a9 * (y[14] - y[8])  
    dy[15] = a17 * y[17]**2 * (gh0 - y[15]) - a18 * y[16] * y[15]
    dy[16] = (a19*y[15]**2 * (th0 - y[16])  - a20 * y[16]) 
    dy[17] = (a21*y[29]*(bh0 - y[17])  - a22*y[17]) 
    dy[18] = k_ps * y[20] - k_sp * y[18]
    dy[19] = dy[18] - k_si * y[19] + k_is * y[21]
    dy[20] = k_sp * y[18]  - k_ps * y[20]
    dy[21] = k_si * y[19]  - k_is * y[21]


    # Escitalopram pharmacokinetics model
    # Rates between comparments (h-1).
    k01 = 0.6
    k10 = 3
    k12 = 9.9
    k21 = 2910
    k13 = 6
    k31 = 0.6

    # Parameters.
    protein_binding = 0.56
    protein_brain_binding = 0.15

    # y[22] = Peritoneum concentration in ug.
    # y[23] = Blood concentration in ug.
    # y[24] = Brain concentration in ug.
    # y[25] = Periphery concentration in ug.

    # Differential equations.
    dy[22] = SSRI_inj(t, SSRI_start_time, SSRI_repeat_time, SSRI_q_inj) - k01*(y[22])
    dy[23] = k01*(y[22]) - (k10 + k12)*(y[23]*(1-protein_binding)) + k21*(y[24]*(1-protein_brain_binding)) - k13*(y[23]*(1-protein_binding)) + k31*(y[25])
    dy[24] = k12*(y[23]*(1-protein_binding)) - k21*(y[24]*(1-protein_brain_binding))
    dy[25] = k13*(y[23]*(1-protein_binding)) - k31*(y[25])


    ## Histamine Terminal Model. 
    b1 = 15  #HA leakage from the cytosol to the extracellular space. 
    b2 = 3.5 #HA release per action potential. 
    b3 = 15 #HA leakage from glia to the extracellular space.
    b4 = 0.05 #HA removal from the extracellular space
    b5 = 0.25  #Strength of stabilization of blood HT near 100μM. 
    b6 = 2.5 #From cHT to HTpool.
    b7 = 1 #From HTpool to cHT. 
    b8 = 1 #Other uses of HT remove HT. 
    b9 = 1  #From gHT to gHTpool. 
    b10 = 1 #From gHTpool to gHT.  
    b11 = 1 #Removal of gHT or use somewhere else. 
    b12 = 10 #Factor of activation of glia histamine production. 
    b13 = 100 #Bound eha to autoreceptors produce G∗. 
    b14 = 961.094 #T∗ facilitates the reversion of G∗ to G. 
    b15 = 20 #G∗ produces T∗. 
    b16 = 66.2992 #decay coefficient of T∗
    b17 = 5  #eha binds to autoreceptors. 
    b18 = 65.6179 #eha dissociates from autoreceptors.
    b19 = 20  #bound e5ht to heteroreceptors to produce G5ht* 
    b20 = 200  #T5ht* reverses G5ht* to G5ht.
    b21 = 30   #G5ht* produces T5ht* 
    b22 =  200  #decay of T5ht*.
    b23 =  36  #rate eht bounding to heteroreceptors.
    b24 =  20  #rate of 5ht unbonding from heteroreceptors. 
    g0HH = 10  #Total gstar for H3 on HA neuron
    t0HH = 10 #Total tstar for H3 autoreceptors on HA neuron
    b0HH = 10  #Total H3 autoreceptors on HA neuron
    g05ht = 10 # total g-protein of serotonin heteroreceptors in histamine varicosity.
    t05ht = 10 # total T protein serotonin heteroreceptors in histamine varicosity.
    b05ht = 10 # total serotonin heteroreceptors in histamine varicosities.
    HTin = 636.5570 # Histidine input to blood histidine uM/h. 

    # y[26]= cha
    # y[27] = vha 
    # y[28] = vha_reserve
    # y[29] = eha
    # y[30] = gha 
    # y[31] = bht 
    # y[32] = cht 
    # y[33] = chtpool 
    # y[34] = gstar 
    # y[35] = tstar 
    # y[36] = bound 
    # y[37] = gstar5ht 
    # y[38] = tstar5ht 
    # y[39] = bound5ht 
    # y[40] = ght
    # y[41] = ghtpool


    dy[26] = inhibsynHAtoHA(y[34], gstar_ha_basal) * y[50] * VHTDC(y[32]) - VMATH(y[26], y[27]) - VHNMT(y[26]) - b1 * (y[26] - y[29]) + VHAT(y[29]) - VMATH(y[26], y[28])
    dy[27] = VMATH(y[26], y[27]) - fireha(t, inhibRHAtoHA(y[34], gstar_ha_basal) * activR5HTtoHA(y[37], gstar_5ht_basal)) * b2 * y[27] + vha_trafficking(y[27], vha_basal)
    dy[28] = VMATH(y[26], y[28]) - vha_trafficking(y[27], vha_basal)

    dy[29] = fireha(t, inhibRHAtoHA(y[34], gstar_ha_basal) * activR5HTtoHA(y[37], gstar_5ht_basal)) * b2 * y[27] - VHAT(y[29]) + b3 * (y[30] - y[29]) + b1 * (y[26] - y[29]) - H1ha(y[29]) * VHATg(y[29]) - b4 * y[29] - mc_activation(t, mc_switch, mc_start_time) * VHATmc(y[29]) + degran_ha_mc(mc_activation(t, mc_switch, mc_start_time)) * y[45]

    dy[30] = H1ha(y[29]) * VHATg(y[29]) - b3 * (y[30] - y[29]) - VHNMTg(y[30]) + (1 + b12 * mc_activation(t, mc_switch, mc_start_time)) * y[50] * VHTDCg(y[40])



    dy[31] = HTin - VHTL(y[31]) - VHTLg(y[31]) - b5 * (y[31] - bht0) - mc_activation(t, mc_switch, mc_start_time) * VHTLmc(y[31])


    dy[32] = VHTL(y[31]) - inhibsynHAtoHA(y[34], gstar_ha_basal) * y[50] * VHTDC(y[32]) - b6 * y[32] + b7 * y[33]
    dy[33] = b6 * y[32] - b7 * y[33] - b8 * y[33]
    dy[34] = b13 * y[36]**2 * (g0HH - y[34]) - b14 * y[35] * y[34]
    dy[35] = b15 * y[34]**2 * (t0HH - y[35]) - b16 * y[35]
    dy[36] = b17 * y[29] * (b0HH - y[36]) - b18 * y[36]
    dy[37] = b19 * y[39]**2 * (g05ht - y[37]) - b20 * y[38] * y[37]
    dy[38] = (b21*y[37]**2*(t05ht - y[38])  - b22*y[38])
    dy[39] = (b23*y[8]*(b05ht - y[39])  - b24*y[39])
    dy[40] = VHTLg(y[31]) - (1 + b12*mc_activation(t, mc_switch, mc_start_time)) * y[50] * VHTDCg(y[40]) - b9 * y[40] + b10 * y[41]
    dy[41] = b9 * y[40] - b10 * y[41] - b11 * y[41]




    # Mast Cell Model
    # y[42] = cht. 
    # y[43] = chtpool.
    # y[44] = cha. 
    # y[45] = vha. 


    c1 = 1 #From cHT to HTpool.
    c2 = 1 #From HTpool to cHT. 
    c3 = 1 #Removal of cHT or use somewhere else. 


    dy[42] = mc_activation(t, mc_switch, mc_start_time) * VHTLmc(y[31]) - y[50] * VHTDCmc(y[42]) - c1 * y[42] + c2 * y[43]

    dy[43] = c1 * y[42] - c2 * y[43] - c3 * y[43]
    dy[44] =  y[50] * VHTDCmc(y[42]) - VMATHmc(y[44], y[45]) - VHNMTmc(y[44]) + mc_activation(t, mc_switch, mc_start_time) * VHATmc(y[29])
    dy[45] = VMATHmc(y[44], y[45]) - degran_ha_mc(mc_activation(t, mc_switch, mc_start_time)) * y[45]


    ## FMH Pharmacokinetics Model
    # Rates between comparments (h-1). 
    k01f = 3.75
    k10f = 1.75
    k12f = 0.1875
    k21f = 3.5
    k13f = 5
    k31f = 1.5

    #Parameters.
    protein_binding_fmh = 0.60
    protein_brain_binding_fmh = 0.15
    fmh = (y[48]/v2)*1000/(fmh_molecular_weight) # Concentration of FMH in uM -> umol/L. 

    #y[46] = Peritoneum concentration in ug.
    #y[47] = Blood concentration in ug.
    #y[48] = Brain concentration in ug.
    #y[49] = Periphery concentration in ug. 
    #y[50] = Ratio of active HTDC in cytosol of histamine, glia and mast cells.

    dy[46] = FMH_inj(t, FMH_start_time, FMH_repeat_time, FMH_q_inj) - k01f * y[46]
    dy[47] = k01f * y[46] - (k10f + k12f) * (y[47] * (1 - protein_binding_fmh)) + k21f * (y[48] * (1 - protein_brain_binding_fmh)) - k13f * (y[47] * (1 - protein_binding_fmh)) + k31f * (y[49])
    dy[48] = k12f * (y[47] * (1 - protein_binding_fmh)) - k21f * (y[48] * (1 - protein_brain_binding_fmh))
    dy[49] = k13f * (y[47] * (1 - protein_binding_fmh)) - k31f * y[49]
    dy[50] = -k_fmh_inh(fmh) * y[50] + HTDCin(y[50])
    
    
    
    ##-------------------- HH MODEL -----------------
    # Parameters of HH model
    C_m = 1.0  # membrane capacitance (uF/cm^2)

    # Define Poisson input parameters
    rate = 1*2*(1/10)  # firing rate (ms-1).
    w_NMDA = 5
    w_GABA_A = 1
    w_AMPA = 5
    w_GABA_B = 1
    # w1 = 0.75  # Excitatory noise synaptic weight
    # w2 = 0.2 #Inhibitory noise synaptic weight

    spike_ex = np.random.poisson(rate, 1)
    spike_inh = np.random.poisson(rate, 1)

    #conductance at baseline, unit: mS
    g_AMPA = w_AMPA * y[58]
    g_GABA_A = w_GABA_A * y[59]
    g_NMDA = w_NMDA * y[60]
    g_GABA_B = w_GABA_B * y[61]

    spike = spike_boolean(y[51])
    w1 = 0.5 #0.5
    w2 = 0.001 #0.001
    w3 = 1.6#0.08 0.4 1.6
    w4 = 0.10 #0.1
    w5 = -1 #1
    w6 = -2 #2

    w7 = 0.05#0.05# Rate of increase of excitatory w_fast_AMPA from neurotransmitter presence (ms-1).
    w8 = 0.01#0.0065 # Rate of decrease of excitatory w_fast_AMPA

    w9 = 0.03#0.04 # increase inhibitory GABA A
    w10 = 0.008#0.005 # decrease

    w11 = 0.01#0.025 # increase excitatory NMDA
    w12 = 0.005#0.010# decrease

    w13 = 0.001#0.001#increase inhibitory GABA B
    w14 = 0.0005#0.0005 # decrease


    # Variable in ODE.
    # y[51] = Vm, membrane potential of neurons.
    # y[52] = m, activation gating variable for the voltage-gated sodium (Na+) channels.
    # y[53] = h, activation gating variable for the voltage-gated potassium (K+) channels.
    # y[54] = n, Inactivation gating variable for the Na+ channels.
    # y[55] = c, activation gating variable of VGCC channels. 
    # y[56] = Cai, internal calcium concentration (uM).
    # y[57] = activity.
    # y[58] = w_fast_AMPA (synaptic weight of activation of AMPA)
    # y[59] = w_fast_GABA_A
    # y[60] = w_slow_NMDA
    # y[61] = w_slow_GABA_B


    #Differential equations
    # dy[51] = (- I_Na(y[52], y[53], y[51]) - I_K(y[54], y[51]) - I_L(y[51]) - I_AMPA(g_AMPA, y[51]) - I_NMDA(g_NMDA, y[51]) - I_GABA_A(g_GABA_A, y[51]) - I_GABA_B(g_GABA_B, y[51]) + I_ext(t))/C_m
    dy[51] = (- I_Na(y[52], y[53], y[51]) - I_K(y[54], y[51]) - I_L(y[51]) - I_VGCC(y[55], y[51]) + w3*g_AMPA + w4*g_NMDA + w5*g_GABA_A + w6*g_GABA_B + I_ext(t))/C_m
    dy[52] = alpha_m(y[51])*(1.0 - y[52]) - beta_m(y[51])*y[52]
    dy[53] = alpha_h(y[51])*(1.0 - y[53]) - beta_h(y[51])*y[53]
    dy[54] = alpha_n(y[51])*(1.0 - y[54]) - beta_n(y[51])*y[54]
    dy[55] = alpha_c(y[51]) * (1 - y[55]) - beta_c(y[51]) * y[55]
    dy[56] = inward_Ca(g_NMDA, y[51], y[55]) - outward_Ca(y[56]) 
    dy[57] = w1 * spike * (1 - y[57]) - w2 * y[57]
    dy[58] = w_update_AMPA(w7, w8, spike_ex, y[58])
    dy[59] = w_update_GABA_A(w9, w10, spike_inh, y[59])
    dy[60] = w_update_NMDA(w11, w12, spike_ex, y[60])
    dy[61] = w_update_GABA_B(w13, w14, spike_inh, y[61])


  
    #Change from ms -> h
    dy[51:] = dy[51:]*3600000 #(h-1)
    
    
    
    return dy