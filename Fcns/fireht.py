# Function of serotonin neuron firing.
# Commented functions are different firing paradigms. 
# UNITS of f() in events/h, time variables in seconds. 

import numpy as np

def fireht(activity, i_factor):
  # UNITS of f() in events/h, time variables in seconds.
    return (activity + 1)*i_factor