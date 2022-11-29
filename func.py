import math

import numpy as np

def blackbody_emit(c1_prime, c2_prime, spec, temp_K):
    return math.pi * c1_prime / ((spec**5) * (np.exp(c2_prime/(temp_K*spec)) - 1))


# def calc_sim_signal(bb_emit, emissivity, absorbance):
#     return sum(bb_emit * emissivity * absorbance)

    