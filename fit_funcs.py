#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma

def deap_expo(x, p0, p1):
    """
    Exponential function from Eq. (5) of 'In-situ characterization of the
    Hamamatsu R5912-HQE photomultiplier tubes used in the DEAP-3600 experiment'

    Parameters
    ----------
        p0 : float
            Amplitude
        p1 : float
            Parameter l from Eq. (5)
    """
    return p0 * p1 * np.exp(-x * p1)

def deap_ped(x, p0, p1, p2):
    """
    Pedastal function from Eq. (6) of 'In-situ characterization of the
    Hamamatsu R5912-HQE photomultiplier tubes used in the DEAP-3600 experiment'

    Parameters
    ----------
        p0 : float
            Amplitude
        p1 : float
            Parameter mu_ped from Eq. (6)
        p2 : float
            Parameter sigma_ped from Eq. (6)
    """
    return p0 / (np.sqrt(2 * np.pi) * p2) * np.exp(-(x - p1)**2 / (2 * p2**2))


def deap_gamma(x, p0, p1, p2):
    """
    Gamma function from Eq. (4) of 'In-situ characterization of the
    Hamamatsu R5912-HQE photomultiplier tubes used in the DEAP-3600 experiment'

    Parameters
    ----------
        p0: float
            Amplitude
        p1 : float
            Parameter mu from Eq. (4)
        p2 : float
            Parameter b from Eq. (4)
    """
    return p0 / (p1 * p2 * gamma(1 / p2)) * \
        np.power(x / (p1 * p2), 1 / p2 - 1) * np.exp(-x / (p1 * p2))

def spe(x, p0, p1, p2, p3, p4, p5, p6, p7):
    """
    SPE fit corresponding to Eq. (5) of the paper
    """
    main_gamma = deap_gamma(x, p0, p1, p2)
    sec_gamma = deap_gamma(x, p4, p1 * p4, p2 * p5)
    expo = deap_expo(x, p6, p7)
    return main_gamma + sec_gamma + expo * (x <= p4)

