#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def gaussian(x, p0, p1, p2):
    return p0 * np.exp(-0.5 * ((x - p1) / p2)**2)

