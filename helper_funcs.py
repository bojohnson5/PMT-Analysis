#!/usr/bin/env python
# -*- coding: utf-8 -*-

from numba import njit

@njit
def _count_crosses(w, thre):
    count = 0
    for j in range(len(w)):
        cross = False
        adj_w = w[j] - thre
        for k in range(len(adj_w)):
            if adj_w[k] > 0 and cross == False:
                cross = True
                count += 1
            if adj_w[k] < 0 and cross == True:
                cross = False
    return count

@njit
def _count_crosses_single(w, thre):
    count = 0
    adj_w = w - thre
    cross = False
    for i in range(len(adj_w)):
        if not cross and adj_w[i] > 0:
            cross = True
            count += 1
        if cross and adj_w[i] < 0:
            cross = False
    return count

@njit
def _count(waveform, thre, wind):
    count = 0
    cross = False
    st, en = wind
    if st < 0:
        st = 0
    if en > len(waveform):
        en = len(waveform)
    for i in range(st, en):
        if not cross and waveform[i] - thre > 0:
            cross = True
            count += 1
        if cross and waveform[i] - thre < 0:
            cross = False
    return count

