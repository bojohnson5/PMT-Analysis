#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
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


@njit
def _sum(waveforms):
    """
    Sum up a specific window of a 2d array of waveforms for each waveform

    Parameters
    ----------
        waveforms : 2d array
            This array needs to be already filtered to the range to sum over

    Returns
    -------
        spec : array
            The sum over the window for each waveform in waveforms
    """
    spec = np.zeros(len(waveforms))
    for i in range(len(waveforms)):
        total = 0
        for j in range(len(waveforms[i])):
            total += waveforms[i][j]
        spec[i] += total
    return spec


@njit
def _find_max(waveforms):
    """
    Find the maximum of a 2d array of waveforms for each waveform

    Parameters
    ----------
        waveforms : 2d array
            This array needs to be already filtered to the range to sum over

    Returns
    -------
        max_w : array
            The maximum value for each waveform in waveforms
    """
    max_w = np.zeros(len(waveforms))
    for i in range(len(waveforms)):
        max_temp = -np.inf
        for j in range(len(waveforms[i])):
            if waveforms[i][j] > max_temp:
                max_temp = waveforms[i][j]
        max_w[i] = max_temp
    return max_w


@njit
def _count_multi_crosses(w, thres):
    """
    Count the crossings of a certain threshold for multiple
    waveforms

    Parameters
    ----------
        w : 2d array
            Waveforms to find the crossings of
        thres : array of ints or floats
            List of thresholds to find the crossings above

    Returns
    -------
        counts : array of ints
            Number of crossings above threshold in 2d
            array of waveforms
    """
    counts = np.zeros_like(thres)
    for i in range(len(counts)):
        counts[i] += _count_crosses(w, thres[i])
    return counts


@njit
def _pulsing(waveforms, spe_thre, pulse_thre, pre_wind, late_wind,
             after_wind):
    """
    Calculate the pre-, late-, and after-pulsing percentages

    Parameters
    ----------
        waveforms: 2d array of ints or floats
            The waveforms to scan through
        spe_thre : int or float
            Threhold value for single photoelectrons
        pulse_thre : int or float
            Threshold value for pre- and post-pulses
        pre_wind : tuple of ints
            Start and end of window to count pre-pulses, times in ns
        late_wind : tuple of ints
            Start and end of window to count late-pulses, times in ns
        after_wind : tuple of ints
            Start and end of window to count after-pulses, times in ns
    """
    spe = 0
    pre = 0
    late = 0
    after = 0
    for i in range(len(waveforms)):
        cross = False
        for j in range(len(waveforms[i])):
            if not cross and waveforms[i][j] - spe_thre > 0:
                cross = True
                spe += 1
                pre_s = j - pre_wind[1] // 4
                pre_e = j - pre_wind[0] // 4
                late_s = j + late_wind[0] // 4
                late_e = j + late_wind[1] // 4
                after_s = j + after_wind[0] // 4
                after_e = j + after_wind[1] // 4
                pre += _count(waveforms[i], pulse_thre, (pre_s, pre_e))
                late += _count(waveforms[i], pulse_thre, (late_s, late_e))
                after += _count(waveforms[i], pulse_thre, (after_s, after_e))
    return spe, pre, late, after
