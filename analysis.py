#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot

from scipy.optimize import curve_fit
from fit_funcs import gaussian
from numba import jit

def view_waveform(fi, num):
    """
    View a waveform from ROOT file

    Parameters
    ----------
        fi : str
            ROOT file with a waveformTree TTree
        num : int
            Waveform number to view
    """
    with uproot.open(fi + ':waveformTree') as tree:
        waveforms = tree['waveform'].array(library='numpy')
        baselines = tree['baseline'].array(library='numpy')
        polarity = tree['polarity'].array(library='numpy')
        w = ak.to_numpy((waveforms - baselines) * polarity)
        x = np.arange(0, len(waveforms[num]) * 4, 4)
        plt.plot(x, w[num])
        plt.xlabel('Time [ns]', loc='right')
        plt.ylabel('ADC', loc='top')
        plt.title('Run ' + fi[0:2] + ' Waveform ' + str(num))
        plt.show()


def max_amplitudes(fi, n_bins=150, view_wind=(0, 2000)):
    """
    Histogram the maximum amplitudes from waveforms

    Parameters
    ----------
        fi : str
            ROOT file with a waveformTree TTree
        n_bins : int, optional
            Number of histogram bins
        view_wind : tuple of ints, optional
            Viewing window for histogram
    """
    with uproot.open(fi + ':waveformTree') as tree:
        waveforms = tree['waveform'].array()
        baselines = tree['baseline'].array()
        polarity = tree['polarity'].array()
        w = ak.to_numpy((waveforms - baselines) * polarity)
        hist = np.apply_along_axis(np.amax, 1, w)
        plt.hist(hist[(hist > view_wind[0]) * (hist < view_wind[1])], 
                 bins=n_bins, histtype='step')
        plt.xlabel('ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + fi[0:2] + ' Max. Amps.')
        plt.show()


def view_spectrum(fi, wind, n_bins=150, view_wind=(-200, 4000)):
    """
    Histogram the integrated waveforms

    Parameters
    ----------
        fi : str
            ROOT file with a waveformTree TTree
        wind : tuple of ints
            Where to stop and start the integration window, as an indicies not times
        n_bins : int, optional
            Number of histogram bins
        view_wind : tuple of ints, optional
            Viewing window for histogram
    """
    with uproot.open(fi + ':waveformTree') as tree:
        waveforms = tree['waveform'].array()
        baselines = tree['baseline'].array()
        polarity = tree['polarity'].array()
        w = ak.to_numpy((waveforms - baselines) * polarity)
        spec = np.apply_along_axis(np.sum, 1, w[:, wind[0]:wind[1]+1])
        plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])], 
                 bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0])
                 + '-' + str(wind[1]))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + fi[0:2] + ' Spectrum')
        plt.legend()
        plt.show()


def view_multi_spec(fi, winds, n_bins=150, view_wind=(-200, 4000)):
    """
    Plot multiple spectrum histograms of the integrated waveforms

    Parameters
    ----------
        fi : str
            ROOT file with a waveformTree TTree
        winds : array of tuple of ints
            Where to stop and start the integration window, as indicies not times
        n_bins : int, optional
            Number of histogram bins
        view_wind : tuple of ints, optional
            Viewing window for histogram
    """
    with uproot.open(fi + ':waveformTree') as tree:
        waveforms = tree['waveform'].array()
        baselines = tree['baseline'].array()
        polarity = tree['polarity'].array()
        w = ak.to_numpy((waveforms - baselines) * polarity)
        for wind in winds:
            spec = np.apply_along_axis(np.sum, 1, w[:, wind[0]:wind[1]+1])
            plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])], 
                     bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0])
                     + '-' + str(wind[1]))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + fi[0:2] + ' Spectrum')
        plt.legend()
        plt.show()


def fit_spectrum(fi, wind, funcs, bounds, p0s, n_bins=150, view_wind=(-200, 4000)):
    """
    Fit a spectrum with a given function(s) over a specified range(s)

    Parameters
    ----------
        fi : str
            ROOT file with a waveformTree TTree
        wind : tuple of ints
            Where to stop and start the integration window, as indicies not times
        funcs : list of functions
            Functions that will be fitted over specified range
        bounds : list of tuples
            Range where the associated function should be fitted to
        p0s: list of lists
            Initial paramter guesses for fits
        n_bins : int, optional
            Number of histogram bins
        view_wind : tuple of ints, optional
            Viewing window for histogram
    """
    with uproot.open(fi + ':waveformTree') as tree:
        waveforms = tree['waveform'].array()
        baselines = tree['baseline'].array()
        polarity = tree['polarity'].array()
        w = ak.to_numpy((waveforms - baselines) * polarity)
        spec = np.apply_along_axis(np.sum, 1, w[:, wind[0]:wind[1]+1])
        spec = spec[(spec > view_wind[0]) * (spec < view_wind[1])]
        hist, bin_b = np.histogram(spec, bins=n_bins)
        bin_w = np.diff(bin_b)
        bin_c = bin_b[:-1] + bin_w / 2
        plt.bar(bin_c, hist, width=bin_w, label='hist')

        fit_nu = 1
        for func, bound, p0 in zip(funcs, bounds, p0s):
            x_fit = bin_c[(bin_c > bound[0]) * (bin_c < bound[1])]
            y_fit = hist[(bin_c > bound[0]) * (bin_c < bound[1])]
            popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0)
            x = np.linspace(view_wind[0], view_wind[1], 1000)
            y = func(x, *popt)
            label = ('{:.2e} ' * len(popt)).format(*popt)
            plt.plot(x, y, 'r--', label='fit ' + str(fit_nu) + ' ' + label)
            fit_nu += 1
        plt.legend()
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + fi[0:2] + ' Spectrum and Fits')
        plt.show()



if __name__ == '__main__':
    #  max_amplitudes('12.root', 150)
    #  view_waveform('12.root', 3)
    #  view_multi_spec('12.root', [(160, 170), (150, 180), (162, 175)], n_bins=100)
    fit_spectrum('12.root', (160, 170), [gaussian, gaussian], 
                 [(-200, 200), (400, 1000)], [[2000, 0, 50], [200, 500, 100]])

