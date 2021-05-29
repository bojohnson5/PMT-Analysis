#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot

from scipy.optimize import curve_fit
from fit_funcs import gaussian
from numba import jit

class Rooter:
    """
    A class to do the analysis of a ROOT file and associated waveforms

    Attributes
    ----------
        fi : str
            File name used for the analysis
        w : array of arrays
            Contains the waveforms as a 2d array as numpy arrays
    """
    def __init__(self, fi):
        with uproot.open(fi + ':waveformTree') as tree:
            waveforms = tree['waveform'].array()
            baselines = tree['baseline'].array()
            polarity = tree['polarity'].array()
            self.w = ak.to_numpy((waveforms - baselines) * polarity)
            self.fi = fi


    def view_waveform(self, num):
        """
        View a waveform from ROOT file

        Parameters
        ----------
            num : int
                Waveform number to view
        """
        x = np.arange(0, len(self.w[num]) * 4, 4)
        plt.plot(x, self.w[num])
        plt.xlabel('Time [ns]', loc='right')
        plt.ylabel('ADC', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Waveform ' + str(num))
        plt.show()


    def max_amplitudes(self, n_bins=150, view_wind=(0, 2000)):
        """
        Histogram the maximum amplitudes from waveforms

        Parameters
        ----------
            n_bins : int, optional
                Number of histogram bins
            view_wind : tuple of ints, optional
                Viewing window for histogram
        """
        hist = np.apply_along_axis(np.amax, 1, self.w)
        plt.hist(hist[(hist > view_wind[0]) * (hist < view_wind[1])], 
                 bins=n_bins, histtype='step')
        plt.xlabel('ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Max. Amps.')
        plt.show()


    def view_spectrum(self, wind, n_bins=150, view_wind=(-200, 4000)):
        """
        Histogram the integrated waveforms

        Parameters
        ----------
            wind : tuple of ints
                Where to stop and start the integration window, as an indicies not times
            n_bins : int, optional
                Number of histogram bins
            view_wind : tuple of ints, optional
                Viewing window for histogram
        """
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1]+1])
        plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])], 
                 bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0])
                 + '-' + str(wind[1]))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum')
        plt.legend()
        plt.show()


    def view_multi_spec(self, winds, n_bins=150, view_wind=(-200, 4000)):
        """
        Plot multiple spectrum histograms of the integrated waveforms

        Parameters
        ----------
            winds : array of tuple of ints
                Where to stop and start the integration window, as indicies not times
            n_bins : int, optional
                Number of histogram bins
            view_wind : tuple of ints, optional
                Viewing window for histogram
        """
        for wind in winds:
            spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1]+1])
            plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])], 
                     bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0])
                     + '-' + str(wind[1]))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum')
        plt.legend()
        plt.show()


    def fit_spectrum(self, wind, funcs, bounds, p0s, n_bins=150, view_wind=(-200, 4000)):
        """
        Fit a spectrum with a given function(s) over a specified range(s)

        Parameters
        ----------
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
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1]+1])
        spec = spec[(spec > view_wind[0]) * (spec < view_wind[1])]
        hist, bin_b = np.histogram(spec, bins=n_bins)
        bin_w = np.diff(bin_b)
        bin_c = bin_b[:-1] + bin_w / 2
        plt.bar(bin_c, hist, width=bin_w, label='hist')

        fit_num = 1
        for func, bound, p0 in zip(funcs, bounds, p0s):
            x_fit = bin_c[(bin_c > bound[0]) * (bin_c < bound[1])]
            y_fit = hist[(bin_c > bound[0]) * (bin_c < bound[1])]
            popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0)
            x = np.linspace(view_wind[0], view_wind[1], 1000)
            y = func(x, *popt)
            label = ('{:.2e} ' * len(popt)).format(*popt)
            plt.plot(x, y, 'r--', label='fit ' + str(fit_num) + ' ' + label)
            fit_num += 1
        plt.legend()
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
        plt.show()


    def pre_post_pulsing(self, spe_thre):
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
        print(self._count_pulses(spe_thre))


    def _count_pulses(self, thre):
        """
        Counts pulses in group of waveforms above specified threshold

        Parameters
        ----------
            thre : float
                Threshold value above which pulses are counted
        """
        count = 0
        for waveform in self.w:
            x = np.arange(0, len(waveform) * 4, 4)
            w = waveform - thre
            cross = x[1:][w[1:] * w[:-1] < 0]
            count += len(cross) / 2

        return count


if __name__ == '__main__':
    r = Rooter('12.root')
    r.max_amplitudes()
    r.view_waveform(3)
    r.view_multi_spec([(160, 170), (150, 180), (162, 175)], n_bins=100)
    r.fit_spectrum((160, 170), [gaussian, gaussian],
                [(-200, 200), (400, 1000)], [[2000, 0, 50], [200, 500, 100]])
    r.pre_post_pulsing(200)
