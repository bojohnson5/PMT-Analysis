#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot

from scipy.optimize import curve_fit
from fit_funcs import gaussian


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
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1] + 1])
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
            spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1] + 1])
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
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1] + 1])
        spec = spec[(spec > view_wind[0]) * (spec < view_wind[1])]
        hist, bin_b = np.histogram(spec, bins=n_bins)
        bin_w = np.diff(bin_b)
        bin_c = bin_b[:-1] + bin_w / 2
        plt.bar(bin_c, hist, width=bin_w, label='hist', color='k', alpha=0.5)

        fit_num = 1
        params = []
        for func, bound, p0 in zip(funcs, bounds, p0s):
            x_fit = bin_c[(bin_c > bound[0]) * (bin_c < bound[1])]
            y_fit = hist[(bin_c > bound[0]) * (bin_c < bound[1])]
            popt, _ = curve_fit(func, x_fit, y_fit, p0=p0)
            params.append(popt)
            x = np.linspace(view_wind[0], view_wind[1], 1000)
            y = func(x, *popt)
            plt.plot(x, y, '--', label='fit ' + str(fit_num))
            fit_num += 1
        adj_mean = params[1][1] - params[0][1]
        res = adj_mean / params[1][2]
        max_i = np.argmax(hist)
        peak_offset = 5
        peak_i = np.argmax(hist[max_i + peak_offset:])
        min_i = np.argmin(hist[max_i:peak_i])
        pv = hist[max_i + peak_offset:][peak_i] / hist[max_i:peak_i][min_i]
        pv_3 = hist[max_i + peak_offset:][peak_i] / \
            hist[self._find_nearest(bin_c, adj_mean * 0.3)]
        plt.legend()
        plt.text(1000, 1500, f'P/V: {pv:.2f}\nP/V (0.3PE): {pv_3:.2f}\nRes: '
                             f'{res:.2f}\nSPE Peak: {adj_mean:.2f}')
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
        plt.show()

    def _find_nearest(self, array, value):
        """Return index of array which is closest to value"""
        return np.argmin(np.abs(array - value))

    def pre_post_pulsing(self, spe_thre, pulse_thre, pre_wind, late_wind, after_wind):
        """
        Fit a spectrum with a given function(s) over a specified range(s)

        Parameters
        ----------
            spe_thre : int or float
                Threhold value for single photoelectrons
            pulse_thre : int or float
                Threshold value for pre- and post-pulses
            pre_wind : tuple of ints
                Start and end of window to count pre-pulses, indices not times
            late_wind : tuple of ints
                Start and end of window to count late-pulses, indices not times
            after_wind : tuple of ints
                Start and end of window to count after-pulses, indices not times
        """
        spe = 0
        pre = 0
        late = 0
        after = 0
        for waveform in self.w:
            x = np.arange(0, len(waveform))
            w = waveform - spe_thre
            cross = x[1:][w[1:] * w[:-1] < 0]
            if len(cross) > 0:
                spe_x = cross[0]
                pre_s = spe_x - pre_wind[1]
                pre_e = spe_x - pre_wind[0]
                late_s = spe_x + late_wind[0]
                late_e = spe_x + late_wind[1]
                after_s = spe_x + after_wind[0]
                after_e = spe_x + after_wind[1]
                spe += 1
                pre += self._count_pulses(waveform[pre_s:pre_e], pulse_thre)
                late += self._count_pulses(waveform[late_s:late_e], pulse_thre)
                after += self._count_pulses(waveform[after_s:after_e], pulse_thre)
        print(f'Pre-Pulsing: {pre / spe * 100 : .2f}%')
        print(f'Late-Pulsing: {late / spe * 100 : .2f}%')
        print(f'After-Pulsing: {after / spe * 100 : .2f}%')

    def _count_pulses(self, waveform, thre):
        """
        Counts pulses in group of waveforms above specified threshold

        Parameters
        ----------
            waveform : array
                Values correpsonding to the waveform
            thre : float
                Threshold value above which pulses are counted
        """
        count = 0
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
                   [(-200, 200), (500, 900)], [[2000, 0, 50], [200, 500, 100]])
    r.pre_post_pulsing(200, 0.3 * 200, (2, 23), (6, 37), (37, 6250))
