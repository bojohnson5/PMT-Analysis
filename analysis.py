#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot

from scipy.optimize import curve_fit
from fit_funcs import deap_expo, deap_gamma, deap_ped, spe

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

    def view_spectrum(self, wind, n_bins=150, view_wind=(-200, 4000), y_log=False):
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
                 bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0] * 4)
                                                     + '-' + str(wind[1] * 4))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum')
        plt.legend()
        if y_log:
            plt.yscale('log')
        plt.show()

    def gain(self, wind):
        """
        Calculate the gain of the PMT as an average of all waveform integrated currents
        
        Parameters
        ----------
            wind : tuple of ints
                Integration window stop and start for current integration
        """
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1] + 1])
        volts = np.linspace(0, 2, 4096)
        spec = spec.astype(int)
        spec = spec[(spec > 0) * (spec < 4096)]
        charge = volts[spec] / 50 * 4e-9 / 1.6e-19
        print(f'{np.average(charge):.2e}')

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
                     bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0] * 4)
                                                         + '-' + str(wind[1] * 4))
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum')
        plt.legend()
        plt.show()

    def fit_spectrum(self, wind, funcs, bounds, p0s, n_bins=150, view_wind=(-200, 4000),
                     y_log=False, print_res=False):
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
        plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])],
                 bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0] * 4)
                 + '-' + str(wind[1] * 4))

        fit_num = 1
        params = []
        for func, bound, p0 in zip(funcs, bounds, p0s):
            x_fit = bin_c[(bin_c > bound[0]) * (bin_c < bound[1])]
            y_fit = hist[(bin_c > bound[0]) * (bin_c < bound[1])]
            popt, _ = curve_fit(func, x_fit, y_fit, p0=p0)
            params.append(popt)
            x = np.linspace(view_wind[0], view_wind[1], 1000)
            y = func(x, *popt)
            plt.plot(x[y > 1e-2], y[y > 1e-2], '--', label='fit ' + str(fit_num))
            fit_num += 1
        if print_res:
            adj_mean = params[1][1] - params[0][1]
            res = adj_mean / params[1][2]
            max_i = np.argmax(hist)
            peak_offset = 5
            peak_i = np.argmax(hist[max_i + peak_offset:])
            min_i = np.argmin(hist[max_i:peak_i])
            pv = hist[max_i + peak_offset:][peak_i] / hist[max_i:peak_i][min_i]
            pv_3 = hist[max_i + peak_offset:][peak_i] / \
                hist[self._find_nearest(bin_c, adj_mean * 0.3)]
            plt.text(1000, 1500, f'P/V: {pv:.2f}\nP/V (0.3PE): {pv_3:.2f}\nRes: '
                                 f'{res:.2f}\nSPE Peak: {adj_mean:.2f}')
        if y_log:
            plt.yscale('log')
        plt.legend()
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
        plt.show()

    def deap_fit_spectrum(self, wind, funcs, bounds, p0s, n_bins=150, view_wind=(-200, 4000), 
                          y_log=False):
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
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Count', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
        if y_log:
            plt.yscale('log')
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
        i = 0
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
                new_pre = self._count_pulses(waveform[pre_s:pre_e], pulse_thre)
                new_late = self._count_pulses(waveform[late_s:late_e], pulse_thre)
                new_after = self._count_pulses(waveform[after_s:after_e], pulse_thre)
                pre += new_pre
                late += new_late
                after += new_after
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
        count += len(cross) // 2

        return count


if __name__ == '__main__':
    r = Rooter('12.root')
    r.max_amplitudes()
    #  r.view_waveform(i)
    #  r.view_spectrum((160, 170), y_log=True)
    #  r.deap_fit_spectrum((160, 170), [spe_fit], [(-200, 4000)],
                   #  [[2000, 0, 50, 200, 50, 50, 100, 10, 10, 10, 1]], y_log=True)
    #  r.view_multi_spec([(160, 170), (150, 180), (162, 175)], n_bins=100)
    #  r.fit_spectrum((160, 170), [deap_ped, deap_gamma],
                   #  [(-200, 200), (500, 1000)],
                   #  [[2000, 0, 50], [400, 10, 5]],
                   #  y_log=True)
    #  r.fit_spectrum((160, 170), [gaussian, gaussian],
                   #  [(-200, 200), (500, 900)], [[2000, 0, 50], [200, 500, 100]])
    spe_adc = 187.5
    r.pre_post_pulsing(spe_adc, 0.3 * spe_adc, (2, 23), (6, 37), (37, 6250))
    #  r.gain((160, 170))
