#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import uproot
import copy

from scipy.optimize import curve_fit
from fit_funcs import deap_expo, deap_gamma, deap_ped, spe, ped_spe

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

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

    def __init__(self, fi, thre, filt=True):
        """
        Parameters
        ----------
            fi : string
                File name used for the analysis
            thre : float
                SPE threshold value
            filt : bool, optional
                Use waveforms filtered for noisiness in the analysis
        """
        with uproot.open(fi + ':waveformTree') as tree:
            waveforms = tree['waveform'].array()
            baselines = tree['baseline'].array()
            polarity = tree['polarity'].array()
            self.w = ak.to_numpy((waveforms - baselines) * polarity)
            self.fi = fi
            self.thre = thre
            if filt:
                self.w = self._cut_noise(self.thre)
            else:
                self.filt_w = self._cut_noise(self.thre)

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

    def view_max_amplitudes(self, n_bins=150, view_wind=(0, 2000)):
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
            y_log : bool, optional
                Plot y axis with log scale
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
                     y_log=False, print_res=False, view=True, convolve=False):
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
            y_log : bool, optional
                If y axis should be log
            print_res : bool, optional
                Print calculation results on figure
            view : bool, optional
                Plot the fits and histogram
            convolve : bool, optional
                Convolve fits and plot them
        """
        spec = np.apply_along_axis(np.sum, 1, self.w[:, wind[0]:wind[1] + 1])
        spec = spec[(spec > view_wind[0]) * (spec < view_wind[1])]
        hist, bin_b = np.histogram(spec, bins=n_bins)
        bin_w = np.diff(bin_b)
        bin_c = bin_b[:-1] + bin_w / 2

        fit_num = 1
        params = []
        x = np.linspace(view_wind[0], view_wind[1], 1000)
        ys = []
        for func, bound, p0 in zip(funcs, bounds, p0s):
            x_fit = bin_c[(bin_c > bound[0]) * (bin_c < bound[1])]
            y_fit = hist[(bin_c > bound[0]) * (bin_c < bound[1])]
            popt, _ = curve_fit(func, x_fit, y_fit, p0=p0)
            params.append(popt)
            y = func(x, *popt)
            ys.append(y)
            if view:
                plt.plot(x[(y > 1) * (y < 2000)], y[(y > 1) * (y < 2000)],
                         '--', label='fit ' + str(fit_num))
            fit_num += 1
        if print_res:
            adj_mean = params[1][1] - params[0][1]
            res = adj_mean / params[1][2]
            max_i = np.argmax(hist)
            peak_offset = 5
            peak_i = np.argmax(hist[max_i + peak_offset:])
            peak_i = max_i + peak_offset + peak_i
            min_i = np.argmin(hist[max_i:peak_i])
            min_i = max_i + min_i
            pv = hist[peak_i] / hist[min_i]
            plt.text(1000, 1500, f'P/V: {pv:.2f}\nRes: '
                                 f'{res:.2f}\nSPE Peak: {adj_mean:.2f}')
        if y_log:
            plt.yscale('log')
        if view:
            plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])],
                     bins=n_bins, histtype='step', label='Int. wind. ' + str(wind[0] * 4)
                     + '-' + str(wind[1] * 4))
            plt.legend()
            plt.xlabel('Integrated ADC', loc='right')
            plt.ylabel('Count', loc='top')
            plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
            plt.show()
        if convolve:
            self._convolve_fits(spec, x, ys)

    def _convolve_fits(self, hist, x, ys):
        """
        Convolve the pedestal with spe function fits from paper 'In-situ characterization of the
        Hamamatsu R5912-HQE photomultiplier tubes used in the DEAP-3600 experiment'

        Parameters
        ----------
            hist : array
                The histogram values the fit functions are fitting
            x : array
                The x range to plot
            ys : list of arrays
                the y values from the fits for pedestal and spe
        """
        plt.hist(hist, bins=150, histtype='step', label='Spectrum')
        ped = ys[0]
        spe = np.nan_to_num(ys[1])
        win = ped[50:115]
        conv1 = np.convolve(spe, win, 'same') / np.sum(win)
        conv1 = self._pad_array_left(conv1, ped)
        plt.plot(x, ped + conv1, label='Full model')
        plt.xlabel('Integrated ADC', loc='right')
        plt.ylabel('Counts', loc='top')
        plt.semilogy()
        plt.title('Run ' + self.fi[0:2] + ' Spectrum and DEAP Fit')
        plt.ylim(bottom=10, top=2000)
        plt.legend()
        plt.show()

    def _pad_array_left(self, array, array_ref):
        """
        Pad an array to the left with zeros to match the length of another array

        Parameters
        ----------
            array : array
                Array to pad
            array_ref : array
                Array for which the length of parameter array should match

        Returns
        -------
            res : array
                A copy of array with zeros padded on the left to match
                the length of array_ref
        """
        len1 = array.shape[0]
        len2 = array_ref.shape[0]
        res = np.zeros(array_ref.shape)
        res[len2 - len1:] = array
        return res

    def _find_nearest(self, array, value):
        """
        Return index of array which is closest to value

        Parameters
        ----------
            array : array
                Array to find the index of value in
            value : float
                The value to find in array, or closest to it

        Returns
        -------
            Index value of array with entry closest to value
        """
        return np.argmin(np.abs(array - value))

    def pre_post_pulsing(self, spe_thre, pulse_thre, pre_wind, late_wind, after_wind):
        """
        Calculate the pre-, late-, and after-pulsing percentages

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
                if i < 5550 and (new_pre > 1 or new_late > 1 or new_after > 1):
                    plt.plot(waveform)
                    plt.axhline(spe_thre, c='r', ls='--')
                    plt.axhline(pulse_thre, c='r', ls='-.')
                    plt.show()
                pre += new_pre
                late += new_late
                after += new_after
            i += 1
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

    def _cut_noise(self, thre):
        """
        Remove noisy waveforms

        Parameters
        ----------
            thre : float
                SPE threshold value

        Returns
        -------
            2d array of waveforms
        """
        filt_w = copy.deepcopy(self.w)
        to_delete = []
        for i in range(len(filt_w)):
            peak = np.max(filt_w[i])
            if peak > thre:
                w = filt_w[i] - thre
                cross = w[1:] * w[:-1] < 0
                idx = np.where(cross == True)
                main_st = idx[0][0]
                main_en = idx[0][1]
                pre = filt_w[i][:main_st - 1] + 15
                post = filt_w[i][main_en + 2:] + 15
                pre_cross = pre[1:][pre[1:] * pre[:-1] < 0]
                post_cross = post[1:][post[1:] * post[:-1] < 0]
                len_pre = len(pre_cross)
                len_post = len(post_cross)
                if len_pre // 2 > 3 or len_post // 2 > 3:
                    to_delete.append(i)
        return np.delete(filt_w, to_delete, axis=0)

    def post_pulse_hist(self, spe_thre, pulse_thre):
        hist = np.array([])
        for waveform in self.w:
            x = np.arange(0, len(waveform))
            w = waveform - spe_thre
            if len(x[w > 0]) > 1:
                st_i = x[w > 0][1]
            elif len(x[w > 0]) == 1:
                st_i = x[w > 0][0]
            else:
                continue
            late_st = 25 // 4
            st = st_i + late_st
            w = w[st:]
            w = w - pulse_thre
            x = (x[st:] - st) * 4
            pulses = x[w > 0]
            #  pulses = np.delete(pulses, [i for i in range(len(pulses)) if i % 2 != 0])
            if pulses.size != 0:
                hist = np.append(hist, pulses)
        plt.hist(hist, bins=10, histtype='step')
        plt.show()

if __name__ == '__main__':
    thre = 187.5
    r = Rooter('12.root', thre, filt=True)
    #  r.view_max_amplitudes()
    #  r.gain((160, 170))
    #  r.view_spectrum((162, 175), y_log=True)
    #  r.fit_spectrum((162, 175), [deap_ped, spe],
                   #  [(-200, 200), (0, 900)],
                   #  [[6.7e4, -20, 25], [9e4, 8.8e2, 8e-2, 1e5, 1.43, 7.14, 2.2e-1, 0.02, 500]],
                   #  y_log=True, view_wind=(-200, 2000), view=False, convolve=True)
    #  r.pre_post_pulsing(thre, thre * 0.3, (2, 23), (6, 38), (38, 6250))
    r.post_pulse_hist(thre, thre * 0.3)
    #  #  Classification of waveforms using scikit-learn
    #  #  use with filtering turned off
    #  data = r.w
    #  x_train = data[:10000]
    #  y_train = []
    #  for i in range(len(x_train)):
        #  peak = np.max(x_train[i])
        #  if peak > thre:
            #  w = x_train[i] - thre
            #  cross = w[1:] * w[:-1] < 0
            #  idx = np.where(cross == True)
            #  main_st = idx[0][0]
            #  main_en = idx[0][1]
            #  pre = x_train[i][:main_st - 1] + 15
            #  post = x_train[i][main_en + 2:] + 15
            #  pre_cross = pre[1:][pre[1:] * pre[:-1] < 0]
            #  post_cross = post[1:][post[1:] * post[:-1] < 0]
            #  len_pre = len(pre_cross)
            #  len_post = len(post_cross)
            #  if len_pre // 2 > 3 or len_post // 2 > 3:
                #  y_train.append(0)
            #  else:
                #  y_train.append(1)
        #  else:
            #  y_train.append(2)
    #  x_test = data[10000:]
    #  y_test = []
    #  for i in range(len(x_test)):
        #  peak = np.max(x_test[i])
        #  if peak > thre:
            #  w = x_test[i] - thre
            #  cross = w[1:] * w[:-1] < 0
            #  idx = np.where(cross == True)
            #  main_st = idx[0][0]
            #  main_en = idx[0][1]
            #  pre = x_test[i][:main_st - 1] + 15
            #  post = x_test[i][main_en + 2:] + 15
            #  pre_cross = pre[1:][pre[1:] * pre[:-1] < 0]
            #  post_cross = post[1:][post[1:] * post[:-1] < 0]
            #  len_pre = len(pre_cross)
            #  len_post = len(post_cross)
            #  if len_pre // 2 > 3 or len_post // 2 > 3:
                #  y_test.append(0)
            #  else:
                #  y_test.append(1)
        #  else:
            #  y_test.append(2)
    #  classifier = RandomForestClassifier()
    #  model = classifier.fit(x_train, y_train)
    #  pred = model.predict(x_test)
    #  print('Random Forest')
    #  print(' accuracy = ', accuracy_score(y_test, pred))
    #  print(confusion_matrix(y_test, pred))
    #  classifier = gbc()
    #  model = classifier.fit(x_train, y_train)
    #  pred = model.predict(x_test)
    #  print('Gradient Boosting')
    #  print(' accuracy = ', accuracy_score(y_test, pred))
    #  print(confusion_matrix(y_test, pred))
