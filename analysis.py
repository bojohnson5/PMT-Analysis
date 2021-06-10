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

    def __init__(self, fi, filt=False, thre=None):
        """
        Parameters
        ----------
            fi : string
                File name used for the analysis
            filt : bool, optional
                Use waveforms filtered for noisiness in the analysis
            thre : float, optional
                SPE threshold value for filtering waveforms
        """
        with uproot.open(fi + ':waveformTree') as tree:
            waveforms = tree['waveform'].array()
            baselines = tree['baseline'].array()
            polarity = tree['polarity'].array()
            self.w = ak.to_numpy((waveforms - baselines) * polarity)
            self.raw_w = ak.to_numpy(waveforms)
            self.fi = fi
            if filt:
                self.w = self._cut_noise(thre)

    def view_waveform(self, num, view_wind=None, view_raw=False):
        """
        View a waveform from ROOT file

        Parameters
        ----------
            num : int
                Waveform number to view
            view_wind : tuple of ints, optional
                How much of the waveform to view
            view_raw : bool, optional
                View waveforms with no polarity or baseline adjustments
        """
        if view_wind is None:
            x = np.arange(0, len(self.w[num]) * 4, 4)
            if view_raw:
                plt.plot(x, self.raw_w[num])
            else:
                plt.plot(x, self.w[num])
        else:
            view_st, view_en = view_wind
            i_st = view_st // 4 # convert ns to index
            i_en = view_en // 4 # convert ns to index
            x = np.arange(view_st, view_en, 4)
            try:
                if view_raw:
                    plt.plot(x, self.raw_w[num][i_st:i_en])
                else:
                    plt.plot(x, self.w[num][i_st:i_en])
            except ValueError:
                if view_raw:
                    plt.plot(x, self.raw_w[num][i_st:i_en+1])
                else:
                    plt.plot(x, self.w[num][i_st:i_en+1])
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
        int_st, int_en = wind
        i_st = int_st // 4 # convert ns to index
        i_en = int_en // 4 # convert ns to index
        spec = np.apply_along_axis(np.sum, 1, self.w[:, i_st:i_en + 1])
        plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])],
                 bins=n_bins, histtype='step', label='Int. wind. ' + str(int_st)
                                                     + '-' + str(int_en))
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
        int_st, int_en = wind
        i_st = int_st // 4 # convert ns to index
        i_en = int_en // 4 # convert ns to index
        spec = np.apply_along_axis(np.sum, 1, self.w[:, i_st:i_en + 1])
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
                plt.plot(x[(y > 1) * (y < 15000)], y[(y > 1) * (y < 15000)],
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
            zeros = np.sum(hist[:min_i])
            ones = np.sum(hist[min_i:])
            per_0pe = zeros / (zeros + ones)
            plt.text(500, 1500, f'P/V: {pv:.2f}\nRes: '
                                 f'{res:.2f}\nSPE Peak: {adj_mean:.2f}\n'
                                 f'% 0-PE: {per_0pe:.2f}')
        if y_log:
            plt.yscale('log')
        if view:
            plt.hist(spec[(spec > view_wind[0]) * (spec < view_wind[1])],
                     bins=n_bins, histtype='step', label='Int. wind. ' + str(int_st)
                     + '-' + str(int_en))
            plt.legend()
            plt.xlabel('Integrated ADC', loc='right')
            plt.ylabel('Count', loc='top')
            plt.title('Run ' + self.fi[0:2] + ' Spectrum and Fits')
            plt.show()
            print(params)
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
        i = 0
        for waveform in self.w:
            x = np.arange(0, len(waveform))
            w = waveform - spe_thre
            cross = x[1:][w[1:] * w[:-1] < 0]
            if len(cross) > 0:
                spe_x = cross[0]
                pre_s = spe_x - pre_wind[1] // 4
                pre_e = spe_x - pre_wind[0] // 4
                late_s = spe_x + late_wind[0] // 4
                late_e = spe_x + late_wind[1] // 4
                after_s = spe_x + after_wind[0] // 4
                after_e = spe_x + after_wind[1] // 4
                spe += 1
                pre += self._count_pulses(waveform[pre_s:pre_e], pulse_thre)
                late += self._count_pulses(waveform[late_s:late_e], pulse_thre)
                after += self._count_pulses(waveform[after_s:after_e], pulse_thre)
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
                try:
                    main_st = idx[0][0]
                    main_en = idx[0][1]
                except IndexError as i_err:
                    print(f'IndexError on waveform {i}')
                    continue
                pre = filt_w[i][:main_st - 1] + 15
                post = filt_w[i][main_en + 2:] + 15
                pre_cross = pre[1:][pre[1:] * pre[:-1] < 0]
                post_cross = post[1:][post[1:] * post[:-1] < 0]
                len_pre = len(pre_cross)
                len_post = len(post_cross)
                if len_pre // 2 > 3 or len_post // 2 > 3:
                    to_delete.append(i)
        return np.delete(filt_w, to_delete, axis=0)

    def post_pulse_hist(self, spe_thre, pulse_thre, n_bins=10):
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
        plt.hist(hist, bins=n_bins, histtype='step')
        plt.xlabel('Time after main pulse [ns]', loc='right')
        plt.ylabel('Counts', loc='top')
        plt.title('Run ' + self.fi[0:2] + ' ' +'After-Pulsing Time Distribution')
        plt.show()

    def view_dark_rate(self, thre_st, thre_en=None, thre_step=10, up_time=5*60):
        """
        Count the number of pulses above a certain threshold and determine the corresponding
        rate

        Parameters
        ----------
            thre_st : int
                The threshold value to find the rate at, or the beginning threshold value
                in a range of values
            thre_en : int, optional
                If given, the ending threshold value to find the rate at
            thre_step : int, optional
                The step value between beginning and ending threshold
            up_time : float, optional
                The runtime of the data set in seconds
        """
        if thre_en is not None:
            thres = np.arange(thre_st, thre_en, thre_step)
        else:
            thres = np.array([thre_st])
        counts = np.zeros_like(thres)
        for i, thre in np.ndenumerate(thres):
            for j in range(len(self.w)):
                w = self.w[j] - thre
                cross = w[1:] * w[:-1] < 0
                counts[i] += len(w[1:][cross]) // 2
            print(f'For threshold {thre} saw {counts[i]} events for a rate of {counts[i] / up_time:.2f} Hz')
        plt.scatter(thres, counts / up_time)
        plt.xlabel('Threshold [ADC]', loc='right')
        plt.ylabel('Rate [Hz]', loc='top')
        plt.title(f'Run {self.fi[0:2]} Rate Counts')
        plt.show()

if __name__ == '__main__':
    #  #  Fit values and initial parameters for run 12
    r1 = Rooter('12.root')
    #  r1.fit_spectrum((160 * 4, 170 * 4), [deap_ped, spe],
    #                 [(-200, 200), (0, 900)],
    #                 [[6.7e4, -20, 25], [9e4, 8.8e2, 8e-2, 1e5, 1.43, 7.14, 2.2e-1, 0.02, 500]],
    #                 y_log=True, view_wind=(-200, 2000), view=True, convolve=True)
    r1.view_dark_rate(180)

    #  #  Fit values and initial parameters for run 10
    #  r2 = Rooter('10.root', filt=True, thre=187.5)
    #  r2.fit_spectrum((660, 690), [deap_ped], [(-100, 150)],
    #                 [[5e5, -70, 20]], y_log=True)
    #  r2.fit_spectrum((660, 690), [deap_gamma], [(500, 1500)],
    #                 [[1e5, 200, .25]], y_log=True)
    #  r2.fit_spectrum((660, 690), [deap_expo], [(0, 300)],
    #                 [[12.0, 0.02, 500]], y_log=True)
    #  r2.fit_spectrum((660, 690), [deap_ped, deap_ped], [(-200, 200), (450, 1120)],
                    #  [[5e5, -70, 20], [1e5, 700, 350]], y_log=True, print_res=True)

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
