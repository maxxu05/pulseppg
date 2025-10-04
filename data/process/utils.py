import pandas as pd
import numpy as np
from dotmap import DotMap
from scipy.signal import filtfilt
from scipy import signal
from sklearn.utils import resample
from math import gcd
from scipy.signal import filtfilt, resample_poly
from fractions import Fraction

class Preprocess:

    ###########################################################################
    ######################## Initialization of Biomarkers #####################
    ###########################################################################
    def __init__(self,fL=0.5000001, fH=12, order=4, sm_wins={'ppg':50,'vpg':10,'apg':10,'jpg':10}):
        """
        The purpose of the Preprocess class is to filter and calculate the PPG, PPG', PPG", and PPG'" signals.

        :param fL: Lower cutoff frequency (Hz)
        :type fL: float
        :param fH: Upper cutoff frequency (Hz)
        :type fH: float
        :param order: Filter order
        :type order: int
        :param sm_wins: dictionary of smoothing windows in millisecond:
            - ppg: window for PPG signal
            - vpg: window for PPG' signal
            - apg: window for PPG" signal
            - jpg: window for PPG'" signal
        :type sm_wins: dict

        """

        self.fL = fL
        self.fH = fH
        self.order=order
        self.sm_wins=sm_wins


    def get_signals(self, s: DotMap):
        '''This function calculates the preprocessed PPG, PPG', PPG", and PPG'" signals.

        :param s: a struct of PPG signal:
            - s.v: a vector of PPG values
            - s.fs: the sampling frequency of the PPG in Hz
            - s.filtering: a bool for filtering
        :type s: DotMap

        :return: ppg, vpg, apg, jpg: preprocessed PPG, PPG', PPG", and PPG'"
        '''

        ## PPG filtering
        if s.filtering:
            fL = self.fL
            fH = self.fH
            order = self.order

            if fL==0:
                b,a = signal.cheby2(order, 20, [fH], 'low', fs=s.fs)
            else:
                b, a = signal.cheby2(order, 20, [fL,fH], 'bandpass', fs=s.fs)

            ppg_cb2 = filtfilt(b, a, s.v)

            if s.fs >= 75:
                win = round(s.fs * self.sm_wins['ppg']/1000)
                B = 1 / win * np.ones(win)
                ppg = filtfilt(B, 1, ppg_cb2)
            else:
                ppg=ppg_cb2
        else:
            ppg=s.v

        if s.fs >= 150 and s.filtering:
            ## PPG' filtering
            win = round(s.fs * self.sm_wins['vpg']/1000)
            B1 = 1 / win * np.ones(win)
            dx = np.gradient(ppg)
            vpg = filtfilt(B1, 1, dx)

            ## PPG" filtering
            win = round(s.fs * self.sm_wins['apg']/1000)
            B2 = 1 / win * np.ones(win)
            ddx = np.gradient(vpg)
            apg = filtfilt(B2, 1, ddx)

            ## PPG'" filtering
            win = round(s.fs * self.sm_wins['jpg']/1000)
            B3 = 1 / win * np.ones(win)
            dddx = np.gradient(apg)
            jpg = filtfilt(B3, 1, dddx)
        else:
            vpg = np.gradient(ppg)
            apg = np.gradient(vpg)
            jpg = np.gradient(apg)

        return ppg, vpg, apg, jpg


def resample_lerp_vectorized(signal, orighz=64, newhz=50):
    """
    Resamples a signal using linear interpolation, with vectorized operations
    and ignoring overlapping timestamps. Includes a check to prevent index
    out of bounds error.
    """
    time_length = signal.shape[0]

    # Create arrays of timestamps at the original and new frequencies
    orig_timestamps = np.arange(time_length) / orighz
    new_timestamps = np.arange(0, time_length / orighz, 1 / newhz)

    # Find indices of closest timestamps before in the original timestamps
    closest_before_idx = np.searchsorted(orig_timestamps, new_timestamps, side='right') - 1

    # Handle edge case where new_timestamps[0] == orig_timestamps[0]
    closest_before_idx[0] = max(0, closest_before_idx[0])

    # Prevent index out of bounds error (new)
    closest_before_idx = np.clip(closest_before_idx, 0, time_length - 2)

    # Extract timestamps and values
    ts1 = orig_timestamps[closest_before_idx]
    ts2 = orig_timestamps[closest_before_idx + 1]  # Now safe
    v1 = signal[closest_before_idx]
    v2 = signal[closest_before_idx + 1]  # Now safe

    # Calculate the slope and intercept
    slope = (v2 - v1) / (ts2 - ts1)
    intercept = v1 - slope * ts1

    # Calculate the interpolated values
    resampled_signal = slope * new_timestamps + intercept

    return resampled_signal
    
def preprocess_one_ppg_signal(waveform,
                          frequency,
                          fL=0.5, 
                          fH=12, 
                          order=4, 
                          smoothing_windows={"ppg":50, "vpg":10, "apg":10, "jpg":10}):
    
    """
    Preprocessing a single PPG waveform using py PPG.
    https://pyppg.readthedocs.io/en/latest/Filters.html
    
    Args:
        waveform (numpy.array): PPG waveform for processing
        frequency (int): waveform frequency
        fL (float/int): high pass cut-off for chebyshev filter
        fH (float/int): low pass cut-off for chebyshev filter
        order (int): filter order
        smoothing_windows (dictionary): smoothing window sizes in milliseconds as dictionary
    
    Returns:
        ppg (numpy.array): filtered ppg signal
        ppg_d1 (numpy.array): first derivative of filtered ppg signal
        ppg_d2 (numpy.array): second derivative of filtered ppg signal
        ppg_d3 (numpy.array): third derivative of filtered ppg signal

    """

    prep = Preprocess(fL=fL,
                    fH=fH,
                    order=order,
                    sm_wins=smoothing_windows)
    
    signal = DotMap()
    signal.v = waveform
    signal.fs = frequency
    signal.filtering = True

    ppg, ppg_d1, ppg_d2, ppg_d3 = prep.get_signals(signal)

    return ppg, ppg_d1, ppg_d2, ppg_d3    
    
def resample_batch_signal(X, fs_original=64, fs_target=50, axis=-1):
    """
    Apply resampling to a 2D array with no of segments x values

    Args:
        X (np.array): 2D segments x values array
        fs_original (int/float): Source frequency 
        fs_target (int/float): Target frequency
        axis (int): index to apply the resampling.
    
    Returns:
        X (np.array): Resampled 2D segments x values array
    """
    # Convert fs_original and fs_target to Fractions
    fs_original_frac = Fraction(fs_original).limit_denominator()
    fs_target_frac = Fraction(fs_target).limit_denominator()
    
    # Find the least common multiple of the denominators
    lcm_denominator = np.lcm(fs_original_frac.denominator, fs_target_frac.denominator)
    
    # Scale fs_original and fs_target to integers
    fs_original_scaled = fs_original_frac * lcm_denominator
    fs_target_scaled = fs_target_frac * lcm_denominator
    
    # Calculate gcd of the scaled frequencies
    gcd_value = gcd(fs_original_scaled.numerator, fs_target_scaled.numerator)
    
    # Calculate the up and down factors
    up = fs_target_scaled.numerator // gcd_value
    down = fs_original_scaled.numerator // gcd_value
    
    # Perform the resampling
    X = resample_poly(X, up, down, axis=axis)
    
    return X    
