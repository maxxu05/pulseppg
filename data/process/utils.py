import pandas as pd
import numpy as np


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
    
    
def upsample_lerp_vectorized(signal, orighz=64, newhz=50):
    pass