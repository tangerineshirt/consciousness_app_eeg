import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, tf2sos


def apply_eeg_filter(signal, fs=128):
    signal = np.asarray(signal, dtype=float).ravel()
    signal = signal - np.mean(signal)

    sos_band = butter(
        N=4,
        Wn=[0.5, 45.0],
        btype="bandpass",
        fs=fs,
        output="sos"
    )

    filtered = sosfiltfilt(sos_band, signal)

    b_notch, a_notch = iirnotch(
        w0=50.0,
        Q=30.0,
        fs=fs
    )

    sos_notch = tf2sos(b_notch, a_notch)
    filtered = sosfiltfilt(sos_notch, filtered)

    return filtered

def normalize_window(signal):
    signal = np.asarray(signal, dtype=float).ravel()
    signal = signal - np.mean(signal)
    signal = signal / (np.std(signal) + 1e-8)
    return signal