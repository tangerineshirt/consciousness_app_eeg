import time
from math import gcd

import numpy as np
import pandas as pd
import joblib

from scipy.signal import decimate, resample_poly

from config import (
    MODEL_PATH,
    WINDOW_SEC,
    MODEL_FS,
    STD_FLAT_THR,
    K,
    THRESHOLD,
    FEATURE_COLUMNS,
)

from core.preprocessing import apply_eeg_filter
from core.vmd_processor import apply_vmd
from core.feature_extraction import extract_features_from_modes


def load_classifier_model():
    return joblib.load(MODEL_PATH)


def format_pred(pred):
    if pred is None:
        return "-"
    return "Conscious" if int(pred) == 1 else "Unconscious"


def resample_to_model_fs(signal, input_fs, model_fs):
    signal = np.asarray(signal, dtype=float).ravel()

    if int(input_fs) == int(model_fs):
        return signal

    g = gcd(int(input_fs), int(model_fs))
    up = int(model_fs) // g
    down = int(input_fs) // g

    return resample_poly(signal, up, down)


def convert_input_window_to_model_fs(raw_window, input_fs):
    raw_window = np.asarray(raw_window, dtype=float).ravel()

    if int(input_fs) == int(MODEL_FS):
        return raw_window

    # Kasus umum Muse 2: 256 Hz -> 128 Hz
    if int(input_fs) == 256 and int(MODEL_FS) == 128:
        return decimate(raw_window, 2, zero_phase=True)

    # Untuk sampling rate lain, gunakan resample_poly
    return resample_to_model_fs(raw_window, input_fs, MODEL_FS)


def predict_window_5s(raw_window, model, input_fs=256):
    """
    raw_window:
        Window EEG 5 detik dari Muse real-time atau CSV replay.

    input_fs:
        Sampling rate dari input.
        - Muse 2 biasanya 256 Hz.
        - CSV bisa 256 Hz atau 128 Hz, tergantung file.

    Return:
        pred, prob, status, latency, modes, feats, filtered
    """

    raw_window = np.asarray(raw_window, dtype=float).ravel()

    expected_size = int(input_fs * WINDOW_SEC)

    if len(raw_window) != expected_size:
        return (
            None,
            None,
            f"Invalid input window size: expected {expected_size}, got {len(raw_window)}",
            0.0,
            None,
            None,
            None,
        )

    t0 = time.perf_counter()

    try:
        seg_5s = convert_input_window_to_model_fs(raw_window, input_fs=input_fs)
    except Exception:
        return None, None, "Downsampling/resampling failed", 0.0, None, None, None

    expected_model_size = int(MODEL_FS * WINDOW_SEC)

    if len(seg_5s) != expected_model_size:
        if len(seg_5s) > expected_model_size:
            seg_5s = seg_5s[:expected_model_size]
        else:
            return (
                None,
                None,
                f"Invalid model window size after resampling: expected {expected_model_size}, got {len(seg_5s)}",
                0.0,
                None,
                None,
                None,
            )

    # DC offset correction
    seg_5s = seg_5s - np.mean(seg_5s)

    if np.std(seg_5s) < STD_FLAT_THR:
        t1 = time.perf_counter()
        return None, None, "Flat signal window", t1 - t0, None, None, None

    try:
        filtered = apply_eeg_filter(seg_5s, fs=MODEL_FS)
        modes = apply_vmd(filtered, K=K)
        feats = extract_features_from_modes(modes, sf=MODEL_FS)
    except Exception:
        t1 = time.perf_counter()
        return None, None, "Feature extraction failed", t1 - t0, None, None, None

    if feats is None:
        t1 = time.perf_counter()
        return None, None, "Feature extraction failed", t1 - t0, None, None, None

    if not np.all(np.isfinite(list(feats.values()))):
        t1 = time.perf_counter()
        return None, None, "Feature contains NaN/Inf", t1 - t0, modes, feats, filtered

    x_df = pd.DataFrame(
        [[feats[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(x_df)[0, 1])
        pred = int(prob >= THRESHOLD)
    else:
        prob = None
        pred = int(model.predict(x_df)[0])

    t1 = time.perf_counter()

    return pred, prob, "OK", t1 - t0, modes, feats, filtered