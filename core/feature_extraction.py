import numpy as np
import antropy as ant
from scipy.stats import skew, kurtosis


def extract_features(signal, sf=128):
    signal = np.asarray(signal, dtype=float).ravel()

    if len(signal) < sf or (not np.isfinite(np.std(signal))) or np.std(signal) < 1e-8:
        return None

    features = {}

    try:
        features["sample_entropy"] = float(ant.sample_entropy(signal))
    except Exception:
        features["sample_entropy"] = np.nan

    try:
        features["spectral_entropy"] = float(
            ant.spectral_entropy(signal, sf=sf, method="welch", normalize=True)
        )
    except Exception:
        features["spectral_entropy"] = np.nan

    try:
        features["skewness"] = float(skew(signal))
    except Exception:
        features["skewness"] = np.nan

    try:
        features["kurtosis"] = float(kurtosis(signal))
    except Exception:
        features["kurtosis"] = np.nan

    return features


def extract_features_from_modes(modes, sf=128):
    feature_dict = {}

    for i in range(modes.shape[0]):
        feats = extract_features(modes[i], sf=sf)

        if feats is None:
            return None

        feature_dict[f"mode{i+1}_sample_entropy"] = feats["sample_entropy"]
        feature_dict[f"mode{i+1}_spectral_entropy"] = feats["spectral_entropy"]
        feature_dict[f"mode{i+1}_skewness"] = feats["skewness"]
        feature_dict[f"mode{i+1}_kurtosis"] = feats["kurtosis"]

    return feature_dict