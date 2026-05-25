MODEL_PATH = "/home/razzannr/Documents/GitHub/Python-notes/Skripsi/consciousness_detection/models/model_VMD_filter_normal.pkl"

MUSE_FS = 256
MODEL_FS = 128

WINDOW_SEC = 5
MUSE_WINDOW_SIZE = MUSE_FS * WINDOW_SEC
MODEL_WINDOW_SIZE = MODEL_FS * WINDOW_SEC

DISPLAY_SECONDS = 5
STD_FLAT_THR = 1.0

K = 3
VMD_ALPHA = 100
VMD_TAU = 0.0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

THRESHOLD = 0.5

CSV_SIGNAL_COLUMN = "AF7"
DEFAULT_CSV_FS = 256

FEATURE_COLUMNS = [
    "mode1_sample_entropy",
    "mode1_spectral_entropy",
    "mode1_skewness",
    "mode1_kurtosis",
    "mode2_sample_entropy",
    "mode2_spectral_entropy",
    "mode2_skewness",
    "mode2_kurtosis",
    "mode3_sample_entropy",
    "mode3_spectral_entropy",
    "mode3_skewness",
    "mode3_kurtosis",
]