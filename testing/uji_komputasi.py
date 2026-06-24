import time
import pandas as pd
import numpy as np

WINDOW_SEC = 5
DEFAULT_CSV_FS = 256
from core.classifier import load_classifier_model
from core.classifier import predict_window_5s


csv_path = "/home/razzannr/Documents/GitHub/consciousness_app_eeg/recorded_raw/fadlan_sadar.csv"
signal_column = "AF7"
input_fs = DEFAULT_CSV_FS

model = load_classifier_model()

df = pd.read_csv(csv_path)
signal = df[signal_column].dropna().to_numpy(dtype=float)

window_size = int(input_fs * WINDOW_SEC)

results = []

for i, start in enumerate(range(0, len(signal) - window_size + 1, window_size), start=1):
    raw_window = signal[start:start + window_size]

    t0 = time.perf_counter()

    pred, prob, status, latency, modes, feats, filtered = predict_window_5s(
        raw_window,
        model=model,
        input_fs=input_fs
    )

    t1 = time.perf_counter()

    computation_time = t1 - t0

    results.append({
        "window": i,
        "start_sample": start,
        "end_sample": start + window_size,
        "start_time_sec": start / input_fs,
        "end_time_sec": (start + window_size) / input_fs,
        "prediction": pred,
        "probability": prob,
        "status": status,
        "latency_from_function_sec": latency,
        "computation_time_sec": computation_time
    })

result_df = pd.DataFrame(results)

summary_df = pd.DataFrame({
    "metric": [
        "jumlah_window",
        "mean_time_sec",
        "min_time_sec",
        "max_time_sec",
        "std_time_sec"
    ],
    "value": [
        len(result_df),
        result_df["computation_time_sec"].mean(),
        result_df["computation_time_sec"].min(),
        result_df["computation_time_sec"].max(),
        result_df["computation_time_sec"].std()
    ]
})

result_df.to_csv("hasil_waktu_komputasi_per_window.csv", index=False)
summary_df.to_csv("ringkasan_waktu_komputasi.csv", index=False)

print(result_df)
print(summary_df)