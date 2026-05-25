import time
import traceback
from collections import deque

import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

from config import DISPLAY_SECONDS, WINDOW_SEC, CSV_SIGNAL_COLUMN
from core.classifier import predict_window_5s
from core.voting import (
    majority_vote_predictions,
    average_probability,
    average_latency,
)


class CSVReplayWorker(QThread):
    sample_received = pyqtSignal(float, float)
    status_changed = pyqtSignal(str)

    # pred, prob, status, latency, win_end_sec
    prediction_ready = pyqtSignal(object, object, str, float, float)

    # modes, features, filtered_signal, status, win_end_sec
    vmd_ready = pyqtSignal(object, object, object, str, float)

    error_occurred = pyqtSignal(str)

    finished_replay = pyqtSignal()

    def __init__(
        self,
        model,
        csv_path,
        csv_fs=256,
        signal_column=CSV_SIGNAL_COLUMN,
        realtime=True,
    ):
        super().__init__()

        self.model = model
        self.csv_path = csv_path
        self.csv_fs = int(csv_fs)
        self.signal_column = signal_column
        self.realtime = realtime

        self.running = False

        self.display_buffer = deque(maxlen=self.csv_fs * DISPLAY_SECONDS)
        self.display_time = deque(maxlen=self.csv_fs * DISPLAY_SECONDS)

        self.window_buffer = []
        self.vote_buffer = []

    def stop(self):
        self.running = False

    def run(self):
        try:
            self.status_changed.emit("Loading CSV file...")

            df = pd.read_csv(self.csv_path)

            if self.signal_column not in df.columns:
                self.error_occurred.emit(
                    f"Column '{self.signal_column}' was not found in CSV.\n"
                    f"Available columns: {list(df.columns)}"
                )
                return

            signal = df[self.signal_column].dropna().to_numpy(dtype=float)

            if len(signal) == 0:
                self.error_occurred.emit("CSV file does not contain valid AF7 data.")
                return

            window_size = int(self.csv_fs * WINDOW_SEC)

            if len(signal) < window_size:
                self.error_occurred.emit(
                    f"CSV data is too short. Minimum required samples: {window_size}."
                )
                return

            self.status_changed.emit(
                f"CSV loaded. Samples: {len(signal)} | FS: {self.csv_fs} Hz | Column: {self.signal_column}"
            )

            self.running = True

            sample_interval = 1.0 / self.csv_fs

            for idx, af7_val in enumerate(signal):
                if not self.running:
                    break

                t_rel = idx / self.csv_fs
                af7_val = float(af7_val)

                self.display_time.append(t_rel)
                self.display_buffer.append(af7_val)
                self.window_buffer.append(af7_val)

                self.sample_received.emit(af7_val, t_rel)

                if len(self.window_buffer) == window_size:
                    raw_window = np.array(self.window_buffer, dtype=float)

                    pred, prob, status, latency, modes, feats, filtered = predict_window_5s(
                        raw_window,
                        self.model,
                        input_fs=self.csv_fs,
                    )

                    self.vmd_ready.emit(modes, feats, filtered, status, t_rel)

                    self.vote_buffer.append((pred, prob, status, latency, t_rel))

                    if len(self.vote_buffer) == 3:
                        final_pred = majority_vote_predictions(self.vote_buffer)
                        final_prob = average_probability(self.vote_buffer)
                        final_latency = average_latency(self.vote_buffer)
                        final_time = self.vote_buffer[-1][4]

                        if final_pred is None:
                            final_status = "Voting failed: fewer than 2 valid windows"
                        else:
                            final_status = "15-second majority voting"

                        self.prediction_ready.emit(
                            final_pred,
                            final_prob,
                            final_status,
                            final_latency,
                            final_time,
                        )

                        self.vote_buffer = []

                    self.window_buffer = []

                if self.realtime:
                    time.sleep(sample_interval)

            self.status_changed.emit("CSV replay finished.")
            self.finished_replay.emit()

        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(f"{e}\n\n{tb}")