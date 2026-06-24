import traceback
from collections import deque

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from PyQt5.QtCore import QThread, pyqtSignal

from config import MUSE_FS, DISPLAY_SECONDS, MUSE_WINDOW_SIZE
from core.classifier import predict_window_5s
from core.voting import (
    majority_vote_predictions,
    average_probability,
    average_latency,
)


class MuseStreamWorker(QThread):
    sample_received = pyqtSignal(float, float)
    status_changed = pyqtSignal(str)

    # pred, prob, status, latency, win_end_sec
    prediction_ready = pyqtSignal(object, object, str, float, float)
    window_prediction_ready = pyqtSignal(object, object, str, float, float)

    # modes, features, filtered_signal, status, win_end_sec
    vmd_ready = pyqtSignal(object, object, object, str, float)

    error_occurred = pyqtSignal(str)

    def __init__(self, model):
        super().__init__()

        self.model = model
        self.running = False

        self.display_buffer = deque(maxlen=MUSE_FS * DISPLAY_SECONDS)
        self.display_time = deque(maxlen=MUSE_FS * DISPLAY_SECONDS)

        self.window_buffer = []
        self.first_lsl_timestamp = None
        self.vote_buffer = []

    def stop(self):
        self.running = False

    def run(self):
        try:
            self.status_changed.emit("Searching for EEG stream...")

            streams = resolve_byprop("type", "EEG", timeout=10)

            if not streams:
                self.error_occurred.emit(
                    "EEG stream was not found. Please run 'muselsl stream' first."
                )
                return

            inlet = StreamInlet(streams[0])
            info = inlet.info()

            channel_names = []
            desc = info.desc()
            channels = desc.child("channels")

            if channels.name():
                ch = channels.child("channel")

                while ch.name():
                    label_node = ch.child("label")

                    if label_node.name():
                        channel_names.append(label_node.child_value())
                    else:
                        channel_names.append(f"ch_{len(channel_names) + 1}")

                    ch = ch.next_sibling()

            if "AF7" in channel_names:
                af7_idx = channel_names.index("AF7")
                self.status_changed.emit(
                    f"Stream connected. AF7 channel found at index {af7_idx}."
                )
            else:
                # Common Muse order: TP9, AF7, AF8, TP10, AUX
                af7_idx = 1 if len(channel_names) > 1 else 0
                self.status_changed.emit(
                    f"AF7 was not found in metadata. Using fallback index {af7_idx}. Channels: {channel_names}"
                )

            self.running = True

            while self.running:
                sample, timestamp = inlet.pull_sample(timeout=0.1)

                if sample is None:
                    continue

                if self.first_lsl_timestamp is None:
                    self.first_lsl_timestamp = timestamp

                t_rel = timestamp - self.first_lsl_timestamp
                af7_val = float(sample[af7_idx])

                self.display_time.append(t_rel)
                self.display_buffer.append(af7_val)
                self.window_buffer.append(af7_val)

                self.sample_received.emit(af7_val, t_rel)

                if len(self.window_buffer) == MUSE_WINDOW_SIZE:
                    raw_window = np.array(self.window_buffer, dtype=float)

                    pred, prob, status, latency, modes, feats, vmd_input = predict_window_5s(
                    raw_window,
                    self.model,
                    input_fs=MUSE_FS,
                    )

                    self.window_prediction_ready.emit(
                        pred,
                        prob,
                        "5-second window classification",
                        latency,
                        t_rel,
                    )

                    self.vmd_ready.emit(modes, feats, vmd_input, status, t_rel)

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

        except Exception as e:
            tb = traceback.format_exc()
            self.error_occurred.emit(f"{e}\n\n{tb}")