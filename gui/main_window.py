from collections import deque
from datetime import datetime
from pathlib import Path
import csv

import numpy as np

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QFileDialog,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QLineEdit
)

import pyqtgraph as pg

from config import (
    MUSE_FS,
    DISPLAY_SECONDS,
    MODEL_FS,
    WINDOW_SEC,
    K,
    DEFAULT_CSV_FS,
)

from core.classifier import format_pred, load_classifier_model
from acquisition.muse_stream import MuseStreamWorker
from acquisition.csv_replay import CSVReplayWorker
from gui.styles import APP_STYLESHEET


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Real-Time EEG Consciousness Detection - VMD + Ensemble Bagged Tree"
        )
        self.resize(1600, 950)

        try:
            self.model = load_classifier_model()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Model Loading Error",
                f"Failed to load model:\n{e}",
            )
            raise

        self.worker = None
        self.csv_path = None

        self.is_muse_realtime_recording = False
        self.realtime_raw_records = []
        self.last_saved_raw_csv = None

        self.x_data = deque()
        self.y_data = deque()

        self.last_modes = None
        self.last_features = None
        self.last_filtered = None

        self._build_ui()

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.refresh_plot)
        self.plot_timer.start(80)

    def make_card(self, title_text):
        card = QFrame()
        card.setObjectName("Card")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        title = QLabel(title_text)
        title.setObjectName("CardTitle")

        layout.addWidget(title)

        return card, layout

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("Central")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        self.setStyleSheet(APP_STYLESHEET)

        # =====================================================
        # HEADER
        # =====================================================
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)

        title_col = QVBoxLayout()
        title_col.setSpacing(1)

        app_title = QLabel("Real-Time EEG Consciousness Detection")
        app_title.setObjectName("AppTitle")

        app_subtitle = QLabel(
            f"Muse 2 AF7 • 5s Window • VMD K={K} • Ensemble Bagged Tree • Majority Voting 15s"
        )
        app_subtitle.setObjectName("AppSubtitle")

        title_col.addWidget(app_title)
        title_col.addWidget(app_subtitle)

        self.btn_start = QPushButton("START MONITORING")
        self.btn_start.setObjectName("StartButton")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setMaximumHeight(48)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("StopButton")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setMaximumHeight(48)
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_stream)
        self.btn_stop.clicked.connect(self.stop_stream)

        # =====================================================
        # COMPACT INPUT SOURCE CONTROLS
        # =====================================================
        source_controls = QHBoxLayout()
        source_controls.setSpacing(6)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["Muse 2 Real-Time", "CSV Replay"])
        self.source_combo.currentIndexChanged.connect(self.on_source_changed)
        self.source_combo.setFixedWidth(155)
        self.source_combo.setFixedHeight(36)

        self.btn_browse_csv = QPushButton("CSV")
        self.btn_browse_csv.clicked.connect(self.browse_csv_file)
        self.btn_browse_csv.setEnabled(False)
        self.btn_browse_csv.setFixedWidth(58)
        self.btn_browse_csv.setFixedHeight(36)

        self.csv_fs_spin = QSpinBox()
        self.csv_fs_spin.setMinimum(1)
        self.csv_fs_spin.setMaximum(5000)
        self.csv_fs_spin.setValue(DEFAULT_CSV_FS)
        self.csv_fs_spin.setEnabled(False)
        self.csv_fs_spin.setFixedWidth(72)
        self.csv_fs_spin.setFixedHeight(36)

        self.realtime_csv_check = QCheckBox("Real-time")
        self.realtime_csv_check.setChecked(True)
        self.realtime_csv_check.setEnabled(False)

        self.csv_path_box = QLineEdit()
        self.csv_path_box.setReadOnly(True)
        self.csv_path_box.setPlaceholderText("No CSV selected")
        self.csv_path_box.setFixedWidth(260)
        self.csv_path_box.setFixedHeight(36)

        source_controls.addWidget(self.source_combo)
        source_controls.addWidget(self.btn_browse_csv)
        source_controls.addWidget(QLabel("FS:"))
        source_controls.addWidget(self.csv_fs_spin)
        source_controls.addWidget(self.realtime_csv_check)
        source_controls.addWidget(self.csv_path_box)

        header_layout.addLayout(title_col, stretch=1)
        header_layout.addLayout(source_controls)
        header_layout.addWidget(self.btn_start)
        header_layout.addWidget(self.btn_stop)

        root.addWidget(header)

        # =====================================================
        # RAW EEG CARD
        # =====================================================
        raw_card, raw_layout = self.make_card("Real-Time EEG Signal - AF7 Channel")
        raw_layout.setContentsMargins(14, 10, 14, 12)
        raw_layout.setSpacing(8)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#05070a")
        self.plot_widget.setLabel("left", "Amplitude")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.22)
        self.plot_widget.setMinimumHeight(190)
        self.plot_widget.setMaximumHeight(240)

        self.plot_widget.getAxis("left").setPen(pg.mkPen("#9ca3af"))
        self.plot_widget.getAxis("bottom").setPen(pg.mkPen("#9ca3af"))
        self.plot_widget.getAxis("left").setTextPen(pg.mkPen("#9ca3af"))
        self.plot_widget.getAxis("bottom").setTextPen(pg.mkPen("#9ca3af"))

        pen = pg.mkPen(color=(45, 212, 191), width=1.4)
        self.curve = self.plot_widget.plot(pen=pen)

        raw_layout.addWidget(self.plot_widget)
        root.addWidget(raw_card)

        # =====================================================
        # BOTTOM AREA
        # =====================================================
        bottom = QHBoxLayout()
        bottom.setSpacing(12)

        # =====================================================
        # VMD CARD
        # =====================================================
        vmd_card, vmd_layout = self.make_card("Signal Decomposition using VMD")
        vmd_layout.setContentsMargins(14, 10, 14, 12)
        vmd_layout.setSpacing(6)

        self.mode_plots = []
        self.mode_curves = []

        colors = [
            (245, 158, 11),
            (34, 197, 94),
            (236, 72, 153),
        ]

        for i in range(K):
            p = pg.PlotWidget()
            p.setBackground("#05070a")
            p.setLabel("left", "Amp")
            p.setLabel("bottom", "Time", units="s")
            p.showGrid(x=True, y=True, alpha=0.18)
            p.setMinimumHeight(115)
            p.setMaximumHeight(140)

            p.getAxis("left").setPen(pg.mkPen("#9ca3af"))
            p.getAxis("bottom").setPen(pg.mkPen("#9ca3af"))
            p.getAxis("left").setTextPen(pg.mkPen("#9ca3af"))
            p.getAxis("bottom").setTextPen(pg.mkPen("#9ca3af"))

            p.setTitle(f"Mode {i + 1}", color="#e5e7eb", size="10pt")

            curve = p.plot(
                pen=pg.mkPen(color=colors[i % len(colors)], width=1.2)
            )

            self.mode_plots.append(p)
            self.mode_curves.append(curve)
            vmd_layout.addWidget(p)

        bottom.addWidget(vmd_card, stretch=2)

        # =====================================================
        # RIGHT PANEL
        # =====================================================
        panel_card, panel_layout = self.make_card("Result and Feature Panel")
        panel_layout.setContentsMargins(14, 10, 14, 12)
        panel_layout.setSpacing(8)

        self.lbl_status = QLabel("Status: Idle")
        self.lbl_status.setObjectName("MetricValue")

        self.lbl_pred = QLabel("Prediction: -")
        self.lbl_pred.setObjectName("PredictionBadge")
        self.lbl_pred.setAlignment(Qt.AlignCenter)
        self.lbl_pred.setMinimumHeight(54)
        self.lbl_pred.setMaximumHeight(70)

        self.lbl_window_pred = QLabel("Current 5s window: -")
        self.lbl_window_pred.setObjectName("MetricValue")
        self.lbl_window_pred.setAlignment(Qt.AlignCenter)

        panel_layout.addWidget(self.lbl_status)
        panel_layout.addWidget(self.lbl_pred)
        panel_layout.addWidget(self.lbl_window_pred)

        metric_grid = QGridLayout()
        metric_grid.setHorizontalSpacing(14)
        metric_grid.setVerticalSpacing(4)

        self.lbl_prob_name = QLabel("Conscious probability")
        self.lbl_prob_name.setObjectName("MetricLabel")
        self.lbl_prob = QLabel("-")
        self.lbl_prob.setObjectName("MetricValue")

        self.lbl_lat_name = QLabel("Latency")
        self.lbl_lat_name.setObjectName("MetricLabel")
        self.lbl_lat = QLabel("-")
        self.lbl_lat.setObjectName("MetricValue")

        self.lbl_win_name = QLabel("Voting window")
        self.lbl_win_name.setObjectName("MetricLabel")
        self.lbl_win = QLabel("-")
        self.lbl_win.setObjectName("MetricValue")

        self.lbl_vmd_name = QLabel("VMD window")
        self.lbl_vmd_name.setObjectName("MetricLabel")
        self.lbl_vmd = QLabel("-")
        self.lbl_vmd.setObjectName("MetricValue")
        self.lbl_vmd.setWordWrap(True)

        metric_grid.addWidget(self.lbl_prob_name, 0, 0)
        metric_grid.addWidget(self.lbl_prob, 0, 1)

        metric_grid.addWidget(self.lbl_lat_name, 1, 0)
        metric_grid.addWidget(self.lbl_lat, 1, 1)

        metric_grid.addWidget(self.lbl_win_name, 2, 0)
        metric_grid.addWidget(self.lbl_win, 2, 1)

        metric_grid.addWidget(self.lbl_vmd_name, 3, 0)
        metric_grid.addWidget(self.lbl_vmd, 3, 1)

        panel_layout.addLayout(metric_grid)

        feature_title = QLabel("Latest VMD Window Feature Values")
        feature_title.setObjectName("CardTitle")
        panel_layout.addWidget(feature_title)

        self.feature_table = QTableWidget()
        self.feature_table.setRowCount(4)
        self.feature_table.setColumnCount(4)
        self.feature_table.setHorizontalHeaderLabels(
            ["Feature", "Mode 1", "Mode 2", "Mode 3"]
        )

        feature_names = [
            "Sample Entropy",
            "Spectral Entropy",
            "Skewness",
            "Kurtosis",
        ]

        for row, feature_name in enumerate(feature_names):
            item = QTableWidgetItem(feature_name)
            item.setTextAlignment(Qt.AlignCenter)
            self.feature_table.setItem(row, 0, item)

            for col in range(1, 4):
                item = QTableWidgetItem("-")
                item.setTextAlignment(Qt.AlignCenter)
                self.feature_table.setItem(row, col, item)

        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.setMinimumHeight(190)
        self.feature_table.setMaximumHeight(230)
        self.feature_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.feature_table.resizeRowsToContents()

        panel_layout.addWidget(self.feature_table)
        panel_layout.addStretch(1)

        bottom.addWidget(panel_card, stretch=1)
        root.addLayout(bottom, stretch=1)

    # =====================================================
    # GUI EVENTS
    # =====================================================
    def on_source_changed(self):
        is_csv = self.source_combo.currentText() == "CSV Replay"

        self.btn_browse_csv.setEnabled(is_csv)
        self.csv_fs_spin.setEnabled(is_csv)
        self.realtime_csv_check.setEnabled(is_csv)

        if not is_csv:
            self.csv_path_box.setText("")

    def browse_csv_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if path:
            self.csv_path = path
            self.csv_path_box.setText(path)

    def reset_display(self):
        self.x_data.clear()
        self.y_data.clear()

        self.last_modes = None
        self.last_features = None
        self.last_filtered = None

        self.curve.setData([], [])

        for curve in self.mode_curves:
            curve.setData([], [])

        for row in range(4):
            for col in range(1, 4):
                item = QTableWidgetItem("-")
                item.setTextAlignment(Qt.AlignCenter)
                self.feature_table.setItem(row, col, item)

        self.lbl_status.setText("Status: Idle")
        self.lbl_pred.setText("Waiting for data...")
        self.lbl_window_pred.setText("Current 5s window: -")
        self.lbl_prob.setText("-")
        self.lbl_lat.setText("-")
        self.lbl_win.setText("-")
        self.lbl_vmd.setText("-")
    
    def save_muse_raw_csv(self):
        if not self.is_muse_realtime_recording:
            return None

        if len(self.realtime_raw_records) == 0:
            return None

        output_dir = Path("recorded_raw")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"muse2_raw_af7_{timestamp}.csv"

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec", "AF7"])
            writer.writerows(self.realtime_raw_records)

        self.last_saved_raw_csv = str(output_path)
        return str(output_path)
    
    def on_worker_finished(self):
        self.worker = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def start_stream(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.reset_display()

        self.realtime_raw_records = []
        self.last_saved_raw_csv = None
        self.is_muse_realtime_recording = False

        source = self.source_combo.currentText()

        if source == "Muse 2 Real-Time":
            self.lbl_status.setText("Status: Starting Muse 2 stream...")
            self.is_muse_realtime_recording = True
            self.worker = MuseStreamWorker(self.model)

        elif source == "CSV Replay":
            self.is_muse_realtime_recording = False
            if not self.csv_path:
                QMessageBox.warning(
                    self,
                    "CSV File Missing",
                    "Please select a CSV file first.",
                )
                self.btn_start.setEnabled(True)
                self.btn_stop.setEnabled(False)
                return

            csv_fs = int(self.csv_fs_spin.value())
            realtime = self.realtime_csv_check.isChecked()

            self.lbl_status.setText("Status: Starting CSV replay...")

            self.worker = CSVReplayWorker(
                model=self.model,
                csv_path=self.csv_path,
                csv_fs=csv_fs,
                signal_column="AF7",
                realtime=realtime,
            )

            self.worker.finished_replay.connect(self.on_csv_finished)

        else:
            QMessageBox.critical(
                self,
                "Input Source Error",
                "Unknown input source.",
            )
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        self.worker.sample_received.connect(self.on_sample_received)
        self.worker.status_changed.connect(self.on_status_changed)
        self.worker.window_prediction_ready.connect(self.on_window_prediction_ready)
        self.worker.prediction_ready.connect(self.on_prediction_ready)
        self.worker.vmd_ready.connect(self.on_vmd_ready)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.start()

    def stop_stream(self):
        was_muse_realtime = self.is_muse_realtime_recording

        worker = self.worker
        self.worker = None

        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass

            if not worker.wait(5000):
                print("Worker did not stop in time. Terminating thread...")
                worker.terminate()
                worker.wait(2000)

        saved_path = None

        if was_muse_realtime:
            saved_path = self.save_muse_raw_csv()

        self.is_muse_realtime_recording = False

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        if saved_path is not None:
            self.lbl_status.setText(f"Status: Stopped | Raw CSV saved: {saved_path}")
        else:
            self.lbl_status.setText("Status: Stopped")

    def on_csv_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Status: CSV replay finished")

    def on_sample_received(self, af7_val, t_rel):
        t_rel = float(t_rel)
        af7_val = float(af7_val)

        self.x_data.append(t_rel)
        self.y_data.append(af7_val)

        cutoff = t_rel - float(DISPLAY_SECONDS)

        while len(self.x_data) > 0 and self.x_data[0] < cutoff:
            self.x_data.popleft()
            self.y_data.popleft()

        if self.is_muse_realtime_recording:
            self.realtime_raw_records.append([t_rel, af7_val])

    def on_status_changed(self, msg):
        self.lbl_status.setText(f"Status: {msg}")

    def on_vmd_ready(self, modes, feats, filtered, status, win_end_sec):
        self.last_modes = modes
        self.last_features = feats
        self.last_filtered = filtered

        self.lbl_vmd.setText(f"{status} | {win_end_sec:.2f} s")

        if feats is None:
            for row in range(4):
                for col in range(1, 4):
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    self.feature_table.setItem(row, col, item)
            return

        feature_rows = [
            ("sample_entropy", "Sample Entropy"),
            ("spectral_entropy", "Spectral Entropy"),
            ("skewness", "Skewness"),
            ("kurtosis", "Kurtosis"),
        ]

        for row, (feat_key, _) in enumerate(feature_rows):
            for mode_idx in range(1, K + 1):
                key = f"mode{mode_idx}_{feat_key}"
                val = feats.get(key, np.nan)

                item_text = f"{val:.6f}" if np.isfinite(val) else "NaN"

                item = QTableWidgetItem(item_text)
                item.setTextAlignment(Qt.AlignCenter)

                self.feature_table.setItem(row, mode_idx, item)

        self.feature_table.resizeRowsToContents()
        self.update_vmd_plots()
    
    def on_window_prediction_ready(self, pred, prob, status, latency, win_end_sec):
        pred_str = format_pred(pred)
        prob_str = "-" if prob is None else f"{prob:.4f}"

        self.lbl_window_pred.setText(
            f"Current 5s window: {pred_str} | Prob: {prob_str} | {win_end_sec:.2f} s"
        )

    def on_prediction_ready(self, pred, prob, status, latency, win_end_sec):
        pred_str = format_pred(pred)
        prob_str = "-" if prob is None else f"{prob:.4f}"

        self.lbl_status.setText(f"Status: {status}")
        self.lbl_pred.setText(pred_str)
        self.lbl_prob.setText(prob_str)
        self.lbl_lat.setText(f"{latency:.4f} s")
        self.lbl_win.setText(f"{win_end_sec:.2f} s")

        if pred is None:
            self.lbl_pred.setStyleSheet("""
                QLabel#PredictionBadge {
                    padding: 16px;
                    border-radius: 14px;
                    font-size: 24px;
                    font-weight: 800;
                    background-color: #92400e;
                    color: #fffbeb;
                }
            """)
        elif int(pred) == 1:
            self.lbl_pred.setStyleSheet("""
                QLabel#PredictionBadge {
                    padding: 16px;
                    border-radius: 14px;
                    font-size: 24px;
                    font-weight: 800;
                    background-color: #166534;
                    color: #dcfce7;
                }
            """)
        else:
            self.lbl_pred.setStyleSheet("""
                QLabel#PredictionBadge {
                    padding: 16px;
                    border-radius: 14px;
                    font-size: 24px;
                    font-weight: 800;
                    background-color: #991b1b;
                    color: #fee2e2;
                }
            """)

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.stop_stream()

    # =====================================================
    # PLOT UPDATE
    # =====================================================
    def refresh_plot(self):
        if len(self.x_data) == 0:
            return

        x = np.array(self.x_data, dtype=float)
        y = np.array(self.y_data, dtype=float)

        y_disp = y - np.mean(y)

        self.curve.setData(x, y_disp)

        x_max = x[-1]
        x_min = max(0.0, x_max - float(DISPLAY_SECONDS))

        if x_max <= x_min:
            x_max = x_min + 1.0

        self.plot_widget.setXRange(x_min, x_max, padding=0)

        y_visible = y_disp

        if len(y_visible) > 0:
            y_min = np.min(y_visible)
            y_max = np.max(y_visible)

            if y_min == y_max:
                y_min -= 1
                y_max += 1

            pad = 0.1 * (y_max - y_min)

            self.plot_widget.setYRange(
                y_min - pad,
                y_max + pad,
                padding=0,
            )

    def update_vmd_plots(self):
        if self.last_modes is None:
            return

        modes = np.asarray(self.last_modes)

        if modes.ndim != 2:
            return

        t = np.arange(modes.shape[1]) / MODEL_FS

        for i in range(min(K, modes.shape[0])):
            mode = modes[i]
            mode_disp = mode - np.mean(mode)

            self.mode_curves[i].setData(t, mode_disp)
            self.mode_plots[i].setXRange(0, WINDOW_SEC, padding=0)

            y_min = np.min(mode_disp)
            y_max = np.max(mode_disp)

            if y_min == y_max:
                y_min -= 1
                y_max += 1

            pad = 0.1 * (y_max - y_min)

            self.mode_plots[i].setYRange(
                y_min - pad,
                y_max + pad,
                padding=0,
            )

    def closeEvent(self, event):
        self.stop_stream()
        event.accept()