APP_STYLESHEET = """
            QWidget#Central {
                background-color: #0f1117;
                color: #e5e7eb;
                font-family: Segoe UI, Arial;
            }

            QFrame#Card {
                background-color: #171a23;
                border: 1px solid #2a2f3a;
                border-radius: 16px;
            }

            QLabel#AppTitle {
                color: #f9fafb;
                font-size: 22px;
                font-weight: 800;
            }

            QLabel#AppSubtitle {
                color: #9ca3af;
                font-size: 12px;
            }

            QLabel#CardTitle {
                color: #f3f4f6;
                font-size: 14px;
                font-weight: 800;
            }

            QLabel#MetricLabel {
                color: #9ca3af;
                font-size: 12px;
            }

            QLabel#MetricValue {
                color: #f9fafb;
                font-size: 13px;
                font-weight: 700;
            }

            QLabel#PredictionBadge {
                padding: 10px;
                border-radius: 14px;
                font-size: 22px;
                font-weight: 900;
                background-color: #374151;
                color: #f9fafb;
            }

            QPushButton {
                border: none;
                border-radius: 12px;
                padding: 10px 16px;
                font-size: 13px;
                font-weight: 800;
                color: white;
                background-color: #374151;
            }

            QPushButton:hover {
                background-color: #4b5563;
            }

            QPushButton:disabled {
                background-color: #1f2937;
                color: #6b7280;
            }

            QPushButton#StartButton {
                background-color: #16a34a;
            }

            QPushButton#StartButton:hover {
                background-color: #22c55e;
            }

            QPushButton#StopButton {
                background-color: #dc2626;
            }

            QPushButton#StopButton:hover {
                background-color: #ef4444;
            }

            QComboBox, QSpinBox, QLineEdit {
                background-color: #0b0d12;
                color: #f3f4f6;
                border: 1px solid #374151;
                border-radius: 8px;
                padding: 6px;
                font-size: 12px;
            }

            QCheckBox {
                color: #f3f4f6;
                font-size: 12px;
            }

            QTableWidget {
                background-color: #0b0d12;
                color: #f3f4f6;
                border: 1px solid #2a2f3a;
                border-radius: 10px;
                gridline-color: #2a2f3a;
                font-size: 12px;
                selection-background-color: #2563eb;
            }

            QHeaderView::section {
                background-color: #1f2937;
                color: #f9fafb;
                font-weight: 800;
                padding: 5px;
                border: none;
                border-right: 1px solid #374151;
            }

            QTableWidget::item {
                padding: 4px;
            }
        """