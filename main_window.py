from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QWidget, QPushButton, QComboBox, QLabel

from data_acquisition_thread import DataAcquisitionThread
from stimulus_grid import StimulusGrid


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.combo_patterns = None
        self.label_letters = None
        self.label_highlights = None
        self.label_rounds = None
        self.btn_start = None
        self.combo_letters = None
        self.combo_highlights = None
        self.combo_rounds = None
        self.stimulus_grid = None
        self.data_acquisition_thread = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('P300 Speller Controller')
        self.setGeometry(300, 300, 600, 600)

        # Select Letter dropdown
        self.label_letters = QLabel('Select Letter:', self)
        self.label_letters.setGeometry(30, 50, 150, 30)
        self.combo_letters = QComboBox(self)
        self.combo_letters.setGeometry(200, 50, 100, 30)
        self.combo_letters.addItems(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                     'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

        # Select No. of Highlights dropdown
        self.label_highlights = QLabel('Select No. of Highlights:', self)
        self.label_highlights.setGeometry(30, 100, 200, 30)
        self.combo_highlights = QComboBox(self)
        self.combo_highlights.setGeometry(200, 100, 100, 30)
        self.combo_highlights.addItems(['1', '2', '3', '4', '5'])

        # Select No. of Rounds dropdown
        self.label_rounds = QLabel('Select No. of Rounds:', self)
        self.label_rounds.setGeometry(30, 150, 200, 30)
        self.combo_rounds = QComboBox(self)
        self.combo_rounds.setGeometry(200, 150, 100, 30)
        self.combo_rounds.addItems(['1', '2', '3', '4', '5'])

        # Select the Pattern
        self.label_pattern = QLabel('Select the Highlight Method:', self)
        self.label_pattern.setGeometry(30, 200, 200, 30)
        self.combo_patterns = QComboBox(self)
        self.combo_patterns.setGeometry(200, 200, 100, 30)
        self.combo_patterns.addItems(['RC', 'SC'])

        # Subject
        self.subject_label = QLabel('Subject:', self)
        self.subject_label.setGeometry(30, 250, 200, 30)
        self.subject = QComboBox(self)
        self.subject.setGeometry(200, 250, 100, 30)
        self.subject.addItems(['0', '1', '2', '3', '4', '5'])

        # Start button
        self.btn_start = QPushButton('Start', self)
        self.btn_start.setGeometry(250, 350, 100, 30)
        self.btn_start.clicked.connect(self.start_p300_speller)

        self.show()

    @pyqtSlot()
    def start_p300_speller(self):
        selected_letter = self.combo_letters.currentText()  # Get the selected letter from the combobox
        selected_highlight = int(self.combo_highlights.currentText())
        selected_round = int(self.combo_rounds.currentText())
        selected_pattern = self.combo_patterns.currentText()
        selected_subject = self.subject.currentText()
        self.stimulus_grid = StimulusGrid(selected_letter, selected_highlight, selected_round, selected_pattern)
        self.stimulus_grid.showMaximized()
        # Pass the selected letter to the constructor of DataAcquisitionThread
        self.data_acquisition_thread = DataAcquisitionThread(serial_port="COM5",
                                                             board_id=2,
                                                             letter=selected_letter,
                                                             subject=selected_subject,
                                                             pattern=selected_pattern)

        self.data_acquisition_thread.start()
        self.stimulus_grid.highlighting_finished.connect(self.close_all_windows)
        self.stimulus_grid.closing_signal.connect(self.close_all_windows)

    def close_all_windows(self):
        # Stop any running threads or timers here
        if self.data_acquisition_thread:
            # Schedule the stopping of data_acquisition_thread after 5 seconds
            QTimer.singleShot(5000, self.stop_data_acquisition_thread)
        if self.stimulus_grid:
            self.stimulus_grid.close()
        self.close()

    def stop_data_acquisition_thread(self):
        if self.data_acquisition_thread:
            self.data_acquisition_thread.stop()
