import csv
import sys
import random
import argparse
import time

import numpy as np
import pandas as pd

from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer, Qt, QThread
from brainflow import DataFilter
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QDesktopWidget


class StimulusGrid(QWidget):
    closing_signal = pyqtSignal()
    highlighting_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.labels = None
        self.stimuli = None
        self.grid = None
        self._flags = None
        self.initUI()

    def initUI(self):
        self.grid = QGridLayout()
        self.setLayout(self.grid)

        # Create labels for each stimulus
        self.stimuli = ['A', 'B', 'C', 'D', 'E', 'F',
                        'G', 'H', 'I', 'J', 'K', 'L',
                        'M', 'N', 'O', 'P', 'Q', 'R',
                        'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z', '1', '2', '3', '4',
                        '5', '6', '7', '8', '9', '-']

        self.labels = []
        for row in range(6):
            for col in range(6):
                if row * 6 + col < len(self.stimuli):
                    label = QLabel(self.stimuli[row * 6 + col])  # Assign the letter to the label
                    label.setAlignment(Qt.AlignCenter)
                    self.grid.addWidget(label, row, col)
                    self.labels.append(label)

        self.setWindowTitle('Farewell and Donchin\'s P300 Speller Paradigm')
        self.setStyleSheet("background-color: black; color: white;")
        desktop = QApplication.desktop()  # Get desktop widget
        taskbar_height = desktop.height() - desktop.availableGeometry().height()  # Calculate height of the taskbar
        self.setGeometry(0, 0, desktop.screenGeometry().width(), desktop.screenGeometry().height() - taskbar_height)  # Set window geometry to screen size minus taskbar height
        self.show()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.highlight_random_location)
        self.timer.start(1000)  # Update every 1000 ms (1 second)

    def closeEvent(self, event):
        self.closing_signal.emit()

    def highlight_random_location(self):
        # Reset all labels to default style
        for label in self.labels:
            label.setStyleSheet('color: white')

        # Shuffle the list of labels
        random.shuffle(self.labels)

        # Highlight all labels with a delay of 1 second between each highlight
        for label in self.labels:
            label.setStyleSheet('color: red')
            time.sleep(1)  # Wait for 1 second
            QApplication.processEvents()  # Process pending events to update UI
            label.setStyleSheet('color: white')  # Reset color to original

        self.highlighting_finished.emit()

    def resizeEvent(self, event):
        # Get the current font from the first label
        current_font = self.labels[0].font()

        # Check if the current font size is smaller or larger than the desired sizes
        if current_font.pointSize() < 80:
            # Increase label font size when the window is maximized
            current_font.setPointSize(80)
        elif current_font.pointSize() > 20:
            # Reset label font size when the window is restored
            current_font.setPointSize(70)

        # Apply the font changes to all labels
        for label in self.labels:
            label.setFont(current_font)


class DataAcquisitionThread(QThread):
    def __init__(self):
        super().__init__()
        self.sample_index = 0  # Initialize sample index
        self._running = True  # Flag to control the loop

    def run(self):
        parser = argparse.ArgumentParser()
        # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
        parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                            default=0)
        parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
        parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                            default=0)
        parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
        parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
        parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
        parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
        parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
        parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                            required=True)
        parser.add_argument('--file', type=str, help='file', required=False, default='')
        parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                            required=False, default=BoardIds.NO_BOARD)
        args = parser.parse_args()

        params = BrainFlowInputParams()
        params.ip_port = args.ip_port
        params.serial_port = args.serial_port
        params.mac_address = args.mac_address
        params.other_info = args.other_info
        params.serial_number = args.serial_number
        params.ip_address = args.ip_address
        params.ip_protocol = args.ip_protocol
        params.timeout = args.timeout
        params.file = args.file
        params.master_board = args.master_board
        board_id = BoardIds.SYNTHETIC_BOARD.value
        args.board_id = board_id
        self.board = BoardShim(args.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        # Generate filename with current timestamp
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = f"data_{self.timestamp}.csv"

        with open(self.filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the headers
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
            headers = [channel for channel in eeg_channels]
            writer.writerow(headers)

            try:
                while self._running:
                    data = self.board.get_board_data()  # get the latest raw data
                    # Write the data for the current sample index
                    for i in range(len(data[0])):  # Iterate through the number of values in each channel
                        row = [data[channel][i] for channel in range(32)]  # Get the ith value for each channel
                        writer.writerow(row)
                    # DataFilter.write_file(data, 'test.csv', 'a')  # use 'a' for append mode

                    # Increment the sample index
                    self.sample_index += 1

                    time.sleep(1)  # Adjust the interval as needed
            finally:
                self.board.stop_stream()
                self.board.release_session()

    def stop(self):
        self._running = False
        self.wait()  # Wait for the thread to finish


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.stimulus_grid = None
        self.data_acquisition_thread = None

    def initUI(self):
        self.setWindowTitle('P300 Speller Controller')
        self.setGeometry(100, 100, 300, 150)

        self.btn_start = QPushButton('Start', self)
        self.btn_start.clicked.connect(self.start_p300_speller)
        self.btn_start.setGeometry(50, 50, 100, 30)

        self.btn_start.move((self.width() - self.btn_start.width()) // 2, 50)
        # Set the color of the start button to red
        self.btn_start.setStyleSheet("background-color: red; color: white;")

        self.show()

    def start_p300_speller(self):
        self.stimulus_grid = StimulusGrid()
        self.data_acquisition_thread = DataAcquisitionThread()
        self.data_acquisition_thread.start()
        self.stimulus_grid.closing_signal.connect(self.stop_data_acquisition)
        self.stimulus_grid.closing_signal.connect(self.close_all_windows)
        self.stimulus_grid.highlighting_finished.connect(self.stop_data_acquisition)
        self.stimulus_grid.highlighting_finished.connect(self.close_all_windows)

    def stop_data_acquisition(self):
        if self.data_acquisition_thread:
            self.data_acquisition_thread.stop()

    def close_all_windows(self):
        if self.stimulus_grid:
            self.stimulus_grid.close()
        if self.data_acquisition_thread:
            self.data_acquisition_thread.stop()
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
