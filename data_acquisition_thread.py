import csv
import os
import time
from datetime import datetime

from PyQt5.QtCore import QThread
from brainflow.board_shim import BoardShim, BrainFlowInputParams


class DataAcquisitionThread(QThread):
    def __init__(self, serial_port, board_id, letter, subject, pattern):
        super().__init__()
        self.board = None
        self.serial_port = serial_port
        self.board_id = board_id
        self.letter = letter
        self.subject = subject
        self.pattern = pattern
        self._running = True

    def run(self):
        params = BrainFlowInputParams()
        params.board_id = self.board_id
        params.serial_port = self.serial_port
        self.board = BoardShim(params.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Data/P300/Subject {self.subject}/{self.pattern}/{self.letter}/data_{timestamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Add a column for letters
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            headers = [channel for channel in eeg_channels]
            headers.append("Letter")  # Add a new header for the letter column
            writer.writerow(headers)

            try:
                while self._running:
                    data = self.board.get_board_data(250*1)
                    for i in range(len(data[0])):
                        row = [str(float(data[channel][i])) for channel in eeg_channels]  # Convert to float first, then to string
                        row.append(self.letter)  # Append the same letter for each row
                        writer.writerow(row)

                    time.sleep(1)
            finally:
                self.board.stop_stream()
                self.board.release_session()

    def stop(self):
        self._running = False
