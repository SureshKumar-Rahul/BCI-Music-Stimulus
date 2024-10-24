import csv
import os
import time
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal
from brainflow.board_shim import BoardShim, BrainFlowInputParams


class DataAcquisitionThread(QThread):

    def __init__(self, serial_port, board_id, subject, current_track_index):
        super().__init__()
        self.tracks_folder = "audio"  # Folder containing audio tracks
        self.tracks = self.load_tracks()
        self.board = None
        self.serial_port = serial_port
        self.board_id = board_id
        self.subject = subject
        self.current_track_index = current_track_index
        self._running = True

        plot_signal = pyqtSignal(str)
        self.latest_file_path = None  # Variable to store the latest file path


    def load_tracks(self):
        tracks = []
        for file in os.listdir(self.tracks_folder):
            if file.endswith(".mp3"):
                tracks.append(file)
        return tracks

    def run(self):

        params = BrainFlowInputParams()
        params.board_id = self.board_id
        params.serial_port = self.serial_port
        self.board = BoardShim(params.board_id, params)
        self.board.prepare_session()
        self.board.start_stream()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"Data/Music/Subject {self.subject}/{self.tracks[self.current_track_index]}/data_{timestamp}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.latest_file_path = filename

        with open(filename, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Add a column for letters
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            headers = [channel for channel in eeg_channels]
            headers.append("Track")  # Add a new header for the letter column
            writer.writerow(headers)


            try:
                while self._running:
                    data = self.board.get_board_data(250*1) ## (250 Hz @ 1sec) ##
                    for i in range(len(data[0])):
                        row = [str(float(data[channel][i])) for channel in
                               eeg_channels]  # Convert to float first, then to string
                        row.append(self.current_track_index)  # Append the same letter for each row
                        writer.writerow(row)


                    time.sleep(1)
            finally:
                self.board.stop_stream()
                self.board.release_session()

    def stop(self):
        self._running = False
