import csv
import os
import time
from datetime import datetime

from PyQt5.QtCore import QThread, pyqtSignal
from brainflow.board_shim import BoardShim, BrainFlowInputParams


class DataAcquisitionThread(QThread):
    # Emitted once the recording file is fully written and closed.
    # Carries (csv_path, track_index) so the receiver plots under the correct track.
    # Must be a class attribute: a pyqtSignal assigned inside __init__ does not work.
    plot_signal = pyqtSignal(str, int)

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
        self.latest_file_path = None  # Path of the CSV written by this run


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
                    # Pull up to 250 latest samples. The board streams at 125 Hz,
                    # so roughly 125 samples accumulate per 1 s loop.
                    data = self.board.get_board_data(250)
                    for i in range(len(data[0])):
                        row = [str(float(data[channel][i])) for channel in
                               eeg_channels]  # Convert to float first, then to string
                        row.append(self.current_track_index)  # Tag every row with the track index
                        writer.writerow(row)

                    time.sleep(1)
            finally:
                self.board.stop_stream()
                self.board.release_session()

        # The CSV is now closed and flushed; notify listeners so they can plot it.
        self.plot_signal.emit(self.latest_file_path, self.current_track_index)

    def stop(self):
        self._running = False
        self.wait()  # Block until run() finishes so the board session is released cleanly
