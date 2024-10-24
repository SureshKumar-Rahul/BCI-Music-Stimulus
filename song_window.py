import os
import sys

import matplotlib.pyplot as plt
import mne
import pandas as pd
import pygame
from PyQt5 import QtGui
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QComboBox
from brainflow import BoardIds

from data_acquisition_thread_music import DataAcquisitionThread


def clean_data(df):
    # Convert DataFrame to MNE RawArray
    ch_names = [str(i) for i in range(1, 17)]
    ch_types = ['eeg'] * 16
    sfreq = 125  # Sampling frequency (Hz)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(df[ch_names].T.values, info)

    # Apply filtering (1-40 Hz band-pass filter)
    raw.filter(1., 40., fir_design='firwin')

    # Detect and interpolate bad channels
    raw.interpolate_bads(reset_bads=True)

    # Convert MNE RawArray back to DataFrame
    cleaned_data = raw.get_data().T
    df_cleaned = pd.DataFrame(cleaned_data, columns=ch_names)

    return df_cleaned


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Data Collection with Music Player")
        self.setGeometry(300, 300, 800, 800)

        self.tracks_folder = "audio"  # Folder containing audio tracks
        self.image_folder = "audio/images"  # Folder containing track images
        self.tracks = self.load_tracks()
        self.current_track_index = 0
        self.break_duration = 15  # Break duration in seconds
        self.data_acquisition_thread = None
        self.break_timer = QTimer(self)
        self.track_timer = QTimer(self)
        self.data_acquisition_thread = None  # Initialize EEG data acquisition thread
        self.song_duration = 30
        self.setup_ui()

        if self.data_acquisition_thread:
            self.data_acquisition_thread.plot_signal.connect(self.generate_and_save_plot)

    def setup_ui(self):
        # Play button
        self.play_button = QPushButton("Play", self)
        self.play_button.clicked.connect(self.play_music)
        self.play_button.setGeometry(50, 50, 100, 50)

        # Pause button
        self.pause_button = QPushButton("Pause", self)
        self.pause_button.clicked.connect(self.pause_music)
        self.pause_button.setGeometry(200, 50, 100, 50)

        # Stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.stop_music)
        self.stop_button.setGeometry(350, 50, 100, 50)

        # Subject label and combo box
        self.subject_label = QLabel('Subject:', self)
        self.subject_label.setGeometry(50, 120, 100, 30)
        self.subject = QComboBox(self)
        self.subject.setGeometry(150, 120, 100, 30)
        self.subject.addItems(['0', '1', '2', '3', '4', '5'])

        self.song_duration_label = QLabel('Song Duration:', self)
        self.song_duration_label.setGeometry(50, 200, 100, 30)
        self.song_time = QComboBox(self)
        self.song_time.setGeometry(150, 200, 100, 30)
        self.song_time.addItems(['60'])

        # Status label
        self.status_label = QLabel("Ready", self)
        self.status_label.setGeometry(50, 300, 500, 30)

        # Break label
        self.break_label = QLabel("", self)
        self.break_label.setGeometry(50, 350, 500, 30)

        # Image label
        self.image_label = QLabel(self)
        self.image_label.setGeometry(200, 400, 400, 400)  # Adjust position and size as needed

        self.break_timer.timeout.connect(self.update_break_time)
        self.track_timer.timeout.connect(self.play_next_track)

    def load_tracks(self):
        tracks = []
        for file in os.listdir(self.tracks_folder):
            if file.endswith(".mp3"):
                tracks.append(os.path.join(self.tracks_folder, file))

        return tracks

    def play_music(self):
        pygame.mixer.init()
        self.play_next_track()

    def play_next_track(self):
        if self.current_track_index < len(self.tracks):
            self.break_label.clear()
            self.track_timer.stop()
            track = self.tracks[self.current_track_index]
            image_path = os.path.join(self.image_folder, f"img{self.current_track_index}.jpg")
            self.set_image(image_path)
            pygame.mixer.music.load(track)
            pygame.mixer.music.play()
            self.status_label.setText(f"Playing: {track}")
            # Start EEG data acquisition when a track starts playing
            if self.data_acquisition_thread is None or not self.data_acquisition_thread.isRunning():
                self.data_acquisition_thread = DataAcquisitionThread(serial_port="COM9",
                                                                     board_id=BoardIds.CYTON_DAISY_BOARD.value,
                                                                     subject=int(self.subject.currentText()),
                                                                     current_track_index=self.current_track_index)

                self.data_acquisition_thread.start()

            self.current_track_index += 1
            self.song_duration = int(self.song_time.currentText())
            self.break_timer.start(self.song_duration * 1000)  # Start break timer


        else:
            self.status_label.setText("All tracks played")
            self.break_label.setText("Ready for another round?")
            if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
                self.data_acquisition_thread.stop()

    def generate_and_save_plot(self, file_path):
        output_base_dir = 'EEG_Plots_RAW'
        os.makedirs(output_base_dir, exist_ok=True)
        df = pd.read_csv(file_path)
        df_cleaned = clean_data(df)
        fig, axes = plt.subplots(nrows=16, ncols=1, figsize=(12, 24), sharex=True)
        for i in range(1, 17):
            ax = axes[i - 1]
            ax.plot(df_cleaned.index, df_cleaned[str(i)], label=f'Electrode {i}')
            ax.set_ylabel(f'Electrode {i}')
            ax.legend(loc='upper right')
        axes[-1].set_xlabel('Sample Index')
        output_dir = os.path.join(output_base_dir, f'Subject {self.subject.currentText()}',
                                  f'track{self.current_track_index}')
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f'{os.path.basename(file_path).replace(".csv", ".png")}')
        plt.savefig(output_file_path)
        plt.close()
        print(f'Completed')

    def update_break_time(self):

        self.break_timer.stop()
        self.stop_music()
        self.break_label.setText(f"Taking a break")
        self.track_timer.start(self.break_duration * 1000)

        if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
            file_path = self.data_acquisition_thread.latest_file_path
            if file_path:  # Check if file path exists
                self.generate_and_save_plot(file_path)  # Emit signal to generate and save plot

        # Stop EEG data acquisition during the break
        if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
            self.data_acquisition_thread.stop()

    def pause_music(self):
        pygame.mixer.music.pause()
        self.status_label.setText("Paused")

    def stop_music(self):
        pygame.mixer.music.stop()
        self.status_label.setText("Stopped")
        self.break_timer.stop()
        self.break_label.clear()
        self.image_label.setPixmap(QtGui.QPixmap())
        # Stop EEG data acquisition when music is stopped
        if self.data_acquisition_thread and self.data_acquisition_thread.isRunning():
            self.data_acquisition_thread.stop()

    def set_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaled(350, 350))  # Adjust size as needed


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
