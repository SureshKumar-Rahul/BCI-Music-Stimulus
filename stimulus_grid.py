# stimulus_grid.py
import random
import sys
import time

from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QApplication


class StimulusGrid(QWidget):
    closing_signal = pyqtSignal()
    highlighting_started = pyqtSignal()
    highlighting_finished = pyqtSignal()

    def __init__(self, selected_letter, selected_highlight, selected_round, selected_pattern):
        super().__init__()
        self.labels = None
        self.stimuli = None
        self.grid = None
        self.selected_letter = selected_letter
        self.selected_highlight = selected_highlight
        self.selected_round = selected_round
        self.selected_pattern = selected_pattern
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
        self.setGeometry(0, 0, desktop.screenGeometry().width(),
                         desktop.screenGeometry().height() - taskbar_height)  # Set window geometry to screen size
        # minus taskbar height
        self.show()

        self.timer = QTimer(self)
        if self.selected_pattern == 'SC':
            self.timer.timeout.connect(self.highlight_random_location_sc)
        else:
            self.timer.timeout.connect(self.highlight_random_location_rc)
        self.timer.start(1)  # Update every 1000 ms (1 second)

    def closeEvent(self, event):
        self.timer.stop()  # Stop the timer when closing the window
        self.closing_signal.emit()

    def highlight_random_location_rc(self):
        # Define the label you want to highlight more often
        total_no_of_appearance = 0
        selected_label = [label for label in self.labels if label.text() == self.selected_letter][0]
        selected_index = self.labels.index(selected_label)
        times_to_highlight = self.selected_highlight

        # Calculate the row and column indices
        selected_row_index = selected_index // 6  # Integer division to get the row index
        selected_col_index = selected_index % 6  # Modulo operator to get the column index
        print(f"selected_row_index = {selected_row_index} \nselected_col_index = {selected_col_index} \n")

        # Create a list of rows and columns
        rows = list(range(6))
        cols = list(range(6))

        for round in range(self.selected_round):
            if total_no_of_appearance < (times_to_highlight * self.selected_round):
                no_of_appearance = 0
                print(f"Round {round}")
                # Shuffle the order of rows and columns separately
                random.shuffle(rows)
                random.shuffle(cols)
                i = 0
                prev_row_index = None
                prev_col_index = None
                length = len(rows) + len(cols)
                row_index = random.randint(0, 5)
                col_index = random.randint(0, 5)
                while no_of_appearance < times_to_highlight:
                    for i in range(length):
                        for label in self.labels:
                            label.setStyleSheet('color: white')

                        # Highlight the row or column
                        if i % 2 == 0:
                            # Ensure that the row index is different from the previous one
                            while (row_index == selected_row_index and prev_col_index == selected_col_index) or (
                                    row_index == prev_row_index and col_index == prev_col_index):
                                row_index = random.randint(0, 5)
                            prev_row_index = row_index
                            if row_index == selected_row_index:
                                if no_of_appearance >= times_to_highlight:
                                    break
                                else:
                                    no_of_appearance += 1
                                    total_no_of_appearance += 1

                            for k in range(6):
                                if self.labels[row_index * 6 + k] == selected_label:
                                    self.labels[row_index * 6 + k].setStyleSheet('color: yellow')
                                else:
                                    self.labels[row_index * 6 + k].setStyleSheet('color: red')

                        else:
                            # Ensure that the column index is different from the previous one
                            while (col_index == selected_col_index and prev_row_index == selected_row_index) or (
                                    row_index == prev_row_index and col_index == prev_col_index):
                                col_index = random.randint(0, 5)
                            prev_col_index = col_index
                            if col_index == selected_col_index:
                                if no_of_appearance >= times_to_highlight:
                                    break
                                else:
                                    no_of_appearance += 1
                                    total_no_of_appearance += 1

                            for j in range(6):
                                if self.labels[j * 6 + col_index] == selected_label:
                                    self.labels[j * 6 + col_index].setStyleSheet('color: yellow')
                                else:
                                    self.labels[j * 6 + col_index].setStyleSheet('color: red')
                        time.sleep(1)
                        QApplication.processEvents()  # Process pending events to update UI
            else:
                break
        time.sleep(2)  # Wait for 1 second before final highlighting
        self.highlighting_finished.emit()  # Emit signal when highlighting finishes

    def highlight_random_location_sc(self):
        self.highlighting_started.emit()  # Emit signal when highlighting starts
        # Define how many times the target label should be highlighted more than the others
        times_to_highlight = self.selected_highlight  # Adjust this number as needed

        # Define the label you want to highlight more often
        selected_label = [label for label in self.labels if label.text() == self.selected_letter][0]
        print(selected_label)
        shuffled_labels = [selected_label] * times_to_highlight + [label for label in self.labels if
                                                                   label != selected_label]
        no_of_appearance = 0

        # Shuffle the list of labels
        random.shuffle(shuffled_labels)
        total_round = self.selected_round
        # Repeat the highlighting process for three rounds
        for round_num in range(total_round):
            # Shuffle the list of labels
            random.shuffle(shuffled_labels)
            appeared_last = False
            # Highlight all labels with a delay of 1 second between each highlight
            for label in shuffled_labels:
                if (self.selected_letter == label.text()) and (appeared_last == False):
                    label.setStyleSheet('color: yellow')
                    appeared_last = True
                    no_of_appearance += 1
                else:
                    label.setStyleSheet('color: red')
                    appeared_last = False

                time.sleep(1)  # Wait for 1 second
                QApplication.processEvents()  # Process pending events to update UI
                label.setStyleSheet('color: white')  # Reset color to original
        print(no_of_appearance)
        self.highlighting_finished.emit()  # Emit signal when highlighting finishes

    def resizeEvent(self, event):
        for label in self.labels:
            label.setFont(
                QFont("Arial", 80) if event.oldSize().height() < event.size().height() else QFont("Arial", 80))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    stimulus_grid = StimulusGrid('D', 5, 2, 'RC')
    stimulus_grid.showMinimized()
    sys.exit(app.exec_())