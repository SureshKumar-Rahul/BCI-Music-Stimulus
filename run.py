#!/usr/bin/env python3
"""Convenience launcher so you don't have to remember long commands.

Usage:
    python run.py play       # open the music player + EEG recorder GUI
    python run.py analyze    # run the ANN feature-extraction / training / evaluation script
    python run.py plots      # batch-plot the recorded EEG signals

You can run this from anywhere. It switches to the repository root first, so the
relative paths the scripts use (Data/, audio/, models/, Performance Plots/) always
resolve.
"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

COMMANDS = {
    "play": "song_window.py",
    "analyze": "shallow_neural_network_cleaned.py",
    "plots": "Plots.py",
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python run.py <command>")
        print("Commands:")
        print("  play      open the music player + EEG recorder GUI")
        print("  analyze   run the ANN feature-extraction / training / evaluation script")
        print("  plots     batch-plot the recorded EEG signals")
        sys.exit(1)

    # Run from the repo root so the scripts' relative paths resolve, and put src/
    # on the path so song_window.py can import data_acquisition_thread_music.
    os.chdir(ROOT)
    sys.path.insert(0, SRC)
    script = os.path.join(SRC, COMMANDS[sys.argv[1]])
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
