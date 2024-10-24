# EEG-Based Music Stimuli Classification

This repository contains the code and resources for the thesis project titled **EEG-Based Brain-Computer Interface Implementation**, which explores the classification of music stimuli using EEG signals collected from an OpenBCI CytonDaisy 16-channel headset.

## Project Overview

In this project, EEG signals were recorded from multiple subjects and processed to classify different music stimuli. We used machine learning techniques, including a shallow artificial neural network (ANN), to analyze the EEG data. The project evaluates the impact of different time window sizes on classification accuracy and compares subject-specific models with an all-subject combined model.

### Key Features:
- **EEG Signal Acquisition**: Data collection using OpenBCI CytonDaisy 16-channel EEG device.
- **Signal Preprocessing**: Filters (bandpass, notch filter) and windowing applied to clean EEG signals.
- **Feature Extraction**: Extracted both time-domain and frequency-domain features.
- **ANN Model**: Implemented a shallow neural network to classify music stimuli.
- **Performance Evaluation**: Metrics like accuracy, precision, recall, F1-score, and confusion matrices were used to evaluate the models.

## Repository Structure

```plaintext
.
├── data_acquisition_thread.py      # Code for EEG data acquisition
├── data_acquisition_thread_music.py# Music data acquisition logic
├── main_window.py                  # GUI for controlling music playback and EEG recording
├── shallow_neural_network_cleaned.py# Implementation of the ANN model
├── song_window.py                  # Interface for controlling music and EEG data collection
├── README.md                       # Project documentation
