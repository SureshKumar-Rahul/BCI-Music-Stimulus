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
```

## Installation

To run the project locally:

1. Clone this repository:

    ```bash
    git clone https://github.com/yourusername/EEG-Music-Classification.git
    cd EEG-Music-Classification
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install the BrainFlow and MNE-Python libraries:

    ```bash
    pip install brainflow mne
    ```

4. Connect the OpenBCI CytonDaisy board to start EEG data acquisition.

## Usage

### Data Acquisition:

- Run the `main_window.py` to start the graphical interface for collecting EEG data while playing music stimuli.
- EEG data will be automatically saved during each session.

### Model Training:

- Use the `shallow_neural_network_cleaned.py` to build and train the ANN model on collected EEG data.
- You can experiment with different time windows (e.g., 2s, 10s, 30s) and evaluate model performance using metrics like accuracy, precision, and recall.

### Evaluation:

- Model performance will be visualized via confusion matrices and learning curves after training, allowing a detailed comparison across different models and time windows.

## Results

Our findings indicate that longer time windows, particularly the 30-second window, provided better classification accuracy under controlled conditions. However, the **All Subjects** model and individual subject models showed a significant drop in accuracy when tested on data collected from different days, revealing challenges with inter-session variability.

### Key Accuracy Highlights:
- **Subject 0 Model**: Achieved up to 80% accuracy on same-day data.
- **All Subjects Model**: Achieved 74% accuracy on same-day data.
- **Performance on different-day data**: Accuracy dropped to 12%-19%, highlighting the importance of addressing session variability.

## Future Work

Future research should focus on:

- Developing adaptive learning techniques to mitigate inter-session variability.
- Expanding the dataset to include more subjects and a broader range of stimuli.
- Investigating more complex neural networks (e.g., deeper or recurrent models) for improved classification accuracy.


## Contact

For questions, collaboration, or further information, feel free to contact:

- **Rahul Suresh Kumar**
- **Email**: rahulsureshkumar8@gmail.com

