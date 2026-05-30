# EEG-based music stimuli classification

Code and data for the paper *EEG-Based Music Stimuli Classification Using Artificial Neural Network and the OpenBCI CytonDaisy System* (Suto and Suresh Kumar, Technologies 2025, 13, 426). We recorded EEG from five people while they listened to ten songs from different genres, then trained a small neural network to tell the songs apart from the brain signals alone.

> **Citation:** Suto, J.; Kumar, R.S. EEG-Based Music Stimuli Classification Using Artificial Neural Network and the OpenBCI CytonDaisy System. *Technologies* **2025**, *13*, 426. https://doi.org/10.3390/technologies13090426

## What the project does

A PyQt5 app plays a one-minute clip of each song and, at the same time, records 16-channel EEG from an OpenBCI CytonDaisy board at 125 Hz, writing one CSV per track tagged with the song. An offline script then filters the signal, splits it into the delta, theta, alpha, beta and gamma bands, pulls time- and frequency-domain features from short windows, and trains a shallow ANN to predict which song the person was hearing. We trained one model per subject and a combined "All Subjects" model, and compared window lengths of 2, 10, 20 and 30 seconds.

## Repository structure

```plaintext
.
├── src/                              # Python source
│   ├── song_window.py                # GUI for music playback and EEG recording (entry point)
│   ├── data_acquisition_thread_music.py # BrainFlow EEG acquisition thread used by the GUI
│   ├── shallow_neural_network_cleaned.py # Feature extraction and ANN training/evaluation
│   └── Plots.py                      # Batch plotting of recorded EEG signals
├── models/                           # Trained ANN models (.keras) and feature scalers (.joblib)
├── Data/Music/                       # Recorded 16-channel EEG, one CSV per track per session
├── audio/                            # Song clips and cover images used as stimuli
├── Performance Plots/                # Figures from the paper (confusion matrices, curves)
├── docs/                             # The published paper (PDF)
├── requirements.txt                  # Pinned Python dependencies
└── README.md                         # Project documentation
```

Run the scripts from the repository root so the relative paths (`Data/`, `audio/`, `models/`, `Performance Plots/`) resolve, for example `python src/song_window.py`.

## Installation

```bash
git clone https://github.com/SureshKumar-Rahul/BCI-Music-Stimulus.git
cd BCI-Music-Stimulus
pip install -r requirements.txt
```

`requirements.txt` pins everything, including BrainFlow, MNE, TensorFlow/Keras and PyQt5.

## Usage

A small launcher (`run.py`) wraps the scripts so you don't have to type long paths. Run it from the repository root, either through `make` or directly:

| Command | What it does |
| --- | --- |
| `make play` or `python run.py play` | Open the music player and EEG recorder GUI |
| `make analyze` or `python run.py analyze` | Run feature extraction and the ANN workflow |
| `make plots` or `python run.py plots` | Batch-plot the recorded EEG signals |

To record EEG, connect the CytonDaisy board, run `make play`, pick a subject, and press Play. The app saves one CSV per track under `Data/Music/` while the music plays.

`make analyze` runs `src/shallow_neural_network_cleaned.py`, which extracts the features and saves the model, the feature scaler, and the plots (confusion matrix, training curves, per-song scores) for each window size. Which workflow it runs is chosen at the bottom of that file: by default it evaluates the saved models, with the full training workflow available just above.

## Results

Longer windows classified better. With the 30-second window, per-subject accuracy ranged from 61% to 96%, and Subject 0 reached 96%. The combined "All Subjects" model reached 53% at 30 seconds, lower than the individual models because brain responses to the same song differ from person to person.

The models do not carry over to another day. When we re-recorded the first three subjects a few days later and tested their existing models, accuracy fell to 12%, 15% and 19%. Day-to-day changes in the recording (electrode placement, skin condition, normal drift in brain activity) outweigh the differences between songs, so training and test data have to come from the same session.

## Future work

- Adaptive methods that survive across sessions.
- More subjects and a wider range of music.
- Deeper or recurrent networks instead of the shallow ANN.

## Contact

Rahul Suresh Kumar, rahulsureshkumar8@gmail.com
