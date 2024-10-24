import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from keras import models, layers, regularizers, optimizers
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from scipy import signal
from scipy.signal import windows
from scipy.stats import kurtosis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from statsmodels.tsa.ar_model import AutoReg


# =======================
# Helper Functions
# =======================

def create_bandpass_filters(fs=125):
    """
    Create bandpass filters for different EEG frequency bands.
    """
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50)]  # EEG Bands
    return [signal.butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band') for lowcut, highcut in bands]


def drawsubplots(data0, data_bands, subject, window, processed_data=None):
    """
    Draw subplots for the original signal, the processed window, and frequency band signals.
    """
    titlesize = 12
    num_channels = len(data_bands)

    # Determine the number of rows based on whether processed_data is provided
    if processed_data is not None:
        f, axarr = plt.subplots(num_channels + 2, sharex=True, figsize=(10, 2 * (num_channels + 2)))
    else:
        f, axarr = plt.subplots(num_channels + 1, sharex=True, figsize=(10, 2 * (num_channels + 1)))

    # Plot original signal
    axarr[0].plot(range(len(data0)), data0, 'k', linewidth=1.0)
    axarr[0].set(ylabel='Amplitude')
    axarr[0].set_title('Original signal', fontsize=titlesize)
    axarr[0].grid(True)

    # If processed_data is provided, plot it as the second row
    if processed_data is not None:
        axarr[1].plot(range(len(processed_data)), processed_data, 'b', linewidth=1.0)
        axarr[1].set(ylabel='Amplitude')
        axarr[1].set_title('Processed signal', fontsize=titlesize)
        axarr[1].grid(True)

    # Plot frequency band signals
    band_names = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']

    for i in range(num_channels):
        axarr[i + (2 if processed_data is not None else 1)].plot(range(len(data_bands[i])), data_bands[i], 'b',
                                                                 linewidth=1.0)
        axarr[i + (2 if processed_data is not None else 1)].set(ylabel='Amplitude')
        axarr[i + (2 if processed_data is not None else 1)].set_title(f'{band_names[i]} band signal',
                                                                      fontsize=titlesize)
        axarr[i + (2 if processed_data is not None else 1)].grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.xlabel('Samples', fontsize=13)

    # Save the plot
    file_path = f'Performance Plots/{subject}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'Band plot {window / 125} sec.png', bbox_inches='tight')

    # Show the plot
    plt.show()


def apply_windowing(data):
    """
    Apply Hann windowing to smooth the signal.
    """
    window = windows.hann(len(data))
    return data * window


def preprocess_signal(data, fs=125, notch_val=50.0):
    """
    Preprocess the signal by applying notch filter, mean subtraction, and windowing.
    """
    data = notch_filter(notch_val, data, fs)
    data = data - np.mean(data)
    return apply_windowing(data)


def notch_filter(val, data, fs=125):
    """
    Apply a notch filter to remove powerline noise at a specified frequency (default: 50Hz).
    """
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        data = signal.filtfilt(b, a, data)
    return data


def bandpass(start, stop, data, fs=125, order=4):
    """
    Apply a bandpass filter between the specified frequency range (start, stop).
    """
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(order, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.filtfilt(b, a, data, axis=0)


def extract_features(window, fs=125):
    """
    Extract features from the signal window including time-domain and frequency-domain features.
    """
    features = [
        np.mean(window),
        np.std(window),
        np.mean(np.abs(window - np.mean(window))),
        np.percentile(window, 75) - np.percentile(window, 25),
        np.percentile(window, 75),
        kurtosis(window) if np.var(window) > 1e-10 else 0,
        np.max(window) - np.min(window),
        np.sum(np.power(np.abs(np.fft.fft(window)), 2))
    ]

    # FFT Features
    fft_freqs = np.fft.fftfreq(len(window), d=1 / fs)
    fft_vals = np.abs(np.fft.fft(window))
    sum_fft_vals = np.sum(fft_vals)

    if sum_fft_vals != 0:
        features.append(np.sum(np.abs(fft_freqs * fft_vals)) / sum_fft_vals)
    else:
        features.append(0)

    # Spectral Entropy
    prob_fft = np.abs(fft_vals) ** 2 / np.sum(np.abs(fft_vals) ** 2)
    spectral_entropy = -np.sum(prob_fft * np.log(prob_fft + 1e-10))
    features.append(spectral_entropy)

    # Peak Frequency
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    features.append(peak_freq)

    # Autoregression coefficients
    try:
        model = AutoReg(window, lags=2, old_names=False)
        model_fit = model.fit()
        features.extend(model_fit.params[1:3])
    except Exception:
        features.extend([0, 0])

    return features


def load_test_data(subject, tracks):
    """
    Load test data for a given subject from specified tracks.
    """
    test_data = []
    for track in tracks:
        files = sorted(glob.glob(f'Data/Music/Test/{subject}/{track}/data_*.csv'))  # Adjust path as necessary
        for file in files:
            recording = pd.read_csv(file)
            test_data.append(recording)
    return pd.concat(test_data)


def process_test_data(subject, tracks, window_size, fs=125):
    """
    Process test data for a given subject and extract features from each track.
    """
    all_features = []

    for track in tracks:
        files = sorted(glob.glob(f'Data/Music/Test/{subject}/{track}/data_*.csv'))  # Adjust path for test data
        for file in files:
            print(file)
            recording = pd.read_csv(file)
            music_track = recording['Track'].unique()[0]
            data = recording.drop(columns=['Track']).to_numpy().T

            for channel_index, channel_data in enumerate(data):
                for start in range(0, len(channel_data), window_size):
                    window = channel_data[start:start + window_size]
                    if len(window) == window_size:
                        raw_window = window
                        window = preprocess_signal(window, fs)

                        # Extract band-specific features
                        delta_band = bandpass(0.5, 4, window, fs)
                        theta_band = bandpass(4, 8, window, fs)
                        alpha_band = bandpass(8, 13, window, fs)
                        beta_band = bandpass(13, 30, window, fs)
                        gamma_band = bandpass(30, 50, window, fs)

                        # Extract features from each band
                        delta_features = extract_features(delta_band)
                        theta_features = extract_features(theta_band)
                        alpha_features = extract_features(alpha_band)
                        beta_features = extract_features(beta_band)
                        gamma_features = extract_features(gamma_band)

                        # Concatenate features
                        features = np.concatenate(
                            [delta_features, theta_features, alpha_features, beta_features,
                             gamma_features])
                        all_features.append({
                            'track': music_track,
                            'channel': channel_index,
                            'features': features
                        })

    return pd.DataFrame(all_features)


def evaluate_model_performance(subjects, tracks, window_size):
    """
    Evaluate model performance using test data for each subject.
    """
    for subject in subjects:

        # Process test data
        all_features_df = process_test_data(subject, tracks, window_size=window_size,
                                            fs=125)  # Adjust window_size as needed
        # Extract features and labels
        all_features_df['label'] = all_features_df['track'].astype(int)
        X_test = pd.DataFrame(all_features_df['features'].tolist())
        y_test = all_features_df['label'].values

        # Load the model
        model = models.load_model(f'best_ann_model_{subject}_{window_size}.keras')  # Adjust path as necessary

        # Load the corresponding scaler
        scaler = load(f'scaler_{subject}_{window_size}.joblib')  # Ensure the correct scaler is loaded
        X_test_scaled = scaler.transform(X_test)

        # Ensure the features are consistent
        if X_test_scaled.shape[1] != model.input_shape[1]:
            raise ValueError(
                f"Mismatch in number of features: expected {model.input_shape[1]}, got {X_test_scaled.shape[1]}")

        # Generate predictions
        y_pred = np.argmax(model.predict(X_test_scaled), axis=1)

        # Calculate and print the classification report
        print(f"Evaluation for {subject}:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, class_names=np.unique(y_test), subject=subject,
                              window=window_size)  # Window is not relevant for test
        plot_classification_report(y_test, y_pred, subject, window_size, type='test')


# =======================
# Main Data Processing Loop
# =======================

def process_subject_data(subject, tracks, window_size, fs=125):
    """
    Process data for a given subject and extract features from each track.
    """
    all_features = []
    first_plot = True  # Flag to track if we have already plotted the bands for this subject
    for track in tracks:
        files = sorted(glob.glob(f'Data/Music/{subject}/{track}/data_*.csv'))
        for file in files:
            print(file)
            recording = pd.read_csv(file)
            music_track = recording['Track'].unique()[0]
            data = recording.drop(columns=['Track']).to_numpy().T
            for channel_index, channel_data in enumerate(data):
                for start in range(0, len(channel_data), window_size):
                    window = channel_data[start:start + window_size]
                    if len(window) == window_size:

                        raw_window = window
                        window = preprocess_signal(window, fs)

                        # Extract raw features
                        raw_features = extract_features(window)

                        # Extract band-specific features and create bands for visualization
                        delta_band = bandpass(0.5, 4, window, fs)
                        theta_band = bandpass(4, 8, window, fs)
                        alpha_band = bandpass(8, 13, window, fs)
                        beta_band = bandpass(13, 30, window, fs)
                        gamma_band = bandpass(30, 50, window, fs)

                        # Plot the bands only once for the subject
                        if first_plot:
                            drawsubplots(raw_window, [delta_band, theta_band, alpha_band, beta_band, gamma_band],
                                         subject=subject, window=window_size, processed_data=window)
                            first_plot = False  # Set the flag to False after plotting the first time

                        # Extract features from each band
                        delta_features = extract_features(delta_band)
                        theta_features = extract_features(theta_band)
                        alpha_features = extract_features(alpha_band)
                        beta_features = extract_features(beta_band)
                        gamma_features = extract_features(gamma_band)

                        # Concatenate features
                        features = np.concatenate(
                            [delta_features, theta_features, alpha_features, beta_features,
                             gamma_features])
                        all_features.append({
                            'track': music_track,
                            'channel': channel_index,
                            'features': features
                        })

    return pd.DataFrame(all_features)


# =======================
# Model Building Functions
# =======================

def build_ann_model(input_shape, num_classes):
    """
    Build an Artificial Neural Network (ANN) model for classification.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.002)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# =======================
# Plotting and Visualization
# =======================

def plot_classification_report(y_test, y_pred, subject, window, type='training'):
    """
    Plot the classification report as a bar chart in percentage with a summary table below.
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2) * 100  # Convert to percentage
    ax = report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(12, 6), grid=True)
    if type == 'training':
        plt.title(f'Classification Report - {subject} Window Size:{window / 125} sec')
    else:
        plt.title(f'Classification Report of New data- {subject} Window Size:{window / 125} sec')

    plt.xticks(rotation=45)
    plt.ylabel('Percentage (%)')
    plt.tight_layout()

    file_path = f'Performance Plots/{subject}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if type == 'training':
        plt.savefig(file_path + f'classification_report Window Size {window / 125} sec.svg', bbox_inches='tight')
    else:
        plt.savefig(file_path + f'cClassification Report of New data Window Size {window / 125} sec.svg',
                    bbox_inches='tight')

    plt.show()


def plot_subject_accuracies(subjects, y_tests, y_preds, window_size):
    """
    Plot accuracies for multiple subjects for a given window size.

    Parameters:
    - subjects: List of subjects
    - y_tests: List of true label arrays for the subjects
    - y_preds: List of predicted label arrays for the subjects
    - window_size: The window size used for the predictions
    """
    accuracies = []

    # Calculate accuracy for each subject
    for i in range(len(subjects)):
        accuracy = classification_report(y_tests[i], y_preds[i], output_dict=True)['accuracy'] * 100
        accuracies.append(accuracy)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(subjects, accuracies, color='blue')
    plt.ylim(0, 100)
    plt.title(f'Accuracies for Each Subject - Window Size: {window_size / 125:.2f} sec')
    plt.ylabel('Accuracy (%)')

    # Add accuracy values on top of the bars
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    file_path = f'Performance Plots/All_in_one/{window_size}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'accuracy_plot_{window_size}.svg')

    plt.show()


def plot_multiple_confusion_matrices(subjects, y_tests, y_preds, class_names, window_size):
    """
    Plot confusion matrices for multiple subjects in one figure.

    Parameters:
    - subjects: List of subjects
    - y_tests: List of true label arrays for each subject
    - y_preds: List of predicted label arrays for each subject
    - class_names: List of class names to display
    - window_size: Size of the data window (for plot title)
    """
    num_subjects = len(subjects)
    num_columns = 2  # Number of columns in the grid
    num_rows = (num_subjects + num_columns - 1) // num_columns  # Calculate number of rows

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 6))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, subject in enumerate(subjects):
        cm = confusion_matrix(y_tests[i], y_preds[i])
        sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, cmap='Blues', fmt='.2f',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i])

        axes[i].set_title(f'Confusion Matrix - {subject} Window Size: {window_size / 125}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')

    # Hide any empty subplots if the number of subjects is not a multiple of num_columns
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    file_path = f'Performance Plots/All_in_one/{window_size}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'confusion_plot_{window_size}.svg')

    plt.show()


def plot_confusion_matrix(y_test, y_pred, class_names, subject, window):
    """
    Plot the confusion matrix as a heatmap.
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm / cm.sum(axis=1)[:, np.newaxis], annot=True, cmap='Blues', fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {subject} Window Size:{window / 125}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    file_path = f'Performance Plots/{subject}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'Normalized Confusion Matrix Window Size {window / 125} sec.svg')

    plt.show()


def plot_training_curves_multiple_subjects(subjects, histories, window_size):
    """
    Plot training and validation curves for multiple subjects in one figure.

    Parameters:
    - subjects: List of subjects
    - histories: List of training histories for each subject
    - window_size: Size of the data window (for plot title)
    """
    plt.figure(figsize=(12, 6))

    for i, subject in enumerate(subjects):
        history = histories[i]
        plt.plot(history.history['accuracy'], label=f'Training Accuracy - {subject}', linestyle='-')
        plt.plot(history.history['val_accuracy'], label=f'Validation Accuracy - {subject}', linestyle='--')

    plt.title(f'Training and Validation Curves - Window Size: {window_size / 125:.2f} sec')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    file_path = f'Performance Plots/All_in_one/{window_size}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'training_curve{window_size}.svg')

    plt.show()


def plot_training_history(history, subject, window):
    """
    Plot the training and validation accuracy and loss curves.
    """
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy - {subject}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss - {subject} Window Size:{window / 125}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    file_path = f'Performance Plots/{subject}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    plt.savefig(file_path + f'Training Plot Window Size {window / 125} sec.svg')
    plt.show()


# =======================
# Full Workflow
# =======================

def run_workflow(subjects, tracks, window_sizes=[1250], fs=125):
    """
    Main function to process subjects, train and evaluate the model.

    Parameters:
    - subjects: List of subjects
    - tracks: List of tracks to process for each subject
    - window_sizes: List of window sizes to use for training and evaluation
    - fs: Sampling frequency
    """
    all_subject_y_tests = {window: [] for window in window_sizes}
    all_subject_y_preds = {window: [] for window in window_sizes}
    all_subject_histories = {window: [] for window in window_sizes}  # Store histories

    for subject in subjects:
        for window_size in window_sizes:
            # Process data for each subject
            all_features_df = process_subject_data(subject, tracks, window_size, fs)

            # Extract feature matrix and labels
            all_features_df['label'] = all_features_df['track'].astype(int)
            X = pd.DataFrame(all_features_df['features'].tolist())
            y = all_features_df['label'].values

            # Split data
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.1, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val
            )

            # Scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
            dump(scaler, f'scaler_{subject}_{window_size}.joblib')

            # Build and train the model
            num_classes = len(np.unique(y_train))
            model = build_ann_model((X_train.shape[1],), num_classes)

            es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=1)
            mc = ModelCheckpoint(f'best_ann_model_{subject}_{window_size}.keras', monitor='val_accuracy', mode='max',
                                 save_best_only=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_lr=1e-13)

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = dict(enumerate(class_weights))

            history = model.fit(X_train, y_train, epochs=1500, batch_size=64, validation_data=(X_val, y_val),
                                class_weight=class_weights_dict, callbacks=[es, mc, reduce_lr], verbose=1)
            all_subject_histories[window_size].append(history)

            # Evaluate the model
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"Test Accuracy for {subject} with window size {window_size}: {test_acc:.4f}")

            # Generate predictions
            y_pred = np.argmax(model.predict(X_test), axis=1)

            # Store the test and predicted labels for each subject and window size
            all_subject_y_tests[window_size].append(y_test)
            all_subject_y_preds[window_size].append(y_pred)
            plot_classification_report(y_test, y_pred, subject=subject, window=window_size)
            plot_confusion_matrix(y_test, y_pred, class_names=np.unique(y_test), subject=subject, window=window_size)
            plot_training_history(history, subject, window=window_size)
    # Plot accuracies after processing all subjects
    for window_size in window_sizes:
        plot_multiple_confusion_matrices(subjects, all_subject_y_tests[window_size], all_subject_y_preds[window_size],
                                         class_names=np.unique(y), window_size=window_size)

        plot_subject_accuracies(subjects, all_subject_y_tests[window_size], all_subject_y_preds[window_size],
                                window_size)
        plot_training_curves_multiple_subjects(subjects, all_subject_histories[window_size], window_size)


# =======================
# Run the training workflow
# =======================
subjects = ['Subject 0', 'Subject 1', 'Subject 2', 'Subject 3', 'Subject 4','All Subject']
tracks = ['track0.mp3', 'track1.mp3', 'track2.mp3', 'track3.mp3', 'track4.mp3', 'track5.mp3', 'track6.mp3',
          'track7.mp3', 'track8.mp3', 'track9.mp3']
window_sizes = [125 * 2, 125 * 10, 125 * 20, 125 * 30]
# Run the full workflow
#run_workflow(subjects, tracks, window_sizes=window_sizes, fs=125)


# =======================
# Run Evaluation Workflow
# =======================
def run_evaluation_workflow(subjects, window_sizes, tracks):
    """
    Run evaluation workflow for testing the model on new data.
    """
    for window_size in window_sizes:
        # Evaluate performance using test data for subjects 0 and 1
        evaluate_model_performance(subjects, tracks, window_size)


# =======================
# Execute Workflows
# =======================

# Run the evaluation workflow
subjects = ['Subject 0', 'Subject 1']
window_sizes = [125 * 30]

run_evaluation_workflow(subjects, window_sizes, tracks)
