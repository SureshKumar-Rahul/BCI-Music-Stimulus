import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from keras import models, layers, regularizers, optimizers
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from scipy import signal  # Importing the signal module
from scipy.signal import windows
from scipy.stats import kurtosis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg


# Function to create bandpass filters
def create_bandpass_filters(fs=250):
    # Define the frequency bands for EEG (Delta, Theta, Alpha, Beta, Gamma)
    bands = [
        (0.5, 4),  # Delta
        (4, 8),  # Theta
        (8, 13),  # Alpha
        (13, 30),  # Beta
        (30, 50)  # Gamma
    ]

    filters = []

    # Create a Butterworth bandpass filter for each frequency band
    for lowcut, highcut in bands:
        b, a = signal.butter(4, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
        filters.append((b, a))  # Store filter coefficients (b, a) for each band

    return filters



def drawsubplots(data0, data_bands):
    titlesize = 12
    num_channels = len(data_bands)
    f, axarr = plt.subplots(num_channels + 1, sharex=True, figsize=(10, 2 * (num_channels + 1)))

    axarr[0].plot(range(len(data0)), data0, 'k', linewidth=1.0)
    axarr[0].set(ylabel='Amplitude')
    axarr[0].set_title('Original signal', fontsize=titlesize)
    axarr[0].grid(True)

    band_names = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']

    for i in range(num_channels):
        axarr[i + 1].plot(range(len(data_bands[i])), data_bands[i], 'b', linewidth=1.0)
        axarr[i + 1].set(ylabel='Amplitude')
        axarr[i + 1].set_title(f'{band_names[i]} band signal', fontsize=titlesize)
        axarr[i + 1].grid(True)

    plt.subplots_adjust(hspace=0.5)
    plt.xlabel('Samples', fontsize=13)
    plt.show()

# New Bandpass Filter
def bandpass(start, stop, data, fs=250, order=4):
    bp_Hz = np.array([start, stop])
    b, a = signal.butter(order, bp_Hz / (fs / 2.0), btype='bandpass')
    return signal.filtfilt(b, a, data, axis=0)
def highpass(cutoff, data, fs=250, order=4):
    b, a = signal.butter(order, cutoff / (fs / 2.0), btype='highpass')
    return signal.filtfilt(b, a, data, axis=0)

# Define Notch Filter for power line noise (50Hz)
def notch_filter(val, data, fs=250):
    notch_freq_Hz = np.array([float(val)])
    for freq_Hz in np.nditer(notch_freq_Hz):
        bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
        b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
        data = signal.filtfilt(b, a, data)  # Using filtfilt for zero-phase filtering
    return data

# Signal Preprocessing
def preprocess_signal(data, fs):
    # Apply notch filter to remove powerline noise
    data = notch_filter(50.0, data, fs)

    # Apply baseline correction (mean subtraction)
    data = data - np.mean(data)

    # Apply windowing to smooth the edges
    data = apply_windowing(data)

    return data
def apply_windowing(data):
    window = windows.hann(len(data))  # Using windows.hann to avoid phase issues
    return data * window
# Feature extraction function
def extract_features(window):
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

    fft_freqs = np.fft.fftfreq(len(window), d=1 / 250)
    fft_vals = np.abs(np.fft.fft(window))
    sum_fft_vals = np.sum(fft_vals)
    features.append(np.sum(np.abs(fft_freqs * fft_vals)) / sum_fft_vals if sum_fft_vals != 0 else 0)
    prob_fft = np.abs(fft_vals) ** 2
    prob_fft /= np.sum(prob_fft)
    spectral_entropy = -np.sum(prob_fft * np.log(prob_fft + 1e-10))
    features.append(spectral_entropy)
    peak_freq = fft_freqs[np.argmax(fft_vals)]
    features.append(peak_freq)
    try:
        model = AutoReg(window, lags=2, old_names=False)
        model_fit = model.fit()
        features.extend(model_fit.params[1:3])
    except Exception:
        features.extend([0, 0])
    return features
# Data Augmentation Function
def augment_signal(data):
    augmented_data = []

    # Additive Gaussian Noise
    noise = np.random.normal(0, 0.01, len(data))
    augmented_data.append(data + noise)

    # Time Shifting
    shift = np.random.randint(1, len(data) // 10)
    augmented_data.append(np.roll(data, shift))

    # Scaling
    scale = np.random.uniform(0.9, 1.1)
    augmented_data.append(data * scale)

    # Flipping (inversion)
    augmented_data.append(-data)

    # Time Stretching (using resample, not directly in time)
    stretched_data = signal.resample(data, int(len(data) * 1.1))
    augmented_data.append(stretched_data[:len(data)])  # Ensure same length

    return augmented_data
# Parameters
window_size = 125 * 10  # for 2 seconds (125 Hz * 2 sec = 250 samples per window)
fs = 125

# Load the EEG data
subjects = ['Subject 0']
#subjects = ['Subject 1']

tracks = ['track0.mp3', 'track1.mp3', 'track2.mp3', 'track3.mp3', 'track4.mp3', 'track5.mp3',
          'track6.mp3', 'track7.mp3', 'track8.mp3', 'track9.mp3']



for subject in subjects:
    # Preprocessing Loop
    all_features = []

    for track in tracks:
        file_pattern = f'Data/Music/{subject}/{track}/data_*.csv'
        files = sorted(glob.glob(file_pattern))

        for file in files:
            recording = pd.read_csv(file)
            music_track = recording['Track'].unique()[0]
            data = recording.drop(columns=['Track']).to_numpy().T

            for channel_index, channel_data in enumerate(data):
                for start in range(0, len(channel_data), window_size):
                    window = channel_data[start:start + window_size]
                    if len(window) == window_size:
                        # Process the original window first
                        window = preprocess_signal(window, fs)
                        raw_features = extract_features(window)

                        # Extract features from frequency bands
                        delta_band = bandpass(0.5, 4.0, window, fs, order=4)
                        theta_band = bandpass(4.0, 8.0, window, fs, order=4)
                        alpha_band = bandpass(8.0, 13.0, window, fs, order=4)
                        beta_band = bandpass(13.0, 30.0, window, fs, order=4)
                        gamma_band = bandpass(30.0, 40.0, window, fs, order=4)

                        # Extract features from each band
                        delta_features = extract_features(delta_band)
                        theta_features = extract_features(theta_band)
                        alpha_features = extract_features(alpha_band)
                        beta_features = extract_features(beta_band)
                        gamma_features = extract_features(gamma_band)
                        drawsubplots(window,[delta_band,theta_band,alpha_band,beta_band,gamma_band])
                        # Concatenate features
                        features = np.concatenate([
                            raw_features,
                            delta_features,
                            theta_features,
                            alpha_features,
                            beta_features,
                            gamma_features
                        ])

                        # Store the features from the original signal
                        feature_dict = {
                            'track': music_track,
                            'channel': channel_index,
                            'features': features
                        }
                        all_features.append(feature_dict)

    # Define the feature names
    bands = ['Raw', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    base_feature_names = [
        'Mean', 'Standard Deviation', 'Mean Absolute Deviation',
        'Interquartile Range', '75th Percentile', 'Kurtosis',
        'Range', 'Sum of Squares of FFT', 'Spectral Centroid',
        'Spectral Entropy', 'Peak Frequency',
        'AR Coefficient 1', 'AR Coefficient 2'
    ]
    feature_names = [f'{band}_{feat}' for band in bands for feat in base_feature_names]

    # Convert features into DataFrame
    all_features_df = pd.DataFrame(all_features)
    all_features_df['label'] = all_features_df['track'].astype(int)

    # Shuffle the dataset and reset the index
    all_features_df = all_features_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the data into features (X) and labels (y)
    X = pd.DataFrame(all_features_df['features'].tolist(), columns=feature_names)
    y = all_features_df['label'].values

    # First, split the data into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.1,  # Reserve 10% for the test set
        random_state=42,
        stratify=y  # Ensures the class distribution remains the same across splits
    )

    # Now, split the training+validation set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15,  # 15% of the remaining data for validation
        random_state=42,
        stratify=y_train_val  # Keeps class distribution in training and validation sets balanced
    )

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler for future use
    dump(scaler, f'scaler_{subjects[0]}.joblib')

    # Determine the number of classes
    num_classes = len(np.unique(y_train))

    # Print dataset distribution for sanity check
    print("Training set distribution:", np.bincount(y_train))
    print("Validation set distribution:", np.bincount(y_val))
    print("Test set distribution:", np.bincount(y_test))


    # Build and train the ANN model
    def build_ann_model(input_shape, num_classes):
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


    # Build and train the ANN model
    model = build_ann_model((X_train.shape[1],), num_classes)

    es = EarlyStopping(monitor='val_loss', mode='min', patience=80, verbose=1)
    mc = ModelCheckpoint(f'best_ann_model_{subjects[0]}.keras', monitor='val_accuracy', mode='max', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, min_lr=1e-13)

    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    batch_size = 64
    history = model.fit(X_train, y_train,
                        epochs=1000,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        class_weight=class_weights_dict,  # Add class weights
                        callbacks=[es, mc, reduce_lr],
                        verbose=1)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Print classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Generate classification report as a dictionary
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    print("Classification Report:\n", classification_report(y_test, y_pred_classes))
    file_path = f'Performance Plots/{subject}/'
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    # Convert the report to a DataFrame
    report_df = pd.DataFrame(report).transpose()

    # Round the values to two decimal places
    report_df = report_df.round(2)

    # List of metrics to plot
    metrics = ['precision', 'recall', 'f1-score']

    # Set up the figure for bar plots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define the width of each bar and x-axis positions
    bar_width = 0.2
    index = np.arange(len(report_df))

    # Loop over the metrics to plot each one
    for i, metric in enumerate(metrics):
        plt.bar(index + i * bar_width, report_df[metric], bar_width, label=metric)

    # Add labels, title, and grid
    plt.xlabel('Songs', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Classification Report for Each Song Window Size {window_size / 250} sec Batch Size{batch_size}', fontsize=14)
    plt.xticks(index + bar_width, report_df.index, rotation=45)
    plt.ylim([0, 1])
    plt.legend()

    # Add gridlines for better readability
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Show the plot
    plt.tight_layout()

    # Save the plot
    plt.savefig(file_path + f'classification_report Window Size {window_size / 250} sec Batch Size{batch_size}.svg', bbox_inches='tight')

    # Show the plot
    plt.show()

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Model Accuracy Window Size: {window_size / 250} second Batch Size: {batch_size} {subject}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss Window Size: {window_size / 250} second Batch Size: {batch_size} {subject}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()

    # Save the plot in the specified directory
    plt.savefig(file_path + f'Performance Plot Window Size {window_size / 250} sec Batch Size{batch_size}.svg')
    plt.show()

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # If using one-hot encoding, otherwise use directly

    # Get the confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt=".2f",
                xticklabels=["song 0", "song 1", "song 2", "song 3", "song 4", "song 5", "song 6", "song 7", "song 8",
                             "song 9"],
                yticklabels=["song 0", "song 1", "song 2", "song 3", "song 4", "song 5", "song 6", "song 7", "song 8",
                             "song 9"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Normalized Confusion Matrix Window Size: {window_size / 250} sec Batch Size: {batch_size} {subject}')
    plt.savefig(file_path + f'Normalized Confusion Matrix Window Size {window_size / 250} sec Batch Size{batch_size}.svg')
    plt.show()

    # Optionally print the classification report
    print(classification_report(y_test, y_pred_classes))