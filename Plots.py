import glob
import os

import matplotlib.pyplot as plt
import mne
import pandas as pd

# Define subjects and tracks
subjects = ['Subject 0'] #, 'Subject 1', 'Subject 2','Subject 3']
tracks = ['track0.mp3', 'track1.mp3', 'track2.mp3', 'track3.mp3', 'track4.mp3', 'track5.mp3',
          'track6.mp3', 'track7.mp3', 'track8.mp3', 'track9.mp3']

# Base directory for saving plots
output_base_dir = 'EEG_Plots_RAW'
os.makedirs(output_base_dir, exist_ok=True)

# Total number of files to process
total_files = 0
for subject in subjects:
    for track in tracks:
        file_pattern = f'Data/Music/{subject}/{track}/data_*.csv'
        total_files += len(glob.glob(file_pattern))

# Progress tracking
completed_files = 0


# Function to clean data using MNE
def clean_data(df):
    # Convert DataFrame to MNE RawArray
    ch_names = [str(i) for i in range(1, 17)]
    ch_types = ['eeg'] * 16
    sfreq = 255  # Sampling frequency (Hz)
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


# Iterate over subjects and tracks
for subject in subjects:
    track_index =0
    for track in tracks:
        # Define the path pattern to match CSV files
        file_pattern = f'Data/Music/{subject}/{track}/data_*.csv'
        track_index +=1

        # Get the list of files matching the pattern
        files = glob.glob(file_pattern)

        # Iterate over each file
        for file in files:
            # Load the CSV file
            df = pd.read_csv(file)

            if output_base_dir != 'EEG_Plots_RAW':
                # Clean the data using MNE
                df_cleaned = clean_data(df)

            else:
                df_cleaned = df

            # Set up the plot with 16 subplots, one for each electrode
            fig, axes = plt.subplots(nrows=16, ncols=1, figsize=(12, 24), sharex=True)
            fig.suptitle(f'EEG Data Visualization for {subject} - {track} - {os.path.basename(file)}')

            # Plot each electrode's data
            for i in range(1, 17):
                ax = axes[i - 1]
                ax.plot(df_cleaned.index, df_cleaned[str(i)], label=f'Electrode {i}')
                ax.set_ylabel(f'Electrode {i}')
                ax.legend(loc='upper right')

            # Add common labels and set the x-axis to represent sample indices
            axes[-1].set_xlabel('Sample Index')

            # Adjust layout to prevent overlap
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Create the output directory for the subject and track
            output_dir = os.path.join(output_base_dir, f'{subject}',
                                      f'track{track_index-1}')
            os.makedirs(output_dir, exist_ok=True)

            # Define the output file path
            output_file_path = os.path.join(output_dir, f'{os.path.basename(file).replace(".csv", ".png")}')

            # Save the plot
            plt.savefig(output_file_path)
            plt.close()

            # Update and print progress
            completed_files += 1
            print(f'Completed {completed_files} out of {total_files} plots')

print("All plots have been generated and saved.")
