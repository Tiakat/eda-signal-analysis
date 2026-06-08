import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter_labeled"
OUTPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_windows"

# Window parameters (based on your methods)
PRE_STIM_DURATION = 5    # seconds before stimulus onset (for baseline)
STIM_DURATION = 40       # seconds of stimulation (from your protocol)
POST_STIM_DURATION = 5   # seconds after stimulus ends (for recovery)

# Calculate total window duration
TOTAL_DURATION = PRE_STIM_DURATION + STIM_DURATION + POST_STIM_DURATION  # 50 seconds

SAMPLING_RATE = 128  # Hz

# Sample counts
PRE_SAMPLES = PRE_STIM_DURATION * SAMPLING_RATE       # 640
STIM_SAMPLES = STIM_DURATION * SAMPLING_RATE          # 5120
POST_SAMPLES = POST_STIM_DURATION * SAMPLING_RATE     # 640
TOTAL_SAMPLES = PRE_SAMPLES + STIM_SAMPLES + POST_SAMPLES  # 6400

# ============================================================================
# CLASSIFICATION OF STIMULI (based on your Methods)
# ============================================================================

# Mapping from sheet names to (Condition, Modality)
# You will need to adjust these based on your actual sheet names

STIMULUS_MAPPING = {
    # Experimental condition (McNicoll)
    'Mc_nicoll': ('experimental', 'visual'),
    'Audio_mc_nicoll': ('experimental', 'auditory'),
    'Audio___tableau_Mc_nicoll': ('experimental', 'auditory'),  # variant
    'Audio___tableau_Mc_nicoll_2': ('experimental', 'auditory'),
    
    # Positive reference (Sisley - Regattas)
    'regata': ('positive_ref', 'visual'),
    'Monet': ('positive_ref', 'visual'),  # Regattas is by Sisley, but sheet named Monet
    'Audio_monet': ('positive_ref', 'auditory'),
    'Audio___tableau_monet': ('positive_ref', 'auditory'),
    'Audio___tableau_monet_2': ('positive_ref', 'auditory'),
    
    # Negative reference (Grigorescu - Smârdan Attack)
    'Bataille': ('negative_ref', 'visual'),
    'Musique__test_EN': ('negative_ref', 'auditory'),
    'Musique__test_EN_2': ('negative_ref', 'auditory'),
    
    # Other stimuli (may not be used in main analysis)
    'Fleurs': ('other', 'visual'),
    'routine_127_EP': ('other', 'auditory'),
    'routine_127_EP___tableau_ragata': ('other', 'combined'),
    'routine_128_EN': ('other', 'auditory'),
    'routine_128_EN__tableau_bataill': ('other', 'combined'),
}

def classify_stimulus(sheet_name):
    """Classify sheet name into (condition, modality)"""
    for key, value in STIMULUS_MAPPING.items():
        if key in sheet_name:
            return value
    return ('unknown', 'unknown')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def time_to_seconds(time_str):
    """Convert time string 'HH:MM:SS.fff' to total seconds (float)."""
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        sec_parts = parts[2].split('.')
        seconds = int(sec_parts[0])
        milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
        return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
    except:
        return None

def extract_full_window(eda_df, start_time_str, end_time_str, stimulus_name, participant_id):
    """
    Extract window: 5s pre-stimulus + full stimulus duration + 5s post-stimulus.
    Uses the ACTUAL stimulus duration from start to end markers.
    """
    # Convert times to seconds
    start_sec = time_to_seconds(start_time_str)
    end_sec = time_to_seconds(end_time_str)
    
    if start_sec is None or end_sec is None:
        return None
    
    # Calculate actual stimulus duration
    actual_stim_duration = end_sec - start_sec
    
    # Define window boundaries
    window_start_sec = start_sec - PRE_STIM_DURATION
    window_end_sec = end_sec + POST_STIM_DURATION
    
    # Convert EDA times to seconds
    eda_df['Time_sec'] = eda_df['Time'].apply(time_to_seconds)
    
    # Extract window
    window_mask = (eda_df['Time_sec'] >= window_start_sec) & (eda_df['Time_sec'] <= window_end_sec)
    window_df = eda_df[window_mask].copy()
    
    if len(window_df) == 0:
        return None
    
    # Add metadata
    window_df['participant'] = participant_id
    window_df['stimulus'] = stimulus_name
    
    # Classify condition and modality
    condition, modality = classify_stimulus(stimulus_name)
    window_df['condition'] = condition
    window_df['modality'] = modality
    
    # Timing variables
    window_df['stimulus_start_sec'] = start_sec
    window_df['stimulus_end_sec'] = end_sec
    window_df['time_in_window'] = window_df['Time_sec'] - window_start_sec
    window_df['time_from_stimulus_onset'] = window_df['Time_sec'] - start_sec
    
    # Phase classification
    window_df['phase'] = 'pre'
    window_df.loc[window_df['time_from_stimulus_onset'] >= 0, 'phase'] = 'stimulus'
    window_df.loc[window_df['time_from_stimulus_onset'] >= actual_stim_duration, 'phase'] = 'post'
    
    # Time since phase start
    window_df['time_in_phase'] = window_df['time_from_stimulus_onset']
    window_df.loc[window_df['phase'] == 'pre', 'time_in_phase'] = window_df['time_from_stimulus_onset'] + PRE_STIM_DURATION
    window_df.loc[window_df['phase'] == 'post', 'time_in_phase'] = window_df['time_from_stimulus_onset'] - actual_stim_duration
    
    return window_df

def process_participant(labeled_file_path, participant_id, output_folder):
    """Process one labeled EDA file and extract windows for each stimulus."""
    
    df = pd.read_csv(labeled_file_path)
    
    # Find all unique stimuli (excluding baseline)
    stimuli = df[df['stimulus_label'] != 'baseline']['stimulus_label'].unique()
    
    print(f"  Found {len(stimuli)} unique stimuli")
    
    all_windows = []
    
    for stimulus in stimuli:
        # Get rows for this stimulus
        stim_df = df[df['stimulus_label'] == stimulus].copy()
        
        # Find start and end markers
        start_rows = stim_df[stim_df['stimulus_start'] == True]
        end_rows = stim_df[stim_df['stimulus_end'] == True]
        
        if len(start_rows) == 0 or len(end_rows) == 0:
            print(f"    WARNING: No start/end markers for '{stimulus}'")
            continue
        
        start_time = start_rows.iloc[0]['Time']
        end_time = end_rows.iloc[0]['Time']
        
        # Extract full window with 5s pre and 5s post
        window_df = extract_full_window(df, start_time, end_time, stimulus, participant_id)
        
        if window_df is not None and len(window_df) > 0:
            all_windows.append(window_df)
            condition, modality = classify_stimulus(stimulus)
            print(f"    Extracted: {stimulus} [{condition}, {modality}] - {len(window_df)} samples")
    
    if len(all_windows) > 0:
        participant_windows = pd.concat(all_windows, ignore_index=True)
        output_file = os.path.join(output_folder, f"{participant_id}_windows.csv")
        participant_windows.to_csv(output_file, index=False)
        print(f"  SAVED: {participant_id}_windows.csv ({len(participant_windows)} total samples)")
        return True
    
    return False

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("STEP 3: Extracting Windows (5s pre + stimulus + 5s post)")
    print("=" * 80)
    print(f"Input: {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Window: -{PRE_STIM_DURATION}s to +{POST_STIM_DURATION}s after stimulus end")
    print(f"Total duration: {TOTAL_DURATION}s ({TOTAL_SAMPLES} samples at {SAMPLING_RATE}Hz)")
    print("-" * 80)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    labeled_files = [f for f in os.listdir(INPUT_PATH) if f.endswith('_labeled.csv')]
    print(f"Found {len(labeled_files)} labeled files")
    print("-" * 80)
    
    successful = 0
    failed = 0
    
    for labeled_file in labeled_files:
        participant_id = labeled_file.replace('_labeled.csv', '')
        print(f"\nProcessing: {participant_id}")
        
        file_path = os.path.join(INPUT_PATH, labeled_file)
        
        try:
            if process_participant(file_path, participant_id, OUTPUT_PATH):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Successfully extracted windows: {successful}")
    print(f"Failed: {failed}")
    print(f"\nOutput: {OUTPUT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()