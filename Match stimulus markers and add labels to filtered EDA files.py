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

# Paths
FILTERED_EDA_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter"
STIMULUS_MARKERS_PATH = r"C:\Users\katia\Desktop\min data\phd\donnees\0 Premiers traitements\0 Premiers traitements\1 Jeunes\Donnees Shimmer"
OUTPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter_labeled"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_participant_id_from_folder(filename):
    """
    Extract participant ID from filtered EDA filename.
    Example: "004-0-B-2-3.csv" -> "004"
    """
    name = filename.replace('.csv', '')
    match = re.match(r'(\d{3})', name)
    if match:
        return match.group(1)
    return None

def find_matching_stimulus_file(participant_num, stimulus_path):
    """
    Find the stimulus Excel file that matches the participant number.
    """
    for file in os.listdir(stimulus_path):
        if file.endswith('.xlsx'):
            # Extract first 3 digits from filename (handle patterns like 025b)
            match = re.match(r'(\d{3})', file)
            if match and match.group(1) == participant_num:
                return os.path.join(stimulus_path, file)
    return None

def parse_stimulus_timestamp(timestamp_str):
    """
    Parse timestamps from stimulus Excel files.
    Handles formats:
    - "11h06.22.845" -> "11:06:22.845"
    - "11h06,22,845" -> "11:06:22.845"
    - "11h06,22,845" with milliseconds
    """
    if pd.isna(timestamp_str):
        return None
    
    ts_str = str(timestamp_str).strip()
    
    # Replace 'h' with ':'
    ts_str = ts_str.replace('h', ':')
    
    # Replace commas with dots (for milliseconds)
    ts_str = ts_str.replace(',', '.')
    
    # Now we have something like "11:06.22.845"
    # Split by colon to get hours, then the rest
    parts = ts_str.split(':')
    
    if len(parts) >= 2:
        hours = parts[0]
        rest = parts[1]  # "06.22.845"
        
        # Split the rest by dot
        rest_parts = rest.split('.')
        
        if len(rest_parts) >= 3:
            minutes = rest_parts[0]
            seconds = rest_parts[1]
            milliseconds = rest_parts[2]
            # Pad milliseconds to 3 digits
            milliseconds = milliseconds.ljust(3, '0')[:3]
            time_str = f"{hours}:{minutes}:{seconds}.{milliseconds}"
        elif len(rest_parts) == 2:
            minutes = rest_parts[0]
            seconds = rest_parts[1]
            time_str = f"{hours}:{minutes}:{seconds}.000"
        else:
            return None
    else:
        return None
    
    return time_str

def load_stimulus_markers_from_excel(excel_path):
    """
    Load all sheets from the stimulus Excel file.
    Each sheet name represents a stimulus.
    Returns a list of (start_time, end_time, stimulus_name)
    """
    markers = []
    
    try:
        # Load all sheet names
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names
        
        print(f"    Found {len(sheet_names)} sheets: {sheet_names[:3]}...")
        
        for sheet_name in sheet_names:
            # Skip metadata sheets
            if sheet_name in ['Sheet1', 'Sheet', 'Metadata', 'Info']:
                continue
            
            # Read the sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            
            # Find the row where the actual data starts
            data_start_row = None
            for idx, row in df.iterrows():
                first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
                # Look for timestamp pattern (contains 'h' or ':' or '.')
                if ('h' in first_cell and any(c.isdigit() for c in first_cell)) or \
                   ('.' in first_cell and len(first_cell) > 5) or \
                   (',' in first_cell and len(first_cell) > 5):
                    data_start_row = idx
                    break
            
            if data_start_row is None:
                print(f"      WARNING: No timestamp data found in sheet '{sheet_name}'")
                continue
            
            # Read the actual data
            df_data = pd.read_excel(excel_path, sheet_name=sheet_name, 
                                    header=None, skiprows=data_start_row)
            
            # Extract timestamps from first column
            timestamps = []
            for idx, row in df_data.iterrows():
                ts = row.iloc[0]
                if pd.notna(ts):
                    parsed_ts = parse_stimulus_timestamp(ts)
                    if parsed_ts:
                        timestamps.append(parsed_ts)
            
            if len(timestamps) >= 2:
                start_time = timestamps[0]
                end_time = timestamps[-1]
                markers.append({
                    'stimulus_name': sheet_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'all_times': timestamps
                })
                print(f"      Sheet '{sheet_name}': {start_time} -> {end_time} ({len(timestamps)} samples)")
            
    except Exception as e:
        print(f"    ERROR loading {excel_path}: {str(e)}")
    
    return markers

def add_stimulus_labels_to_eda(eda_df, markers):
    """
    Add a new column 'stimulus_label' to the EDA dataframe.
    """
    # Initialize column
    eda_df['stimulus_label'] = 'baseline'
    eda_df['stimulus_start'] = False
    eda_df['stimulus_end'] = False
    
    # Convert Time column to string for comparison
    eda_df['Time_str'] = eda_df['Time'].astype(str)
    
    for marker in markers:
        stimulus_name = marker['stimulus_name']
        start_time = marker['start_time']
        end_time = marker['end_time']
        
        # Mark all rows between start and end (inclusive)
        mask = (eda_df['Time_str'] >= start_time) & (eda_df['Time_str'] <= end_time)
        eda_df.loc[mask, 'stimulus_label'] = stimulus_name
        
        # Mark the exact start and end rows
        start_mask = eda_df['Time_str'] == start_time
        end_mask = eda_df['Time_str'] == end_time
        eda_df.loc[start_mask, 'stimulus_start'] = True
        eda_df.loc[end_mask, 'stimulus_end'] = True
    
    # Drop temporary column
    eda_df = eda_df.drop(columns=['Time_str'])
    
    return eda_df

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main():
    print("=" * 80)
    print("STEP 2: Adding Stimulus Markers to Filtered EDA Files")
    print("=" * 80)
    print(f"Filtered EDA path: {FILTERED_EDA_PATH}")
    print(f"Stimulus markers path: {STIMULUS_MARKERS_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print("-" * 80)
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Get all filtered EDA files
    eda_files = [f for f in os.listdir(FILTERED_EDA_PATH) if f.endswith('.csv')]
    print(f"Found {len(eda_files)} filtered EDA files")
    print("-" * 80)
    
    successful = 0
    failed = 0
    no_marker_file = 0
    
    for eda_file in eda_files:
        print(f"\nProcessing: {eda_file}")
        
        # Extract participant number
        participant_num = extract_participant_id_from_folder(eda_file)
        if participant_num is None:
            print(f"  ERROR: Could not extract participant number from {eda_file}")
            failed += 1
            continue
        
        print(f"  Participant number: {participant_num}")
        
        # Find matching stimulus Excel file
        stimulus_excel = find_matching_stimulus_file(participant_num, STIMULUS_MARKERS_PATH)
        if stimulus_excel is None:
            print(f"  WARNING: No matching stimulus file found for participant {participant_num}")
            no_marker_file += 1
            continue
        
        print(f"  Found stimulus file: {os.path.basename(stimulus_excel)}")
        
        # Load filtered EDA data
        eda_path = os.path.join(FILTERED_EDA_PATH, eda_file)
        eda_df = pd.read_csv(eda_path)
        print(f"  Loaded {len(eda_df)} EDA samples")
        
        # Load stimulus markers from Excel
        markers = load_stimulus_markers_from_excel(stimulus_excel)
        
        if len(markers) == 0:
            print(f"  WARNING: No stimulus markers found in {stimulus_excel}")
            no_marker_file += 1
            continue
        
        print(f"  Loaded {len(markers)} stimulus markers")
        
        # Add stimulus labels to EDA dataframe
        eda_labeled = add_stimulus_labels_to_eda(eda_df, markers)
        
        # Count labeled samples
        labeled_count = len(eda_labeled[eda_labeled['stimulus_label'] != 'baseline'])
        print(f"  Labeled {labeled_count} samples as stimuli ({labeled_count/len(eda_labeled)*100:.1f}%)")
        
        # Save labeled file
        output_file = eda_file.replace('.csv', '_labeled.csv')
        output_path = os.path.join(OUTPUT_PATH, output_file)
        eda_labeled.to_csv(output_path, index=False)
        print(f"  SAVED: {output_file}")
        successful += 1
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total filtered EDA files: {len(eda_files)}")
    print(f"Successfully labeled: {successful}")
    print(f"No matching stimulus file: {no_marker_file}")
    print(f"Failed: {failed}")
    print(f"\nOutput saved to: {OUTPUT_PATH}")
    print("=" * 80)

# ============================================================================
# RUN THE SCRIPT
# ============================================================================

if __name__ == "__main__":
    main()