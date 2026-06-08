# -*- coding: utf-8 -*-
"""
ACCURATE STIMULATION SUMMARY - Based on EXACT 40-SECOND WINDOWS
===============================================================
Each stimulation = exactly 40 seconds (5120 samples at 128Hz)
"""

import os
import pandas as pd
import glob

# ============================================================================
# PATHS
# ============================================================================

STIM_40S_PATH = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_40s_stim'
OUTPUT_FILE = r'C:/Users/katia/Desktop/output-tnc-cleaned/stimulation_summary_all_participants.xlsx'

# ============================================================================
# TIME CONVERSION
# ============================================================================

def time_to_seconds(time_str):
    """Convert HH:MM:SS.mmm to seconds"""
    try:
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except:
        return 0.0

# ============================================================================
# EXTRACT STIMULATION INFO
# ============================================================================

def extract_stimulation_info(file_path, participant_id):
    """Extract information for each 40-second stimulation"""
    
    df = pd.read_csv(file_path)
    
    stimulations = []
    
    for stim_num in sorted(df['stimulus_number'].unique()):
        stim_df = df[df['stimulus_number'] == stim_num]
        
        if len(stim_df) > 0:
            start_time = stim_df['Time'].iloc[0]
            end_time = stim_df['Time'].iloc[-1]
            modality = stim_df['modality'].iloc[0] if 'modality' in stim_df.columns else 'unknown'
            stimulus_name = stim_df['stimulus_name'].iloc[0] if 'stimulus_name' in stim_df.columns else stim_df['stimulus_label'].iloc[0]
            
            start_sec = time_to_seconds(start_time)
            end_sec = time_to_seconds(end_time)
            duration = end_sec - start_sec
            
            stimulations.append({
                'participant_id': participant_id,
                'stimulus_number': stim_num,
                'stimulus_name': stimulus_name,
                'modality': modality,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': round(duration, 2),
                'n_samples': len(stim_df)
            })
    
    return stimulations

# ============================================================================
# PROCESS ALL PARTICIPANTS
# ============================================================================

print("="*80)
print("CREATING ACCURATE STIMULATION SUMMARY (40-second windows)")
print("="*80)

# Find all 40-second stimulation files
stim_files = glob.glob(os.path.join(STIM_40S_PATH, "*_labeled.csv"))
print(f"\n📁 Found {len(stim_files)} files with 40-second windows")

all_stimulations = []

for file_path in sorted(stim_files):
    filename = os.path.basename(file_path)
    participant_id = filename.replace('_labeled.csv', '')
    
    print(f"📊 Processing: {participant_id}")
    
    stim_info = extract_stimulation_info(file_path, participant_id)
    all_stimulations.extend(stim_info)
    
    print(f"   ✅ {len(stim_info)} stimulations, each {stim_info[0]['duration_seconds']}s")

# ============================================================================
# CREATE DATAFRAMES
# ============================================================================

print("\n" + "="*80)
print("CREATING EXCEL FILE")
print("="*80)

df_all = pd.DataFrame(all_stimulations)

# Reorder columns
column_order = ['participant_id', 'stimulus_number', 'stimulus_name', 'modality', 
                'start_time', 'end_time', 'duration_seconds', 'n_samples']
df_all = df_all[column_order]

print(f"\n✅ Total stimulations: {len(df_all)}")
print(f"   Expected: 40 participants × 14 stimuli = 560")
print(f"   Actual: {len(df_all)}")

# ============================================================================
# CREATE PIVOT TABLES
# ============================================================================

# Pivot: Duration by participant
pivot_duration = df_all.pivot_table(
    index='participant_id',
    columns='stimulus_number',
    values='duration_seconds',
    aggfunc='first'
)
pivot_duration.columns = [f'stimulus_{col}_duration_sec' for col in pivot_duration.columns]

# Pivot: Stimulus name by participant
pivot_name = df_all.pivot_table(
    index='participant_id',
    columns='stimulus_number',
    values='stimulus_name',
    aggfunc='first'
)
pivot_name.columns = [f'stimulus_{col}_name' for col in pivot_name.columns]

# Pivot: Modality by participant
pivot_modality = df_all.pivot_table(
    index='participant_id',
    columns='stimulus_number',
    values='modality',
    aggfunc='first'
)
pivot_modality.columns = [f'stimulus_{col}_modality' for col in pivot_modality.columns]

# ============================================================================
# STATISTICS
# ============================================================================

# Duration by modality
duration_by_modality = df_all.groupby('modality')['duration_seconds'].agg(['mean', 'std', 'min', 'max'])
duration_by_modality.columns = ['mean_sec', 'std_sec', 'min_sec', 'max_sec']

# Duration by stimulus name
duration_by_stimulus = df_all.groupby('stimulus_name')['duration_seconds'].agg(['mean', 'std', 'min', 'max'])
duration_by_stimulus = duration_by_stimulus.round(2)

# Summary by participant
summary_by_participant = df_all.groupby('participant_id').agg({
    'stimulus_number': 'count',
    'duration_seconds': 'sum',
    'n_samples': 'sum'
}).rename(columns={
    'stimulus_number': 'total_stimuli',
    'duration_seconds': 'total_duration_seconds',
    'n_samples': 'total_samples'
})

# ============================================================================
# SAVE TO EXCEL
# ============================================================================

with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    # Sheet 1: All stimulations detailed
    df_all.to_excel(writer, sheet_name='All_Stimulations_40s', index=False)
    
    # Sheet 2: Duration by participant (pivot)
    pivot_duration.to_excel(writer, sheet_name='Duration_by_Participant')
    
    # Sheet 3: Stimulus names by participant
    pivot_name.to_excel(writer, sheet_name='Stimulus_Names_by_Participant')
    
    # Sheet 4: Modality by participant
    pivot_modality.to_excel(writer, sheet_name='Modality_by_Participant')
    
    # Sheet 5: Statistics by modality
    duration_by_modality.to_excel(writer, sheet_name='Stats_by_Modality')
    
    # Sheet 6: Statistics by stimulus
    duration_by_stimulus.to_excel(writer, sheet_name='Stats_by_Stimulus')
    
    # Sheet 7: Summary by participant
    summary_by_participant.to_excel(writer, sheet_name='Summary_by_Participant')

print(f"\n✅ Saved to: {OUTPUT_FILE}")

# ============================================================================
# VERIFICATION
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION - Summary Statistics")
print("="*80)

print(f"\n📊 Total participants: {df_all['participant_id'].nunique()}")
print(f"📊 Total stimulations: {len(df_all)}")
print(f"📊 Expected: 560, Actual: {len(df_all)}")

print(f"\n📊 Duration Statistics:")
print(f"   Mean duration: {df_all['duration_seconds'].mean():.2f}s")
print(f"   Min duration: {df_all['duration_seconds'].min():.2f}s")
print(f"   Max duration: {df_all['duration_seconds'].max():.2f}s")
print(f"   Samples per stimulus: {df_all['n_samples'].mean():.0f} (at 128Hz = 40s)")

print(f"\n📊 Modality Distribution:")
modality_counts = df_all['modality'].value_counts()
for mod, count in modality_counts.items():
    print(f"   {mod}: {count} stimulations ({count/len(df_all)*100:.1f}%)")

print("\n" + "="*80)
print("✅ ACCURATE SUMMARY COMPLETE!")
print("="*80)
print(f"\n📁 Output: {OUTPUT_FILE}")
print("\nThe file now contains EXACT 40-second stimulation windows:")
print("   - Each stimulus = 40.00 seconds")
print("   - Each stimulus = 5120 samples (at 128Hz)")
print("   - No pre-stimulus baseline included")
print("   - No post-stimulus recovery included")