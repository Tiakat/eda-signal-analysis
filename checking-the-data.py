# -*- coding: utf-8 -*-
"""
ADD MODALITY COLUMN TO ALL LABELED FILES
========================================
Adds 'modality' column to each file with values:
- 'image' for visual stimuli
- 'music' for auditory stimuli  
- 'combined' for audiovisual stimuli
- 'none' for baseline

Processes all 40 files in the original folder
"""

import os
import pandas as pd
import glob

# ============================================================================
# PATHS
# ============================================================================

LABELED_BASE = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_labeled'

# ============================================================================
# MAPPING FOR MODALITY CLASSIFICATION
# ============================================================================

def get_modality(stimulus_label):
    """
    Classify stimulus_label into modality category
    """
    if stimulus_label == 'baseline':
        return 'none'
    
    label_lower = stimulus_label.lower()
    
    # IMAGE stimuli (visual only)
    image_keywords = ['regata', 'monet', 'mc_nicoll', 'bataille', 'fleurs']
    for keyword in image_keywords:
        if keyword in label_lower:
            # Make sure it's not combined (audio___tableau contains audio)
            if 'audio' not in label_lower:
                return 'image'
    
    # MUSIC stimuli (auditory only)
    music_keywords = ['audio_monet', 'audio_mc_nicoll', 'musique', 'routine_127_ep', 'routine_128_en']
    for keyword in music_keywords:
        if keyword in label_lower:
            # Make sure it's not combined (audio___tableau is combined)
            if 'audio___tableau' not in label_lower:
                return 'music'
    
    # COMBINED stimuli (audiovisual)
    combined_keywords = ['audio___tableau', 'routine_127_ep___tableau', 'routine_128_en__tableau']
    for keyword in combined_keywords:
        if keyword in label_lower:
            return 'combined'
    
    # Default (should not happen with clean data)
    return 'unknown'

# ============================================================================
# PROCESS ALL FILES
# ============================================================================

print("="*80)
print("ADDING MODALITY COLUMN TO 40 LABELED FILES")
print("="*80)
print(f"Folder: {LABELED_BASE}")
print("-"*80)

# Find all labeled files
labeled_files = glob.glob(os.path.join(LABELED_BASE, "*_labeled.csv"))
print(f"\n📁 Found {len(labeled_files)} labeled files")

successful = 0
failed = []

for file_path in sorted(labeled_files):
    filename = os.path.basename(file_path)
    participant_id = filename.replace('_labeled.csv', '')
    
    print(f"\n📊 Processing: {participant_id}")
    
    try:
        # Read file
        df = pd.read_csv(file_path)
        print(f"   Original columns: {df.columns.tolist()}")
        
        # Add modality column
        df['modality'] = df['stimulus_label'].apply(get_modality)
        
        # Reorder columns (put modality after stimulus_label)
        cols = df.columns.tolist()
        # Move modality to be after stimulus_label
        if 'stimulus_label' in cols and 'modality' in cols:
            cols.remove('modality')
            stim_idx = cols.index('stimulus_label')
            cols.insert(stim_idx + 1, 'modality')
            df = df[cols]
        
        # Verify all stimuli got classified
        unique_stimuli = df[df['stimulus_label'] != 'baseline']['stimulus_label'].unique()
        unique_modalities = df[df['stimulus_label'] != 'baseline'][['stimulus_label', 'modality']].drop_duplicates()
        
        print(f"   Columns after: {df.columns.tolist()}")
        print(f"   Unique stimuli: {len(unique_stimuli)}")
        
        # Show modality distribution
        modality_counts = df[df['modality'] != 'none']['modality'].value_counts()
        print(f"   Modality distribution:")
        for mod, count in modality_counts.items():
            samples = len(df[df['modality'] == mod])
            print(f"      {mod}: {count} unique stimuli, {samples:,} samples")
        
        # Save back to same file (overwrite)
        df.to_csv(file_path, index=False)
        print(f"   💾 Updated: {filename}")
        
        successful += 1
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        failed.append(participant_id)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"\n✅ Successful: {successful}/{len(labeled_files)}")
print(f"❌ Failed: {len(failed)}")

if failed:
    print(f"Failed participants: {failed}")

# ============================================================================
# VERIFICATION - Show modality mapping for first participant
# ============================================================================

print("\n" + "="*80)
print("VERIFICATION - Modality Mapping Example")
print("="*80)

# Check first file
if labeled_files:
    sample_file = labeled_files[0]
    df_sample = pd.read_csv(sample_file)
    
    print(f"\n📁 Sample from: {os.path.basename(sample_file)}")
    print("\nStimulus to Modality Mapping:")
    print("-"*50)
    
    mapping = df_sample[df_sample['stimulus_label'] != 'baseline'][['stimulus_label', 'modality']].drop_duplicates().sort_values('stimulus_label')
    
    for _, row in mapping.iterrows():
        print(f"   {row['stimulus_label']:30} → {row['modality']}")
    
    print("\n" + "-"*50)
    print(f"\nTotal unique stimuli: {len(mapping)}")
    print(f"  - Image: {len(mapping[mapping['modality'] == 'image'])}")
    print(f"  - Music: {len(mapping[mapping['modality'] == 'music'])}")
    print(f"  - Combined: {len(mapping[mapping['modality'] == 'combined'])}")

# ============================================================================
# FINAL VERIFICATION FOR ALL PARTICIPANTS
# ============================================================================

print("\n" + "="*80)
print("FINAL VERIFICATION - All 40 Participants")
print("="*80)

all_good = True
for participant in sorted([f.replace('_labeled.csv', '') for f in labeled_files])[:10]:  # Check first 10
    file_path = os.path.join(LABELED_BASE, f"{participant}_labeled.csv")
    df = pd.read_csv(file_path)
    
    if 'modality' not in df.columns:
        print(f"❌ {participant}: Missing modality column")
        all_good = False
    else:
        # Count modalities
        n_image = len(df[df['modality'] == 'image']['stimulus_label'].unique())
        n_music = len(df[df['modality'] == 'music']['stimulus_label'].unique())
        n_combined = len(df[df['modality'] == 'combined']['stimulus_label'].unique())
        n_none = len(df[df['modality'] == 'none'])
        
        print(f"\n✅ {participant}:")
        print(f"   Image: {n_image} stimuli, Music: {n_music} stimuli, Combined: {n_combined} stimuli")
        print(f"   Baseline samples: {n_none:,}")

if all_good:
    print("\n" + "="*80)
    print("🎉 PERFECT! All 40 files now have the 'modality' column!")
    print("="*80)
    print("\nColumn order: Time, GSR_raw, stimulus_label, modality")
    print("\nModality values:")
    print("   - 'image' for visual-only stimuli")
    print("   - 'music' for auditory-only stimuli")
    print("   - 'combined' for audiovisual stimuli")
    print("   - 'none' for baseline periods")
else:
    print("\n⚠️ Some files missing modality column - check errors above")

print("\n" + "="*80)
print("✅ DONE!")
print("="*80)