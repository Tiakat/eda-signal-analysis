"""
CORRECTED ANALYSIS: Compare by STIMULUS NAME, not by order
===========================================================
Stimuli are compared by their actual name (e.g., all "Fleurs" together),
not by presentation order (which varies between participants).
ALL 40 participants included for each stimulus they saw.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, kruskal

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

EDA_FOLDER = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter_labeled"
OUTPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\comparison_analysis"
CORRECTED_FOLDER = os.path.join(OUTPUT_PATH, "CORRECTED_BY_STIMULUS_NAME")
os.makedirs(CORRECTED_FOLDER, exist_ok=True)

# ============================================================================
# PARAMETERS
# ============================================================================

SAMPLING_RATE = 128
PRE_SAMPLES = 5 * SAMPLING_RATE
POST_SAMPLES = 5 * SAMPLING_RATE

# ============================================================================
# COLORS
# ============================================================================

BG_FIG = '#0d1117'
BG_AX = '#161b22'
GRID_C = '#21262d'
TEXT_C = '#e6edf3'
LABEL_C = '#8b949e'
SPINE_C = '#30363d'

# Standardized names for display
STANDARD_NAMES = {
    'Fleurs': 'Image_Fleurs',
    'Musique__test_EN': 'Music_Smardan',
    'Musique__test_EN_2': 'Music_Smardan',
    'regata': 'Image_Regattas',
    'Monet': 'Image_Monet',
    'Mc_nicoll': 'Image_McNicoll',
    'Bataille': 'Image_Smardan',
    'Audio_monet': 'Music_Regattas',
    'Audio_mc_nicoll': 'Music_McNicoll',
    'routine_127_EP': 'Music_Routine127',
    'routine_128_EN': 'Music_Routine128',
    'Audio___tableau_monet': 'Combined_Regattas',
    'Audio___tableau_monet_2': 'Combined_Regattas',
    'Audio___tableau_Mc_nicoll': 'Combined_McNicoll',
    'Audio___tableau_Mc_nicoll_2': 'Combined_McNicoll',
    'routine_127_EP___tableau_ragata': 'Combined_Routine127_Regattas',
    'routine_128_EN__tableau_bataill': 'Combined_Routine128_Smardan'
}

MODALITY = {
    'Image_Fleurs': 'image', 'Image_Regattas': 'image', 'Image_Monet': 'image',
    'Image_McNicoll': 'image', 'Image_Smardan': 'image',
    'Music_Regattas': 'music', 'Music_McNicoll': 'music', 'Music_Smardan': 'music',
    'Music_Routine127': 'music', 'Music_Routine128': 'music',
    'Combined_Regattas': 'combined', 'Combined_McNicoll': 'combined',
    'Combined_Routine127_Regattas': 'combined', 'Combined_Routine128_Smardan': 'combined'
}

COLORS_BY_MODALITY = {'image': '#58a6ff', 'music': '#3fb950', 'combined': '#f85149'}

# ============================================================================
# EXTRACTION FUNCTION
# ============================================================================

def get_stimulus_mean_by_name(participant_id, target_stimulus_name):
    """Extract mean SCL for a specific stimulus BY NAME (not by order)"""
    
    file_path = os.path.join(EDA_FOLDER, f"{participant_id}_labeled.csv")
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Find GSR column
    gsr_col = None
    for col in ['GSR_raw', 'GSR_filtered', 'GSR']:
        if col in df.columns:
            gsr_col = col
            break
    if gsr_col is None:
        return None
    
    # Find the stimulus by name
    stim_start = -1
    stim_end = -1
    
    for idx in range(len(df)):
        stim = df.iloc[idx]['stimulus_label']
        if stim == target_stimulus_name:
            stim_start = idx
            # Find where this stimulus ends
            for j in range(idx, len(df)):
                if df.iloc[j]['stimulus_label'] != stim:
                    stim_end = j - 1
                    break
            if stim_end == -1:
                stim_end = len(df) - 1
            break
    
    if stim_start == -1:
        return None
    
    # Add 5s pre and post
    pre_start = max(0, stim_start - PRE_SAMPLES)
    post_end = min(len(df) - 1, stim_end + POST_SAMPLES)
    
    seg_gsr = df[gsr_col].values[pre_start:post_end+1]
    return np.mean(seg_gsr)

# ============================================================================
# COLLECT DATA BY STIMULUS NAME
# ============================================================================

print("="*80)
print("CORRECTED ANALYSIS: Comparing by STIMULUS NAME, not by order")
print("="*80)

# Get all participants
csv_files = glob.glob(os.path.join(EDA_FOLDER, "*_labeled.csv"))
participants = sorted([os.path.basename(f).replace('_labeled.csv', '') for f in csv_files])
print(f"\n📁 Total participants: {len(participants)}")

# Get all unique stimulus names from all participants
all_stimulus_names = set()
for participant_id in participants:
    file_path = os.path.join(EDA_FOLDER, f"{participant_id}_labeled.csv")
    df = pd.read_csv(file_path)
    stimuli = df[df['stimulus_label'] != 'baseline']['stimulus_label'].unique()
    all_stimulus_names.update(stimuli)

# Standardize names
standardized_groups = {}
for name in all_stimulus_names:
    std_name = STANDARD_NAMES.get(name, name)
    if std_name not in standardized_groups:
        standardized_groups[std_name] = []
    standardized_groups[std_name].append(name)

print(f"\n📊 Standardized stimulus groups ({len(standardized_groups)} unique stimuli):")
for std_name, variants in sorted(standardized_groups.items()):
    print(f"   {std_name}: {variants}")

# ============================================================================
# COLLECT DATA FOR EACH STANDARDIZED STIMULUS
# ============================================================================

print("\n⏳ Extracting SCL values for each stimulus across all participants...")

stimulus_data = {}
for std_name, variants in standardized_groups.items():
    values = []
    participant_list = []
    for participant_id in participants:
        for variant in variants:
            mean_val = get_stimulus_mean_by_name(participant_id, variant)
            if mean_val is not None:
                values.append(mean_val)
                participant_list.append(participant_id)
                break  # Found one variant for this participant
    stimulus_data[std_name] = {
        'values': np.array(values),
        'participants': participant_list,
        'n': len(values),
        'modality': MODALITY.get(std_name, 'unknown')
    }
    print(f"   {std_name}: n={len(values)} participants")

# ============================================================================
# APPLY NORMALIZATION PER STIMULUS
# ============================================================================

print("\n📊 Applying range normalization per stimulus...")

for std_name, data in stimulus_data.items():
    values = data['values']
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val > 0:
        data['normalized'] = (values - min_val) / (max_val - min_val)
    else:
        data['normalized'] = values
    data['min_raw'] = min_val
    data['max_raw'] = max_val

# ============================================================================
# GRAPH 1: NORMALIZED_SCL_PER_STIMULUS (ALL participants)
# ============================================================================

print("\n📈 Creating Graph 1: Normalized SCL per stimulus (ALL participants)...")

# Sort stimuli by modality and name
image_stimuli = [(name, data) for name, data in stimulus_data.items() if data['modality'] == 'image']
music_stimuli = [(name, data) for name, data in stimulus_data.items() if data['modality'] == 'music']
combined_stimuli = [(name, data) for name, data in stimulus_data.items() if data['modality'] == 'combined']

sorted_stimuli = sorted(image_stimuli, key=lambda x: x[0]) + sorted(music_stimuli, key=lambda x: x[0]) + sorted(combined_stimuli, key=lambda x: x[0])

n_stimuli = len(sorted_stimuli)
n_cols = 4
n_rows = (n_stimuli + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
fig.patch.set_facecolor(BG_FIG)

if n_rows == 1:
    axes = axes.reshape(1, -1)
axes_flat = axes.flatten()

for idx, (stim_name, data) in enumerate(sorted_stimuli):
    ax = axes_flat[idx]
    ax.set_facecolor(BG_AX)
    
    # Sort by normalized value
    sorted_indices = np.argsort(data['normalized'])
    sorted_participants = [data['participants'][i] for i in sorted_indices]
    sorted_values = data['normalized'][sorted_indices]
    
    bars = ax.bar(range(len(sorted_participants)), sorted_values, 
                  color=COLORS_BY_MODALITY[data['modality']], alpha=0.7, edgecolor='white')
    
    # Highlight max and min
    if len(sorted_values) > 0:
        bars[np.argmax(sorted_values)].set_color('#3fb950')
        bars[np.argmin(sorted_values)].set_color('#f85149')
    
    ax.set_xticks(range(len(sorted_participants)))
    ax.set_xticklabels(sorted_participants, rotation=90, fontsize=6, color=LABEL_C)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalized SCL', color=LABEL_C, fontsize=9)
    ax.set_title(f'{stim_name}\n({data["modality"]}, n={len(sorted_participants)})', 
                 color=TEXT_C, fontsize=10, fontweight='bold')
    ax.tick_params(colors=LABEL_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')
    
    # Add stats
    mean_val = np.mean(data['normalized'])
    std_val = np.std(data['normalized'])
    ax.text(0.02, 0.95, f'Mean: {mean_val:.2f}±{std_val:.2f}', transform=ax.transAxes,
            fontsize=7, color=TEXT_C, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor=BG_AX, alpha=0.8))

# Hide unused subplots
for idx in range(n_stimuli, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.suptitle(f'NORMALIZED SCL PER STIMULUS (Range-corrected)\nGreen=Max, Red=Min | ALL {len(participants)} PARTICIPANTS', 
             color=TEXT_C, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(CORRECTED_FOLDER, 'NORMALIZED_SCL_PER_STIMULUS.png'), 
            dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("   ✅ Saved: NORMALIZED_SCL_PER_STIMULUS.png")

# ============================================================================
# GRAPH 2: BOXPLOT BY STIMULUS (ALL participants)
# ============================================================================

print("\n📈 Creating Graph 2: Boxplot by stimulus...")

fig, ax = plt.subplots(figsize=(16, 10))
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AX)

stimulus_names = [name for name, _ in sorted_stimuli]
box_data = [data['normalized'] for _, data in sorted_stimuli]
colors = [COLORS_BY_MODALITY[data['modality']] for _, data in sorted_stimuli]

bp = ax.boxplot(box_data, labels=stimulus_names, patch_artist=True, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': '#3fb950', 'markeredgecolor': 'white'})

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')

ax.set_ylabel('Normalized SCL (range-corrected)', color=LABEL_C, fontsize=12)
ax.set_title(f'NORMALIZED SCL BY STIMULUS NAME\nALL {len(participants)} PARTICIPANTS', 
             color=TEXT_C, fontsize=14, fontweight='bold')
ax.tick_params(colors=LABEL_C, rotation=45)
ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(CORRECTED_FOLDER, 'BOXPLOT_BY_STIMULUS_NAME.png'), 
            dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("   ✅ Saved: BOXPLOT_BY_STIMULUS_NAME.png")

# ============================================================================
# GRAPH 3: BOXPLOT BY MODALITY (Image vs Music vs Combined)
# ============================================================================

print("\n📈 Creating Graph 3: Boxplot by modality...")

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AX)

# Collect all normalized values by modality
modality_values = {'image': [], 'music': [], 'combined': []}
for name, data in stimulus_data.items():
    modality_values[data['modality']].extend(data['normalized'])

box_data_modality = [modality_values['image'], modality_values['music'], modality_values['combined']]
bp_mod = ax.boxplot(box_data_modality, labels=['Image', 'Music', 'Combined'], 
                    patch_artist=True, showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': '#3fb950', 'markeredgecolor': 'white'})

for patch, modality in zip(bp_mod['boxes'], ['image', 'music', 'combined']):
    patch.set_facecolor(COLORS_BY_MODALITY[modality])
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')

# Add ANOVA test
f_stat, p_val = f_oneway(modality_values['image'], modality_values['music'], modality_values['combined'])
ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.3f}, p={p_val:.4f}', transform=ax.transAxes,
        ha='center', va='top', fontsize=11, color=TEXT_C,
        bbox=dict(boxstyle='round', facecolor=BG_AX, edgecolor=SPINE_C))

ax.set_ylabel('Normalized SCL', color=LABEL_C, fontsize=12)
ax.set_title(f'NORMALIZED SCL BY MODALITY\nALL {len(participants)} PARTICIPANTS', 
             color=TEXT_C, fontsize=14, fontweight='bold')
ax.tick_params(colors=LABEL_C)
ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(CORRECTED_FOLDER, 'BOXPLOT_BY_MODALITY.png'), 
            dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("   ✅ Saved: BOXPLOT_BY_MODALITY.png")

# ============================================================================
# STATISTICS
# ============================================================================

print("\n📊 Computing statistics...")

# Per stimulus statistics
stats_data = []
for name, data in sorted_stimuli:
    stats_data.append({
        'stimulus_name': name,
        'modality': data['modality'],
        'n_participants': data['n'],
        'mean_raw_µS': np.mean(data['values']),
        'std_raw_µS': np.std(data['values']),
        'min_raw_µS': np.min(data['values']),
        'max_raw_µS': np.max(data['values']),
        'mean_normalized': np.mean(data['normalized']),
        'std_normalized': np.std(data['normalized'])
    })

df_stats = pd.DataFrame(stats_data)
df_stats.to_csv(os.path.join(CORRECTED_FOLDER, 'statistics_per_stimulus.csv'), index=False)

# ANOVA by modality
f_stat, p_val = f_oneway(modality_values['image'], modality_values['music'], modality_values['combined'])
kw_stat, kw_p = kruskal(modality_values['image'], modality_values['music'], modality_values['combined'])

anova_results = pd.DataFrame([
    {'test': 'ANOVA (by modality)', 'statistic': f_stat, 'p_value': p_val},
    {'test': 'Kruskal-Wallis (by modality)', 'statistic': kw_stat, 'p_value': kw_p}
])
anova_results.to_csv(os.path.join(CORRECTED_FOLDER, 'anova_by_modality.csv'), index=False)

print("   ✅ Statistics saved")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY - CORRECTED ANALYSIS")
print("="*80)
print(f"\n✅ Total participants: {len(participants)}")
print(f"✅ Total unique stimuli (standardized): {len(stimulus_data)}")
print(f"\n📊 Stimulus counts by modality:")
print(f"   Image: {len([s for s in stimulus_data.values() if s['modality'] == 'image'])} stimuli")
print(f"   Music: {len([s for s in stimulus_data.values() if s['modality'] == 'music'])} stimuli")
print(f"   Combined: {len([s for s in stimulus_data.values() if s['modality'] == 'combined'])} stimuli")

print(f"\n📊 ANOVA by modality: F={f_stat:.3f}, p={p_val:.4f}")

print("\n" + "="*80)
print(f"✅ COMPLETE! Output: {CORRECTED_FOLDER}")
print("="*80)
print("\nGenerated files:")
print("  NORMALIZED_SCL_PER_STIMULUS.png - Each stimulus shown separately")
print("  BOXPLOT_BY_STIMULUS_NAME.png - All stimuli together")
print("  BOXPLOT_BY_MODALITY.png - Image vs Music vs Combined")
print("  statistics_per_stimulus.csv")
print("  anova_by_modality.csv")