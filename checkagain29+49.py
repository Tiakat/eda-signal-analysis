# -*- coding: utf-8 -*-
"""
COMPLETE EDA GRAPHS - Participants 029 and 049 ONLY (FIXED)
============================================================
With signal cut at 1200 seconds (20 minutes)
All operations use the cut data, not original indices
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, ttest_ind

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

EDA_FOLDER = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_labeled'
OUTPUT_PATH = r'C:/Users/katia/Desktop/output-tnc-cleaned/comparison_analysis'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================================
# PARAMETERS
# ============================================================================

SAMPLING_RATE = 128
PRE_STIM_SEC = 5
POST_STIM_SEC = 5
PRE_SAMPLES = PRE_STIM_SEC * SAMPLING_RATE
POST_SAMPLES = POST_STIM_SEC * SAMPLING_RATE
CUT_TIME_SECONDS = 1200  # Keep only 0 to 1200 seconds (20 minutes)

# ============================================================================
# COLORS
# ============================================================================

BG_FIG = '#0d1117'
BG_AX = '#161b22'
GRID_C = '#21262d'
TEXT_C = '#e6edf3'
LABEL_C = '#8b949e'
SPINE_C = '#30363d'

C_STIMULI = ['#58a6ff', '#3fb950', '#ffa657', '#f85149', '#d2a8ff', 
             '#79c0ff', '#ff7b72', '#a5d6ff', '#7ee83f', '#ffb347',
             '#b381b3', '#ff6b9d', '#56d364', '#e3b341']
C_WHOLE = '#ffffff'

# ============================================================================
# STIMULUS FULL NAMES
# ============================================================================

STIMULUS_FULL_NAMES = {
    'regata': 'Regatta (Sisley) - Visual',
    'Monet': 'Monet - Visual',
    'Mc_nicoll': 'McNicoll (Sunny September) - Visual',
    'Bataille': 'Smardan Attack (Grigorescu) - Visual',
    'Fleurs': 'Fleurs - Visual',
    'Audio_monet': 'Regatta (Sisley) - Auditory',
    'Audio_mc_nicoll': 'McNicoll - Auditory',
    'Musique__test_EN': 'Smardan Attack - Auditory',
    'routine_127_EP': 'Routine 127 - Auditory',
    'routine_128_EN': 'Routine 128 - Auditory',
    'Audio___tableau_monet': 'Regatta (Sisley) - Combined',
    'Audio___tableau_Mc_nicoll': 'McNicoll - Combined',
    'routine_127_EP___tableau_ragata': 'Routine 127 + Regatta - Combined',
    'routine_128_EN__tableau_bataill': 'Routine 128 + Smardan - Combined'
}

# ============================================================================
# TIME CONVERSION
# ============================================================================

def time_to_seconds(ts):
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        parts = str(ts).split(':')
        if len(parts) == 3:
            h, m = int(parts[0]), int(parts[1])
            sp = parts[2].split('.')
            s = int(sp[0])
            ms = int(sp[1]) if len(sp) > 1 else 0
            return h * 3600 + m * 60 + s + ms / 1000
        return float(ts)
    except:
        return 0.0

def get_full_name(short_name):
    return STIMULUS_FULL_NAMES.get(short_name, short_name)

# ============================================================================
# LOAD AND CUT DATA - Return cut version only
# ============================================================================

def load_and_cut_participant(participant_id):
    """Load GSR signal, cut at CUT_TIME_SECONDS, return cut data only"""
    file_path = os.path.join(EDA_FOLDER, f"{participant_id}_labeled.csv")
    
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    # Get time array
    if 'Time' in df.columns:
        if isinstance(df['Time'].iloc[0], str) and ':' in str(df['Time'].iloc[0]):
            t = df['Time'].apply(time_to_seconds).values
            t = t - t[0]
        else:
            t = df['Time'].values.astype(float)
    else:
        t = np.arange(len(df)) / SAMPLING_RATE
    
    # Fix negative time if needed
    if len(t) > 0 and t[-1] < 0:
        t = t - t[0]
    
    # Get GSR signal
    gsr = None
    for col in ['GSR_raw', 'GSR_filtered', 'GSR']:
        if col in df.columns:
            gsr = df[col].values.astype(float)
            break
    
    if gsr is None:
        return None
    
    # CUT at CUT_TIME_SECONDS
    cut_mask = t <= CUT_TIME_SECONDS
    t_cut = t[cut_mask]
    gsr_cut = gsr[cut_mask]
    
    # Also cut the dataframe for stimulus labels
    df_cut = df.iloc[:len(t_cut)].copy()
    
    print(f"  Original duration: {t[-1]:.1f}s ({t[-1]/60:.1f} min)")
    print(f"  Cut duration: {t_cut[-1]:.1f}s ({t_cut[-1]/60:.1f} min)")
    
    return df_cut, t_cut, gsr_cut

# ============================================================================
# EXTRACT STIMULATION WINDOWS FROM CUT DATA
# ============================================================================

def extract_stimulation_windows(df, t):
    """Extract each stimulation with 5s before and 5s after from cut data"""
    stim_windows = []
    
    current_stim = None
    start_idx = None
    
    for idx, row in df.iterrows():
        stim = row['stimulus_label']
        if stim != 'baseline' and stim != current_stim:
            if current_stim is not None and start_idx is not None:
                end_idx = idx - 1
                pre_start = max(0, start_idx - PRE_SAMPLES)
                post_end = min(len(df) - 1, end_idx + POST_SAMPLES)
                stim_windows.append({
                    'number': len(stim_windows) + 1,
                    'name': current_stim,
                    'full_name': get_full_name(current_stim),
                    'start_idx': pre_start,
                    'end_idx': post_end,
                    'start_time': t[pre_start],
                    'end_time': t[post_end]
                })
            current_stim = stim
            start_idx = idx
    
    # Add last stimulus
    if current_stim is not None and start_idx is not None:
        pre_start = max(0, start_idx - PRE_SAMPLES)
        post_end = len(df) - 1
        stim_windows.append({
            'number': len(stim_windows) + 1,
            'name': current_stim,
            'full_name': get_full_name(current_stim),
            'start_idx': pre_start,
            'end_idx': post_end,
            'start_time': t[pre_start],
            'end_time': t[post_end]
        })
    
    return stim_windows

# ============================================================================
# GRAPH 1: PER PARTICIPANT (cut version)
# ============================================================================

def plot_participant_cut(participant_id, df, t, gsr, output_path):
    """Generate graph for one participant with cut signal"""
    
    stim_windows = extract_stimulation_windows(df, t)
    
    if len(stim_windows) == 0:
        print(f"  No stimulations found")
        return None
    
    print(f"  Found {len(stim_windows)} stimulations in cut window")
    
    # Extract metrics
    stim_means = []
    stim_maxs = []
    for window in stim_windows:
        seg_gsr = gsr[window['start_idx']:window['end_idx']+1]
        stim_means.append(np.mean(seg_gsr))
        stim_maxs.append(np.max(seg_gsr))
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG_FIG)
    
    # Subplot 1: Time series
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_facecolor(BG_AX)
    
    # Plot whole experiment (cut)
    ax1.plot(t, gsr, color=C_WHOLE, linewidth=1, alpha=0.3, label='Whole experiment')
    
    # Plot each stimulation
    for i, window in enumerate(stim_windows):
        seg_t = t[window['start_idx']:window['end_idx']+1]
        seg_gsr = gsr[window['start_idx']:window['end_idx']+1]
        ax1.plot(seg_t, seg_gsr, color=C_STIMULI[i % len(C_STIMULI)], linewidth=1.5, alpha=0.8,
                label=f'Stim {window["number"]}: {window["full_name"][:25]}')
        
        # Shade the actual stimulation period
        stim_start = window['start_time'] + PRE_STIM_SEC
        stim_end = window['end_time'] - POST_STIM_SEC
        ax1.axvspan(stim_start, stim_end, alpha=0.08, color=C_STIMULI[i % len(C_STIMULI)], zorder=0)
    
    # X-axis ticks every 100 seconds
    max_time = t[-1]
    major_ticks = np.arange(0, max_time + 100, 100)
    ax1.set_xticks(major_ticks)
    ax1.set_xticklabels([f'{int(tick)}' for tick in major_ticks], rotation=45, ha='right', fontsize=8, color=LABEL_C)
    
    ax1.set_xlabel('Time (seconds)', color=LABEL_C, fontsize=11)
    ax1.set_ylabel('Skin Conductance (µS)', color=LABEL_C, fontsize=11)
    ax1.set_title(f'Participant {participant_id}: All Stimulations vs Whole Experiment (cut at {CUT_TIME_SECONDS}s)', 
                  color=TEXT_C, fontsize=14, fontweight='bold')
    ax1.tick_params(colors=LABEL_C)
    ax1.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', facecolor=BG_AX, edgecolor=SPINE_C, labelcolor=TEXT_C, fontsize=7, ncol=2)
    ax1.set_xlim(0, max_time)
    
    # Subplot 2: Bar chart
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_facecolor(BG_AX)
    
    x = np.arange(len(stim_windows))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, stim_means, width, label='Mean', color='#58a6ff', alpha=0.8, edgecolor='white')
    bars2 = ax2.bar(x + width/2, stim_maxs, width, label='Max', color='#3fb950', alpha=0.8, edgecolor='white')
    
    # Whole experiment reference lines
    whole_mean = np.mean(gsr)
    whole_max = np.max(gsr)
    ax2.axhline(y=whole_mean, color='#58a6ff', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Whole mean: {whole_mean:.2f}')
    ax2.axhline(y=whole_max, color='#3fb950', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Whole max: {whole_max:.2f}')
    
    # Value labels
    for bar, val in zip(bars1, stim_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=7, color=LABEL_C)
    for bar, val in zip(bars2, stim_maxs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}', 
                ha='center', va='bottom', fontsize=7, color=LABEL_C)
    
    ax2.set_xlabel('Stimulation Number', color=LABEL_C, fontsize=11)
    ax2.set_ylabel('Amplitude (µS)', color=LABEL_C, fontsize=11)
    ax2.set_title(f'Participant {participant_id}: Mean and Max per Stimulation (cut at {CUT_TIME_SECONDS}s)', 
                  color=TEXT_C, fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{w['number']}\n{w['name'][:12]}" for w in stim_windows], rotation=45, ha='right', fontsize=8, color=LABEL_C)
    ax2.legend(loc='upper right', facecolor=BG_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)
    ax2.tick_params(colors=LABEL_C)
    ax2.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')
    
    plt.tight_layout()
    
    # Create participant folder
    participant_folder = os.path.join(output_path, f"Participant_{participant_id}")
    os.makedirs(participant_folder, exist_ok=True)
    
    plt.savefig(os.path.join(participant_folder, f'{participant_id}_graph_cut_{CUT_TIME_SECONDS}s.png'), 
                dpi=150, facecolor=BG_FIG, bbox_inches='tight')
    plt.close()
    
    # Also save cut data
    cut_df = pd.DataFrame({'Time': t, 'GSR': gsr})
    cut_df.to_csv(os.path.join(participant_folder, f'cut_at_{CUT_TIME_SECONDS}s.csv'), index=False)
    
    print(f"  Saved: {participant_id}_graph_cut_{CUT_TIME_SECONDS}s.png")
    print(f"  Saved: cut_at_{CUT_TIME_SECONDS}s.csv")
    
    return stim_means, stim_maxs

# ============================================================================
# COMPARISON GRAPH
# ============================================================================

def plot_comparison(all_data, output_path):
    """Generate comparison graph between 029 and 049"""
    
    if len(all_data) != 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BG_FIG)
    ax.set_facecolor(BG_AX)
    
    stim_numbers = list(range(1, len(all_data['029']['means']) + 1))
    
    ax.plot(stim_numbers, all_data['029']['means'], 'o-', color='#58a6ff', linewidth=2, markersize=8, label='Participant 029')
    ax.plot(stim_numbers, all_data['049']['means'], 's-', color='#f85149', linewidth=2, markersize=8, label='Participant 049')
    
    # Add t-test result
    t_stat, p_val = ttest_ind(all_data['029']['means'], all_data['049']['means'])
    ax.text(0.5, 0.95, f't-test: t={t_stat:.3f}, p={p_val:.4f}', transform=ax.transAxes,
            ha='center', va='top', fontsize=11, color=TEXT_C,
            bbox=dict(boxstyle='round', facecolor=BG_AX, edgecolor=SPINE_C))
    
    ax.set_xlabel('Stimulus Number', color=LABEL_C, fontsize=12)
    ax.set_ylabel('Mean SCL during 40s stimulation (µS)', color=LABEL_C, fontsize=12)
    ax.set_title(f'Comparison: Participant 029 vs 049 (cut at {CUT_TIME_SECONDS}s)', 
                 color=TEXT_C, fontsize=14, fontweight='bold')
    ax.set_xticks(stim_numbers)
    ax.tick_params(colors=LABEL_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', facecolor=BG_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'COMPARISON_029_VS_049_CUT.png'), 
                dpi=150, facecolor=BG_FIG, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: COMPARISON_029_VS_049_CUT.png")
    print(f"  t-test: t={t_stat:.3f}, p={p_val:.4f}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print(f"EDA GRAPHS - Participants 029 and 049 ONLY (cut at {CUT_TIME_SECONDS}s)")
    print("="*80)
    print(f"Output: {OUTPUT_PATH}")
    print("-"*80)
    
    target_participants = ['029', '049']
    all_data = {}
    
    for participant_id in target_participants:
        print(f"\n📊 Processing: {participant_id}")
        
        result = load_and_cut_participant(participant_id)
        if result is None:
            print(f"  ❌ Failed to load")
            continue
        
        df, t, gsr = result
        print(f"  Cut data: {len(t)} rows, {t[-1]:.1f}s duration")
        
        stim_means, stim_maxs = plot_participant_cut(participant_id, df, t, gsr, OUTPUT_PATH)
        
        if stim_means:
            all_data[participant_id] = {'means': stim_means, 'maxs': stim_maxs}
            print(f"  ✅ Completed")
    
    # Generate comparison graph
    if len(all_data) == 2:
        print("\n" + "="*80)
        print("GENERATING COMPARISON GRAPH")
        print("="*80)
        plot_comparison(all_data, OUTPUT_PATH)
    
    print("\n" + "="*80)
    print("✅ ALL GRAPHS GENERATED!")
    print("="*80)
    print(f"\n📁 Output folder: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()