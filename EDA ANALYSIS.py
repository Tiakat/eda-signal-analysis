# -*- coding: utf-8 -*-
"""
ANALYSE COMPLETE - TOUS LES PARTICIPANTS (004 a 064)
Sans tqdm - avec compteur simple
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import linregress, f_oneway

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
PRE_SEC = 5
POST_SEC = 5
PRE_SAMPLES = PRE_SEC * SAMPLING_RATE
POST_SAMPLES = POST_SEC * SAMPLING_RATE

# ============================================================================
# NOMS DES STIMULATIONS
# ============================================================================

STIM_NAMES_IN_ORDER = [
    '1: Fleurs',
    '2: Smardan (Musique)',
    '3: McNicoll (Image)',
    '4: Smardan (Image)',
    '5: Regatta (Image)',
    '6: Monet (Image)',
    '7: Regatta (Musique)',
    '8: Routine 127',
    '9: Routine 128',
    '10: McNicoll (Musique)',
    '11: Regatta (Combine)',
    '12: McNicoll (Combine)',
    '13: Routine127+Regatta',
    '14: Routine128+Smardan'
]

# ============================================================================
# COULEURS
# ============================================================================

BG = '#0d1117'
BG_AX = '#161b22'
TEXT = '#e6edf3'
LABEL = '#8b949e'
GRID = '#21262d'

# ============================================================================
# FONCTIONS
# ============================================================================

def to_seconds(t):
    try:
        if isinstance(t, (int, float)):
            return float(t)
        parts = str(t).split(':')
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        return float(t)
    except:
        return 0.0

def load_participant(pid):
    path = os.path.join(EDA_FOLDER, f"{pid}_labeled.csv")
    if not os.path.exists(path):
        return None
    
    df = pd.read_csv(path)
    t = df['Time'].apply(to_seconds).values
    t = t - t[0]
    if t[-1] < 0:
        t = np.abs(t)
        t = t - t[0]
    
    gsr = None
    for col in ['GSR_raw', 'GSR_filtered', 'GSR']:
        if col in df.columns:
            gsr = df[col].values.astype(float)
            break
    
    return df, t, gsr

def get_stimuli_order(df):
    stimuli = []
    current = None
    for _, row in df.iterrows():
        s = row['stimulus_label']
        if s != 'baseline' and s != current:
            stimuli.append(s)
            current = s
    return stimuli[:14]

def find_stimulus(df, stim_name):
    start = None
    end = None
    for i in range(len(df)):
        if df.iloc[i]['stimulus_label'] == stim_name:
            start = i
            for j in range(i, len(df)):
                if df.iloc[j]['stimulus_label'] != stim_name:
                    end = j - 1
                    break
            if end is None:
                end = len(df) - 1
            break
    return start, end

def extract_stimulus_mean(df, t, gsr, stim_name):
    start, end = find_stimulus(df, stim_name)
    if start is None:
        return None
    stim_gsr = gsr[start:end+1]
    if len(stim_gsr) == 0:
        return None
    return np.mean(stim_gsr)

# ============================================================================
# COLLECTE DES DONNEES
# ============================================================================

def collect_all_data():
    all_participants = ['004', '005', '008', '009', '010', '011', '012', '016', 
                        '018', '020', '022', '024', '025b', '028', '029', '031', 
                        '032', '033', '034', '035', '036', '037', '038', '039', 
                        '041', '042', '043', '044', '046', '047b', '049', '053', 
                        '054', '055', '057', '058', '060', '061', '062', '064']
    
    all_data = []
    total = len(all_participants)
    
    for i, pid in enumerate(all_participants, 1):
        print(f"  [{i}/{total}] Chargement: {pid}")
        
        data = load_participant(pid)
        if data is None:
            print(f"    Ignore: {pid} non charge")
            continue
        
        df, t, gsr = data
        stimuli = get_stimuli_order(df)
        
        if len(stimuli) != 14:
            print(f"    Ignore: {pid} seulement {len(stimuli)} stimulations")
            continue
        
        for order, stim_name in enumerate(stimuli, 1):
            mean_val = extract_stimulus_mean(df, t, gsr, stim_name)
            if mean_val is not None:
                all_data.append({
                    'participant_id': pid,
                    'stimulus_order': order,
                    'stimulus_name': stim_name,
                    'raw_mean': mean_val
                })
        
        print(f"    OK: {len(stimuli)} stimulations")
    
    df_all = pd.DataFrame(all_data)
    return df_all

# ============================================================================
# GRAPHIQUE PRINCIPAL
# ============================================================================

def plot_main(df_all, output_path):
    # Normalisation par participant
    df_all['normalized_mean'] = df_all.groupby('participant_id')['raw_mean'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() - x.min() > 0 else x
    )
    
    # Moyenne par stimulation
    stim_summary = df_all.groupby('stimulus_order')['normalized_mean'].agg(['mean', 'std', 'count']).reset_index()
    
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_AX)
    
    # Participants individuels avec leurs numeros
    for pid in df_all['participant_id'].unique():
        df_p = df_all[df_all['participant_id'] == pid]
        for _, row in df_p.iterrows():
            ax.scatter(row['stimulus_order'], row['normalized_mean'], 
                      s=40, color='#58a6ff', alpha=0.3, edgecolors='white', linewidth=0.3)
            ax.annotate(str(pid), (row['stimulus_order'], row['normalized_mean']),
                       fontsize=5, ha='center', va='center', color='white', alpha=0.5)
    
    # Moyenne
    ax.plot(stim_summary['stimulus_order'], stim_summary['mean'], 
            color='#f85149', linewidth=2.5, marker='s', markersize=8, label='Moyenne')
    ax.errorbar(stim_summary['stimulus_order'], stim_summary['mean'], 
                yerr=stim_summary['std'], color='#f85149', linewidth=1, capsize=3, alpha=0.5)
    
    # Tendance
    slope, intercept, r, p, _ = linregress(stim_summary['stimulus_order'], stim_summary['mean'])
    trend_line = slope * stim_summary['stimulus_order'] + intercept
    ax.plot(stim_summary['stimulus_order'], trend_line, '--', color='#3fb950', linewidth=2,
           label=f'Tendance: pente={slope:.4f}, p={p:.4f}')
    
    ax.set_xticks(range(1, 15))
    ax.set_xticklabels(STIM_NAMES_IN_ORDER, rotation=45, ha='right', fontsize=8, color=LABEL)
    
    if slope > 0 and p < 0.05:
        trend_text = 'SENSIBILISATION - Augmentation significative'
    elif slope < 0 and p < 0.05:
        trend_text = 'HABITUATION - Diminution significative'
    else:
        trend_text = 'Pas de tendance significative'
    
    ax.text(0.5, 0.95, trend_text, transform=ax.transAxes, ha='center', va='top',
            fontsize=12, color='#3fb950', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=BG_AX, alpha=0.8))
    
    ax.set_xlabel('Stimulation', color=LABEL, fontsize=12)
    ax.set_ylabel('SCL normalise (0=min, 1=max du participant)', color=LABEL, fontsize=12)
    ax.set_title(f'SCL normalise pour tous les participants (N={df_all["participant_id"].nunique()})\nChaque point = un participant (numero)', 
                 color=TEXT, fontsize=14, fontweight='bold')
    ax.tick_params(colors=LABEL)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.5, axis='y')
    ax.legend(loc='upper right', facecolor=BG_AX, edgecolor=GRID, labelcolor=TEXT)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'TOUS_PARTICIPANTS_NORMALISES.png'), 
                dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close()
    
    return slope, p

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("ANALYSE COMPLETE - TOUS LES PARTICIPANTS")
    print("Normalisation: correction d'etendue (0=min, 1=max)")
    print("="*80)
    
    print("\nChargement des donnees...")
    df_all = collect_all_data()
    
    print(f"\nParticipants: {df_all['participant_id'].nunique()}")
    print(f"Stimulations: {len(df_all)}")
    
    print("\nGeneration du graphique...")
    slope, p = plot_main(df_all, OUTPUT_PATH)
    
    print(f"\nTendance: pente={slope:.4f}, p={p:.4f}")
    
    if p < 0.05 and slope > 0:
        print("\nConclusion: SENSIBILISATION - Le SCL augmente significativement")
    elif p < 0.05 and slope < 0:
        print("\nConclusion: HABITUATION - Le SCL diminue significativement")
    else:
        print("\nConclusion: Pas de tendance significative")
    
    # ANOVA
    groups = [df_all[df_all['stimulus_order'] == i]['normalized_mean'].values for i in range(1, 15)]
    groups = [g for g in groups if len(g) > 0]
    f_stat, p_anova = f_oneway(*groups)
    print(f"\nANOVA: F={f_stat:.3f}, p={p_anova:.4f}")
    
    # Sauvegarder les stats
    stats = df_all.groupby('stimulus_order').agg({
        'raw_mean': ['mean', 'std', 'min', 'max'],
        'normalized_mean': ['mean', 'std']
    }).round(4)
    stats.to_csv(os.path.join(OUTPUT_PATH, 'statistiques_par_stimulation.csv'))
    
    print("\n" + "="*80)
    print(f"COMPLET - Output: {OUTPUT_PATH}")
    print("="*80)

if __name__ == "__main__":
    main()