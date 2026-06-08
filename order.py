# -*- coding: utf-8 -*-
"""
SCRIPT REVISE - Analyse de l'effet d'ordre avec plan contrebalancé
==================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress, ttest_ind, f_oneway, ttest_1samp
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

EDA_FOLDER = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_labeled'
OUTPUT_PATH = r'C:/Users/katia/Desktop/output-tnc-cleaned/comparison_analysis/position_vs_scl_revised'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================================
# PARAMETERS
# ============================================================================

SAMPLING_RATE = 128
MIN_SAMPLES_PER_TEST = 5

# ============================================================================
# LISTE DES PARTICIPANTS
# ============================================================================

JEUNES_PARTICIPANTS = ['004', '005', '008', '009', '010', '011', '012', '016', 
                       '018', '020', '022', '024', '025b', '028', '029', '031', 
                       '032', '033', '034', '035', '036', '037', '038', '039', 
                       '041', '042', '043', '044', '046', '047b', '049', '053', 
                       '054', '055', '057', '058', '060', '061', '062', '064']

# ============================================================================
# PARTICIPANT INFO
# ============================================================================

PARTICIPANT_INFO = {
    '004': {'group': 'A', 'block1': 1, 'block2': 1},
    '005': {'group': 'A', 'block1': 1, 'block2': 3},
    '008': {'group': 'A', 'block1': 2, 'block2': 2},
    '009': {'group': 'A', 'block1': 2, 'block2': 4},
    '010': {'group': 'B', 'block1': 3, 'block2': 2},
    '011': {'group': 'A', 'block1': 4, 'block2': 1},
    '012': {'group': 'B', 'block1': 1, 'block2': 4},
    '016': {'group': 'B', 'block1': 4, 'block2': 4},
    '018': {'group': 'B', 'block1': 5, 'block2': 2},
    '020': {'group': 'B', 'block1': 5, 'block2': 4},
    '022': {'group': 'B', 'block1': 6, 'block2': 2},
    '024': {'group': 'B', 'block1': 6, 'block2': 4},
    '025b': {'group': 'A', 'block1': 7, 'block2': 1},
    '028': {'group': 'A', 'block1': 7, 'block2': 4},
    '029': {'group': 'A', 'block1': 8, 'block2': 1},
    '031': {'group': 'A', 'block1': 3, 'block2': 3},
    '032': {'group': 'B', 'block1': 3, 'block2': 4},
    '033': {'group': 'A', 'block1': 1, 'block2': 1},
    '034': {'group': 'B', 'block1': 2, 'block2': 1},
    '035': {'group': 'A', 'block1': 3, 'block2': 1},
    '036': {'group': 'B', 'block1': 2, 'block2': 3},
    '037': {'group': 'A', 'block1': 1, 'block2': 3},
    '038': {'group': 'B', 'block1': 1, 'block2': 4},
    '039': {'group': 'B', 'block1': 1, 'block2': 2},
    '041': {'group': 'A', 'block1': 2, 'block2': 2},
    '042': {'group': 'A', 'block1': 2, 'block2': 4},
    '043': {'group': 'A', 'block1': 4, 'block2': 1},
    '044': {'group': 'B', 'block1': 3, 'block2': 2},
    '046': {'group': 'B', 'block1': 4, 'block2': 4},
    '047b': {'group': 'A', 'block1': 5, 'block2': 1},
    '049': {'group': 'A', 'block1': 5, 'block2': 3},
    '053': {'group': 'A', 'block1': 6, 'block2': 3},
    '054': {'group': 'B', 'block1': 6, 'block2': 4},
    '055': {'group': 'A', 'block1': 7, 'block2': 1},
    '057': {'group': 'B', 'block1': 7, 'block2': 3},
    '058': {'group': 'A', 'block1': 7, 'block2': 4},
    '060': {'group': 'B', 'block1': 8, 'block2': 2},
    '061': {'group': 'A', 'block1': 8, 'block2': 3},
    '062': {'group': 'B', 'block1': 8, 'block2': 4},
    '064': {'group': 'B', 'block1': 8, 'block2': 4},
}

def get_participant_info(pid):
    if pid in PARTICIPANT_INFO:
        return PARTICIPANT_INFO[pid]
    return {'group': 'A', 'block1': 1, 'block2': 1}

# ============================================================================
# ORDRES PAR BLOCK
# ============================================================================

ORDERS_VIS_A = {
    1: ['MonetVis', 'RefPosVis', 'McNicollVis', 'RefNegVis'],
    2: ['RefNegVis', 'MonetVis', 'RefPosVis', 'McNicollVis'],
    3: ['RefNegVis', 'McNicollVis', 'MonetVis', 'RefPosVis'],
    4: ['RefPosVis', 'MonetVis', 'RefNegVis', 'McNicollVis'],
    5: ['McNicollVis', 'MonetVis', 'RefNegVis', 'RefPosVis'],
    6: ['MonetVis', 'McNicollVis', 'RefPosVis', 'RefNegVis'],
    7: ['RefPosVis', 'McNicollVis', 'MonetVis', 'RefNegVis'],
    8: ['McNicollVis', 'RefNegVis', 'RefPosVis', 'MonetVis']
}

ORDERS_AUD_A = {
    1: ['McNicollAud', 'RefNegAud', 'RefPosAud', 'MonetAud'],
    2: ['McNicollAud', 'RefNegAud', 'MonetAud', 'RefPosAud'],
    3: ['RefPosAud', 'RefNegAud', 'McNicollAud', 'MonetAud'],
    4: ['RefPosAud', 'McNicollAud', 'MonetAud', 'RefNegAud'],
    5: ['McNicollAud', 'RefNegAud', 'MonetAud', 'RefPosAud'],
    6: ['RefNegAud', 'McNicollAud', 'RefPosAud', 'MonetAud'],
    7: ['McNicollAud', 'RefNegAud', 'MonetAud', 'RefPosAud'],
    8: ['MonetAud', 'RefPosAud', 'RefNegAud', 'McNicollAud']
}

ORDERS_BIM_A = {
    1: ['RefPosMix', 'RefNegMix', 'McNicollMix', 'MonetMix'],
    2: ['RefPosMix', 'RefNegMix', 'McNicollMix', 'MonetMix'],
    3: ['RefNegMix', 'RefPosMix', 'MonetMix', 'McNicollMix'],
    4: ['MonetMix', 'McNicollMix', 'RefPosMix', 'RefNegMix'],
    5: ['RefPosMix', 'McNicollMix', 'MonetMix', 'RefNegMix'],
    6: ['McNicollMix', 'MonetMix', 'RefPosMix', 'RefNegMix'],
    7: ['McNicollMix', 'RefPosMix', 'RefNegMix', 'MonetMix'],
    8: ['MonetMix', 'McNicollMix', 'RefNegMix', 'RefPosMix']
}

ORDERS_AUD_B = {
    1: ['MonetAud', 'RefPosAud', 'McNicollAud', 'RefNegAud'],
    2: ['RefNegAud', 'MonetAud', 'RefPosAud', 'McNicollAud'],
    3: ['RefNegAud', 'McNicollAud', 'MonetAud', 'RefPosAud'],
    4: ['RefPosAud', 'MonetAud', 'RefNegAud', 'McNicollAud'],
    5: ['McNicollAud', 'MonetAud', 'RefNegAud', 'RefPosAud'],
    6: ['MonetAud', 'McNicollAud', 'RefPosAud', 'RefNegAud'],
    7: ['RefPosAud', 'McNicollAud', 'MonetAud', 'RefNegAud'],
    8: ['McNicollAud', 'RefNegAud', 'RefPosAud', 'MonetAud']
}

ORDERS_VIS_B = {
    1: ['McNicollVis', 'RefNegVis', 'RefPosVis', 'MonetVis'],
    2: ['McNicollVis', 'RefNegVis', 'MonetVis', 'RefPosVis'],
    3: ['RefPosVis', 'RefNegVis', 'McNicollVis', 'MonetVis'],
    4: ['RefPosVis', 'McNicollVis', 'MonetVis', 'RefNegVis'],
    5: ['McNicollVis', 'RefNegVis', 'MonetVis', 'RefPosVis'],
    6: ['RefNegVis', 'McNicollVis', 'RefPosVis', 'MonetVis'],
    7: ['McNicollVis', 'RefNegVis', 'MonetVis', 'RefPosVis'],
    8: ['MonetVis', 'RefPosVis', 'RefNegVis', 'McNicollVis']
}

ORDERS_BIM_B = ORDERS_BIM_A

# ============================================================================
# MAPPING DES STIMULI
# ============================================================================

STIMULI_INFO = {
    'MonetVis': {'modalite': 'visuelle', 'type': 'monet'},
    'RefPosVis': {'modalite': 'visuelle', 'type': 'positif'},
    'McNicollVis': {'modalite': 'visuelle', 'type': 'mcnichol'},
    'RefNegVis': {'modalite': 'visuelle', 'type': 'negatif'},
    'MonetAud': {'modalite': 'auditive', 'type': 'monet'},
    'RefPosAud': {'modalite': 'auditive', 'type': 'positif'},
    'McNicollAud': {'modalite': 'auditive', 'type': 'mcnichol'},
    'RefNegAud': {'modalite': 'auditive', 'type': 'negatif'},
    'MonetMix': {'modalite': 'bimodale', 'type': 'monet'},
    'RefPosMix': {'modalite': 'bimodale', 'type': 'positif'},
    'McNicollMix': {'modalite': 'bimodale', 'type': 'mcnichol'},
    'RefNegMix': {'modalite': 'bimodale', 'type': 'negatif'}
}

FILE_STIMULI_MAPPING = {
    'Monet': 'MonetVis',
    'regata': 'RefPosVis',
    'Mc_nicoll': 'McNicollVis',
    'Bataille': 'RefNegVis',
    'Audio_monet': 'MonetAud',
    'Audio_mc_nicoll': 'McNicollAud',
    'Musique__test_EN': 'RefNegAud',
    'routine_127_EP': 'RefPosAud',
    'routine_128_EN': 'RefPosAud',
    'Audio___tableau_monet': 'MonetMix',
    'Audio___tableau_Mc_nicoll': 'McNicollMix',
    'routine_127_EP___tableau_ragata': 'RefPosMix',
    'routine_128_EN__tableau_bataill': 'RefNegMix'
}

VERSION_MAPPING = {
    'Musique__test_EN_2': 'Musique__test_EN',
    'Audio___tableau_monet_2': 'Audio___tableau_monet',
    'Audio___tableau_Mc_nicoll_2': 'Audio___tableau_Mc_nicoll'
}

# ============================================================================
# FONCTIONS
# ============================================================================

def time_to_seconds(ts):
    try:
        if isinstance(ts, (int, float)):
            return float(ts)
        parts = str(ts).split(':')
        if len(parts) == 3:
            return int(parts[0])*3600 + int(parts[1])*60 + float(parts[2])
        return float(ts)
    except:
        return 0.0

def load_participant_data(pid):
    file_path = os.path.join(EDA_FOLDER, f"{pid}_labeled.csv")
    if not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    t = df['Time'].apply(time_to_seconds).values
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

def find_stimulus_indices(df, stim_name):
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

def extract_stimulus_mean_and_time(df, t, gsr, stim_name):
    original_name = VERSION_MAPPING.get(stim_name, stim_name)
    start, end = find_stimulus_indices(df, original_name)
    
    if start is None:
        for variant in VERSION_MAPPING.keys():
            if VERSION_MAPPING[variant] == original_name:
                start, end = find_stimulus_indices(df, variant)
                if start is not None:
                    break
    
    if start is None:
        return None, None
    
    stim_gsr = gsr[start:end+1]
    if len(stim_gsr) == 0:
        return None, None
    
    mean_val = np.mean(stim_gsr)
    median_time = np.median(t[start:end+1])
    
    return mean_val, median_time

def get_position_in_block(group, block_num, stim_name, modalite):
    if modalite == 'visuelle':
        if group == 'A':
            order = ORDERS_VIS_A.get(block_num, ORDERS_VIS_A[1])
        else:
            order = ORDERS_VIS_B.get(block_num, ORDERS_VIS_B[1])
    elif modalite == 'auditive':
        if group == 'A':
            order = ORDERS_AUD_A.get(block_num, ORDERS_AUD_A[1])
        else:
            order = ORDERS_AUD_B.get(block_num, ORDERS_AUD_B[1])
    else:
        if group == 'A':
            order = ORDERS_BIM_A.get(block_num, ORDERS_BIM_A[1])
        else:
            order = ORDERS_BIM_B.get(block_num, ORDERS_BIM_B[1])
    
    if stim_name in order:
        return order.index(stim_name) + 1
    return None

# ============================================================================
# COLLECTE DES DONNEES
# ============================================================================

print("="*80)
print("PARTIE 1: COLLECTE DES DONNEES")
print("="*80)

data_rows = []

for pid in tqdm(JEUNES_PARTICIPANTS, desc="Traitement participants"):
    data = load_participant_data(pid)
    if data is None:
        continue
    
    df, t, gsr = data
    pinfo = get_participant_info(pid)
    group = pinfo['group']
    block1 = pinfo['block1']
    block2 = pinfo['block2']
    block3 = pinfo.get('block3', block1)
    
    stimuli_order_real = []
    seen = set()
    for _, row in df.iterrows():
        s = row['stimulus_label']
        if s != 'baseline' and s not in seen:
            stimuli_order_real.append(s)
            seen.add(s)
    
    for global_pos, stim_name in enumerate(stimuli_order_real, 1):
        base_name = VERSION_MAPPING.get(stim_name, stim_name)
        
        if base_name in FILE_STIMULI_MAPPING:
            standard_name = FILE_STIMULI_MAPPING[base_name]
        else:
            standard_name = base_name
        
        if standard_name not in STIMULI_INFO:
            continue
        
        mean_val, median_time = extract_stimulus_mean_and_time(df, t, gsr, stim_name)
        if mean_val is None:
            continue
        
        modalite = STIMULI_INFO[standard_name]['modalite']
        
        if modalite == 'visuelle':
            block_num = block1
        elif modalite == 'auditive':
            block_num = block2
        else:
            block_num = block3
        
        position_in_block = get_position_in_block(group, block_num, standard_name, modalite)
        
        if position_in_block is None:
            continue
        
        data_rows.append({
            'participant': pid,
            'group': group,
            'stimulus_code': standard_name,
            'stimulus_type': STIMULI_INFO[standard_name]['type'],
            'modalite': modalite,
            'position_in_block': position_in_block,
            'timestamp_sec': median_time,
            'SCL_uS': mean_val
        })

df = pd.DataFrame(data_rows)
df = df.drop_duplicates(subset=['participant', 'stimulus_code'])

print(f"\nDonnees collectees: {len(df)} observations")
print(f"Participants: {df['participant'].nunique()}")
print(f"Modalites: {df['modalite'].unique()}")

# ============================================================================
# ANALYSE
# ============================================================================

print("\n" + "="*80)
print("PARTIE 2: SENSIBILISATION GLOBALE")
print("="*80)

pentes = []
for pid in JEUNES_PARTICIPANTS:
    df_pid = df[df['participant'] == pid]
    if len(df_pid) >= 5:
        times = df_pid['timestamp_sec'].values
        scls = df_pid['SCL_uS'].values
        if times.max() > times.min():
            slope, _, _, _, _ = linregress(times, scls)
            pentes.append(slope)

print(f"Pente moyenne: {np.mean(pentes):.6f} µS/s (n={len(pentes)})")
t_stat, p_val_global = ttest_1samp(pentes, 0)
print(f"Test t: t={t_stat:.3f}, p={p_val_global:.4f}")

print("\n" + "="*80)
print("PARTIE 3: EFFET DE POSITION INTRA-BLOC")
print("="*80)

results_intra_bloc = []
for modalite in ['visuelle', 'auditive', 'bimodale']:
    print(f"\n{modalite.upper()}:")
    pos_vals = {}
    for pos in range(1, 5):
        vals = df[(df['modalite'] == modalite) & (df['position_in_block'] == pos)]['SCL_uS'].values
        pos_vals[pos] = vals
        print(f"  Position {pos}: n={len(vals)}, mean={np.mean(vals):.3f} uS")
    
    if all(len(pos_vals[p]) >= MIN_SAMPLES_PER_TEST for p in range(1, 5)):
        f_stat, p_val = f_oneway(pos_vals[1], pos_vals[2], pos_vals[3], pos_vals[4])
        print(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
        results_intra_bloc.append({'modalite': modalite, 'F_stat': f_stat, 'p_value': p_val})

print("\n" + "="*80)
print("SAUVEGARDE")
print("="*80)

df.to_csv(os.path.join(OUTPUT_PATH, 'donnees_contrebalancees.csv'), index=False)
pd.DataFrame([{'pente_moyenne': np.mean(pentes), 'p_value': p_val_global}]).to_csv(
    os.path.join(OUTPUT_PATH, 'resume_sensibilisation_globale.csv'), index=False)
pd.DataFrame(results_intra_bloc).to_csv(os.path.join(OUTPUT_PATH, 'effet_position_intra_bloc.csv'), index=False)

print(f"\nOutput: {OUTPUT_PATH}")
print("="*80)
print("\nRESUME DES RESULTATS:")
print("-" * 40)
print(f"Sensibilisation globale: pente moyenne = {np.mean(pentes):.6f} µS/s, p = {p_val_global:.4f}")
print("\nEffet de position intra-bloc (ANOVA):")
for r in results_intra_bloc:
    print(f"  {r['modalite']}: F = {r['F_stat']:.3f}, p = {r['p_value']:.4f}")
print("="*80)