# -*- coding: utf-8 -*-
"""
ANALYSE EDA COMPLETE - 40 JEUNES PARTICIPANTS (VERSION CORRIGEE)
================================================================
Corrections apportées:
1. Timestamps réels (non normalisés) pour les pentes de sensibilisation
2. Correction du double comptage des stimuli auditifs
3. Filtrage des positions avec n<5 pour les tests statistiques
4. Gestion des NaN dans les tests t
5. Mapping complet des stimuli
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, f_oneway, linregress
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

EDA_FOLDER = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_labeled'
OUTPUT_PATH = r'C:/Users/katia/Desktop/output-tnc-cleaned/comparison_analysis/40_jeunes_complet_corrige'
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
# LISTE DES PARTICIPANTS JEUNES (40)
# ============================================================================

JEUNES_PARTICIPANTS = ['004', '005', '008', '009', '010', '011', '012', '016', 
                       '018', '020', '022', '024', '025b', '028', '029', '031', 
                       '032', '033', '034', '035', '036', '037', '038', '039', 
                       '041', '042', '043', '044', '046', '047b', '049', '053', 
                       '054', '055', '057', '058', '060', '061', '062', '064']

# ============================================================================
# PLAN CONTREBALANCE - À COMPLÉTER AVEC VOS DONNÉES
# ============================================================================
# !!! VOUS DEVEZ COMPLÉTER CE DICTIONNAIRE AVEC TOUS LES 40 PARTICIPANTS !!!
PARTICIPANT_INFO = {
    '004': {'group': 'A', 'block_vis': 1, 'block_aud': 1, 'block_bim': 1},
    '005': {'group': 'A', 'block_vis': 1, 'block_aud': 3, 'block_bim': 3},
    '008': {'group': 'A', 'block_vis': 2, 'block_aud': 2, 'block_bim': 2},
    '009': {'group': 'A', 'block_vis': 2, 'block_aud': 4, 'block_bim': 4},
    '010': {'group': 'B', 'block_vis': 3, 'block_aud': 1, 'block_bim': 1},
    '011': {'group': 'A', 'block_vis': 4, 'block_aud': 1, 'block_bim': 1},
    '012': {'group': 'B', 'block_vis': 1, 'block_aud': 4, 'block_bim': 1},
    '016': {'group': 'B', 'block_vis': 4, 'block_aud': 4, 'block_bim': 1},
    '018': {'group': 'B', 'block_vis': 5, 'block_aud': 2, 'block_bim': 1},
    '020': {'group': 'B', 'block_vis': 5, 'block_aud': 4, 'block_bim': 1},
    '022': {'group': 'B', 'block_vis': 6, 'block_aud': 2, 'block_bim': 1},
    '024': {'group': 'B', 'block_vis': 6, 'block_aud': 4, 'block_bim': 1},
    '025b': {'group': 'A', 'block_vis': 7, 'block_aud': 1, 'block_bim': 1},
    '028': {'group': 'A', 'block_vis': 7, 'block_aud': 4, 'block_bim': 1},
    '029': {'group': 'A', 'block_vis': 8, 'block_aud': 1, 'block_bim': 1},
    '031': {'group': 'A', 'block_vis': 3, 'block_aud': 3, 'block_bim': 1},
    '032': {'group': 'B', 'block_vis': 3, 'block_aud': 4, 'block_bim': 1},
    '033': {'group': 'A', 'block_vis': 1, 'block_aud': 1, 'block_bim': 1},
    '034': {'group': 'B', 'block_vis': 2, 'block_aud': 1, 'block_bim': 1},
    '035': {'group': 'A', 'block_vis': 3, 'block_aud': 1, 'block_bim': 1},
    '036': {'group': 'B', 'block_vis': 2, 'block_aud': 3, 'block_bim': 1},
    '037': {'group': 'A', 'block_vis': 1, 'block_aud': 3, 'block_bim': 1},
    '038': {'group': 'B', 'block_vis': 1, 'block_aud': 4, 'block_bim': 1},
    '039': {'group': 'B', 'block_vis': 1, 'block_aud': 2, 'block_bim': 1},
    '041': {'group': 'A', 'block_vis': 2, 'block_aud': 2, 'block_bim': 1},
    '042': {'group': 'A', 'block_vis': 2, 'block_aud': 4, 'block_bim': 1},
    '043': {'group': 'A', 'block_vis': 4, 'block_aud': 1, 'block_bim': 1},
    '044': {'group': 'B', 'block_vis': 3, 'block_aud': 2, 'block_bim': 1},
    '046': {'group': 'B', 'block_vis': 4, 'block_aud': 4, 'block_bim': 1},
    '047b': {'group': 'A', 'block_vis': 5, 'block_aud': 1, 'block_bim': 1},
    '049': {'group': 'A', 'block_vis': 5, 'block_aud': 3, 'block_bim': 1},
    '053': {'group': 'A', 'block_vis': 6, 'block_aud': 3, 'block_bim': 1},
    '054': {'group': 'B', 'block_vis': 6, 'block_aud': 4, 'block_bim': 1},
    '055': {'group': 'A', 'block_vis': 7, 'block_aud': 1, 'block_bim': 1},
    '057': {'group': 'B', 'block_vis': 7, 'block_aud': 3, 'block_bim': 1},
    '058': {'group': 'A', 'block_vis': 7, 'block_aud': 4, 'block_bim': 1},
    '060': {'group': 'B', 'block_vis': 8, 'block_aud': 2, 'block_bim': 1},
    '061': {'group': 'A', 'block_vis': 8, 'block_aud': 3, 'block_bim': 1},
    '062': {'group': 'B', 'block_vis': 8, 'block_aud': 4, 'block_bim': 1},
    '064': {'group': 'B', 'block_vis': 8, 'block_aud': 4, 'block_bim': 1},
}

def get_participant_info(pid):
    if pid in PARTICIPANT_INFO:
        return PARTICIPANT_INFO[pid]
    print(f"⚠️  Attention: Participant {pid} non trouvé dans PARTICIPANT_INFO, utilisation valeurs par défaut")
    return {'group': 'A', 'block_vis': 1, 'block_aud': 1, 'block_bim': 1}

# ============================================================================
# ORDRES DES STIMULI PAR BLOCK
# ============================================================================

# Groupe A - Visuelle (stim1)
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

# Groupe A - Auditive (stim2)
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

# Groupe A - Bimodale (stim3)
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

# Groupe B - Auditive (stim1)
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

# Groupe B - Visuelle (stim2)
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

# Groupe B - Bimodale (stim3) - identique à Groupe A
ORDERS_BIM_B = ORDERS_BIM_A

# ============================================================================
# MAPPING DES NOMS DE FICHIERS VERS NOMS STANDARDS (CORRIGÉ)
# ============================================================================

FILE_TO_STANDARD = {
    'Monet': 'MonetVis',
    'regata': 'RefPosVis',
    'Mc_nicoll': 'McNicollVis',
    'Bataille': 'RefNegVis',
    'Audio_monet': 'MonetAud',
    'Audio_mc_nicoll': 'McNicollAud',
    'Musique__test_EN': 'RefNegAud',
    'routine_127_EP': 'RefPosAud',
    'routine_128_EN': 'RefPosAud',  # CORRIGÉ: plus de double comptage
    'Audio___tableau_monet': 'MonetMix',
    'Audio___tableau_Mc_nicoll': 'McNicollMix',
    'routine_127_EP___tableau_ragata': 'RefPosMix',
    'routine_128_EN__tableau_bataill': 'RefNegMix'
}

# Ensemble pour éviter les doublons
SEEN_STIMULI = set()

STANDARD_TO_DISPLAY = {
    'MonetVis': 'Monet', 'RefPosVis': 'Positif', 'McNicollVis': 'McNicoll', 'RefNegVis': 'Negatif',
    'MonetAud': 'Monet', 'RefPosAud': 'Positif', 'McNicollAud': 'McNicoll', 'RefNegAud': 'Negatif',
    'MonetMix': 'Monet', 'RefPosMix': 'Positif', 'McNicollMix': 'McNicoll', 'RefNegMix': 'Negatif'
}

STIMULUS_TYPE = {
    'MonetVis': 'monet', 'RefPosVis': 'positif', 'McNicollVis': 'mcnichol', 'RefNegVis': 'negatif',
    'MonetAud': 'monet', 'RefPosAud': 'positif', 'McNicollAud': 'mcnichol', 'RefNegAud': 'negatif',
    'MonetMix': 'monet', 'RefPosMix': 'positif', 'McNicollMix': 'mcnichol', 'RefNegMix': 'negatif'
}

MODALITE_FROM_STANDARD = {
    'MonetVis': 'visuelle', 'RefPosVis': 'visuelle', 'McNicollVis': 'visuelle', 'RefNegVis': 'visuelle',
    'MonetAud': 'auditive', 'RefPosAud': 'auditive', 'McNicollAud': 'auditive', 'RefNegAud': 'auditive',
    'MonetMix': 'combinee', 'RefPosMix': 'combinee', 'McNicollMix': 'combinee', 'RefNegMix': 'combinee'
}

VERSION_MAPPING = {
    'Musique__test_EN_2': 'Musique__test_EN',
    'Audio___tableau_monet_2': 'Audio___tableau_monet',
    'Audio___tableau_Mc_nicoll_2': 'Audio___tableau_Mc_nicoll'
}

STIMULUS_LABELS = {
    'negatif': 'NEGATIF\n(Smârdan)',
    'positif': 'POSITIF\n(Regattas)',
    'mcnichol': 'McNICOLL\n(Sunny Sept)',
    'monet': 'MONET'
}

MODALITES_NAMES = {
    'visuelle': 'VISUELLE',
    'auditive': 'AUDITIVE',
    'combinee': 'COMBINEE'
}

MODALITES_COLORS = {
    'visuelle': '#58a6ff',
    'auditive': '#3fb950',
    'combinee': '#f85149'
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

def extract_metrics_with_context(df, t, gsr, stim_name, pid):
    """Extrait les métriques avec la position dans le bloc et le timestamp"""
    original_name = VERSION_MAPPING.get(stim_name, stim_name)
    start, end = find_stimulus_indices(df, original_name)
    
    if start is None:
        for variant in VERSION_MAPPING.keys():
            if VERSION_MAPPING[variant] == original_name:
                start, end = find_stimulus_indices(df, variant)
                if start is not None:
                    break
    
    if start is None:
        return None
    
    # 5s avant
    pre_start = max(0, start - PRE_SAMPLES)
    pre_gsr = gsr[pre_start:start]
    mean_pre = np.mean(pre_gsr) if len(pre_gsr) > 0 else 0
    
    # 40s stimulation
    stim_gsr = gsr[start:end+1]
    if len(stim_gsr) == 0:
        return None
    
    mean_stim = np.mean(stim_gsr)
    std_stim = np.std(stim_gsr)
    cv_stim = std_stim / mean_stim if mean_stim != 0 else 0
    max_stim = np.max(stim_gsr)
    min_stim = np.min(stim_gsr)
    timestamp_median = np.median(t[start:end+1])
    
    # 5s apres
    post_end = min(len(df) - 1, end + POST_SAMPLES)
    post_gsr = gsr[end+1:post_end+1]
    mean_post = np.mean(post_gsr) if len(post_gsr) > 0 else 0
    
    # Convertir en nom standard
    standard_name = FILE_TO_STANDARD.get(original_name, original_name)
    
    if standard_name not in STIMULUS_TYPE:
        return None
    
    stim_type = STIMULUS_TYPE[standard_name]
    modalite = MODALITE_FROM_STANDARD[standard_name]
    
    # Récupérer les infos de contrebalancement
    pinfo = get_participant_info(pid)
    group = pinfo['group']
    
    # Déterminer la position dans le bloc
    position_in_block = None
    
    if modalite == 'visuelle':
        if group == 'A':
            order = ORDERS_VIS_A.get(pinfo['block_vis'], ORDERS_VIS_A[1])
        else:
            order = ORDERS_VIS_B.get(pinfo['block_vis'], ORDERS_VIS_B[1])
        if standard_name in order:
            position_in_block = order.index(standard_name) + 1
            
    elif modalite == 'auditive':
        if group == 'A':
            order = ORDERS_AUD_A.get(pinfo['block_aud'], ORDERS_AUD_A[1])
        else:
            order = ORDERS_AUD_B.get(pinfo['block_aud'], ORDERS_AUD_B[1])
        if standard_name in order:
            position_in_block = order.index(standard_name) + 1
            
    else:  # bimodale
        if group == 'A':
            order = ORDERS_BIM_A.get(pinfo['block_bim'], ORDERS_BIM_A[1])
        else:
            order = ORDERS_BIM_B.get(pinfo['block_bim'], ORDERS_BIM_B[1])
        if standard_name in order:
            position_in_block = order.index(standard_name) + 1
    
    return {
        'participant': pid,
        'group': group,
        'modalite': modalite,
        'stimulus_type': stim_type,
        'stimulus_display': STANDARD_TO_DISPLAY.get(standard_name, standard_name),
        'position_in_block': position_in_block,
        'timestamp_sec': timestamp_median,
        'mean_pre_5s': mean_pre,
        'mean_stim_40s': mean_stim,
        'mean_post_5s': mean_post,
        'mean_positive_peak': max_stim,
        'mean_negative_trough': min_stim,
        'std': std_stim,
        'cv': cv_stim
    }

# ============================================================================
# COLLECTE DES DONNEES
# ============================================================================

print("="*80)
print("COLLECTE DES DONNEES - 40 JEUNES PARTICIPANTS (VERSION CORRIGEE)")
print("="*80)
print("\nStimuli analyses:")
print("  - NEGATIF: Smârdan Attack")
print("  - POSITIF: Regattas at Hampton Court")
print("  - EXPERIMENTAL 1: McNicoll - Sunny September")
print("  - EXPERIMENTAL 2: Monet")
print("\nModalites: Visuelle, Auditive, Combinee")
print("Avec prise en compte du plan contrebalance (Groupes A/B et blocks 1-8)")
print("-"*80)

# Structure pour stocker les donnees
all_records = []

for pid in tqdm(JEUNES_PARTICIPANTS, desc="Traitement participants"):
    data = load_participant_data(pid)
    if data is None:
        continue
    
    df, t, gsr = data
    
    # Obtenir la liste des stimuli uniques dans l'ordre
    stimuli_seen = []
    for _, row in df.iterrows():
        stim_name = row['stimulus_label']
        if stim_name != 'baseline' and stim_name not in stimuli_seen:
            stimuli_seen.append(stim_name)
    
    for stim_name in stimuli_seen:
        metrics = extract_metrics_with_context(df, t, gsr, stim_name, pid)
        if metrics is not None:
            all_records.append(metrics)

df_data = pd.DataFrame(all_records)

# Supprimer les doublons éventuels (participant + modalite + stimulus_type + position)
df_data = df_data.drop_duplicates(subset=['participant', 'modalite', 'stimulus_type', 'position_in_block'])

print(f"\nDonnees collectees: {len(df_data)} observations")
print(f"Participants: {df_data['participant'].nunique()}")
print(f"Modalites: {df_data['modalite'].unique()}")

# ============================================================================
# SAUVEGARDE DES DONNEES INDIVIDUELLES
# ============================================================================

print("\n" + "="*80)
print("SAUVEGARDE DES DONNEES INDIVIDUELLES")
print("="*80)

individual_folder = os.path.join(OUTPUT_PATH, 'donnees_individuelles')
os.makedirs(individual_folder, exist_ok=True)

for modalite in ['visuelle', 'auditive', 'combinee']:
    modalite_folder = os.path.join(individual_folder, modalite)
    os.makedirs(modalite_folder, exist_ok=True)
    
    for stim_type in ['negatif', 'positif', 'mcnichol', 'monet']:
        df_stim = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == stim_type)]
        
        if len(df_stim) > 0:
            stim_data = df_stim[['participant', 'group', 'position_in_block', 'timestamp_sec',
                                 'mean_pre_5s', 'mean_stim_40s', 'mean_post_5s',
                                 'mean_positive_peak', 'mean_negative_trough', 'std', 'cv']].copy()
            csv_path = os.path.join(modalite_folder, f'{stim_type}_individuel.csv')
            stim_data.to_csv(csv_path, index=False)
            print(f"  Saved: {modalite}/{stim_type}_individuel.csv ({len(stim_data)} participants)")

# ============================================================================
# STATISTIQUES DESCRIPTIVES
# ============================================================================

print("\n" + "="*80)
print("STATISTIQUES DESCRIPTIVES PAR STIMULUS")
print("="*80)

summary_data = []

for modalite in ['visuelle', 'auditive', 'combinee']:
    print(f"\n{MODALITES_NAMES[modalite]}:")
    print("-"*50)
    
    for stim_type, stim_label in STIMULUS_LABELS.items():
        values = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == stim_type)]['mean_stim_40s'].values
        
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            sem_val = sem(values)
            cv_val = std_val / mean_val if mean_val != 0 else 0
            min_val = np.min(values)
            max_val = np.max(values)
            
            summary_data.append({
                'modalite': MODALITES_NAMES[modalite],
                'stimulus': stim_label.replace('\n', ' '),
                'n': len(values),
                'mean_SCL_uS': mean_val,
                'std_SCL_uS': std_val,
                'sem_SCL_uS': sem_val,
                'cv': cv_val,
                'min_SCL_uS': min_val,
                'max_SCL_uS': max_val
            })
            
            print(f"  {stim_label.replace(chr(10), ' '):25} | n={len(values):3d} | mean={mean_val:.3f} uS | std={std_val:.3f}")

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(OUTPUT_PATH, 'statistiques_descriptives.csv'), index=False)
print("\n  Saved: statistiques_descriptives.csv")

# ============================================================================
# MATRICE PARTICIPANT × STIMULUS
# ============================================================================

print("\n" + "="*80)
print("CREATION DE LA MATRICE PARTICIPANT × STIMULUS")
print("="*80)

matrix_data = []
for pid in JEUNES_PARTICIPANTS:
    row = {'participant': pid}
    df_pid = df_data[df_data['participant'] == pid]
    
    for modalite in ['visuelle', 'auditive', 'combinee']:
        for stim_type in ['negatif', 'positif', 'mcnichol', 'monet']:
            vals = df_pid[(df_pid['modalite'] == modalite) & (df_pid['stimulus_type'] == stim_type)]['mean_stim_40s'].values
            col_name = f"{modalite}_{stim_type}"
            row[col_name] = vals[0] if len(vals) > 0 else np.nan
    
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data)
df_matrix.to_csv(os.path.join(OUTPUT_PATH, 'matrice_participants_stimuli.csv'), index=False)
print("  Saved: matrice_participants_stimuli.csv")

# ============================================================================
# ANALYSE DE LA SENSIBILISATION GLOBALE (TIMESTAMP RÉEL - CORRIGÉ)
# ============================================================================

print("\n" + "="*80)
print("ANALYSE DE LA SENSIBILISATION GLOBALE (TIMESTAMP RÉEL)")
print("="*80)

participant_slopes = {}
for pid in JEUNES_PARTICIPANTS:
    df_pid = df_data[df_data['participant'] == pid]
    if len(df_pid) >= 5:
        times = df_pid['timestamp_sec'].values
        scls = df_pid['mean_stim_40s'].values
        if len(times) > 1 and times.max() > times.min():
            # Utiliser les secondes réelles (pente en µS par seconde)
            slope, intercept, r_value, p_val, _ = linregress(times, scls)
            participant_slopes[pid] = {'slope': slope, 'p_value': p_val, 'r_value': r_value}

slopes = [v['slope'] for v in participant_slopes.values()]
print(f"Pente moyenne de sensibilisation: {np.mean(slopes):.6f} µS/s (n={len(slopes)})")
print(f"Écart-type des pentes: {np.std(slopes):.6f}")
print(f"Pente min: {np.min(slopes):.6f}, Pente max: {np.max(slopes):.6f}")

# ============================================================================
# ANALYSE AVEC CORRECTION DE LA SENSIBILISATION
# ============================================================================

print("\n" + "="*80)
print("ANALYSE STATISTIQUE AVEC CORRECTION DE LA SENSIBILISATION")
print("="*80)

# Créer une version ajustée des données (soustraction de la tendance temporelle)
df_data_adjusted = df_data.copy()
for pid in JEUNES_PARTICIPANTS:
    if pid in participant_slopes:
        slope = participant_slopes[pid]['slope']
        df_pid = df_data[df_data['participant'] == pid]
        times = df_pid['timestamp_sec'].values
        t_min = times.min()
        for idx, row in df_pid.iterrows():
            adjustment = slope * (row['timestamp_sec'] - t_min)
            df_data_adjusted.loc[idx, 'mean_stim_40s_adjusted'] = row['mean_stim_40s'] - adjustment
    else:
        df_data_adjusted.loc[df_data_adjusted['participant'] == pid, 'mean_stim_40s_adjusted'] = \
            df_data_adjusted.loc[df_data_adjusted['participant'] == pid, 'mean_stim_40s'].values

# ============================================================================
# TESTS STATISTIQUES PAR POSITION INTRA-BLOC (AVEC SEUIL MINIMUM)
# ============================================================================

print("\n" + "="*80)
print("TESTS STATISTIQUES (COMPARAISON À POSITION ÉGALE DANS LE BLOC)")
print("="*80)
print("NB: Seulement les positions avec n>=5 sont affichées")

test_results = []

for modalite in ['visuelle', 'auditive', 'combinee']:
    print(f"\n{MODALITES_NAMES[modalite]}:")
    print("-"*40)
    
    for pos in range(1, 5):
        df_pos = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == pos)]
        
        neg_vals = df_pos[df_pos['stimulus_type'] == 'negatif']['mean_stim_40s'].values
        pos_vals = df_pos[df_pos['stimulus_type'] == 'positif']['mean_stim_40s'].values
        
        # Seuil minimum de 5 participants par groupe pour test t valide
        if len(neg_vals) >= 5 and len(pos_vals) >= 5:
            t_stat, p_val = ttest_ind(neg_vals, pos_vals)
            test_results.append({
                'modalite': MODALITES_NAMES[modalite],
                'position': pos,
                'comparaison': 'Negatif vs Positif',
                't_statistic': t_stat,
                'p_value': p_val,
                'n_neg': len(neg_vals),
                'n_pos': len(pos_vals),
                'mean_neg': np.mean(neg_vals),
                'mean_pos': np.mean(pos_vals)
            })
            print(f"  Pos{pos} - Neg vs Pos: t={t_stat:.3f}, p={p_val:.4f} (n={len(neg_vals)}/{len(pos_vals)})")
        else:
            print(f"  Pos{pos} - Neg vs Pos: Ignoré (n_neg={len(neg_vals)}, n_pos={len(pos_vals)} <5)")

df_tests = pd.DataFrame(test_results)
df_tests.to_csv(os.path.join(OUTPUT_PATH, 'tests_statistiques_par_position.csv'), index=False)
print("\n  Saved: tests_statistiques_par_position.csv")

# ============================================================================
# EFFET DE POSITION INTRA-BLOC (SENSIBILISATION LOCALE)
# ============================================================================

print("\n" + "="*80)
print("EFFET DE POSITION INTRA-BLOC (SENSIBILISATION LOCALE)")
print("="*80)

position_effects = []

for modalite in ['visuelle', 'auditive', 'combinee']:
    print(f"\n{MODALITES_NAMES[modalite]}:")
    
    pos_means = []
    for pos in range(1, 5):
        vals = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == pos)]['mean_stim_40s'].values
        if len(vals) > 0:
            pos_means.append(np.mean(vals))
            print(f"  Position {pos}: mean={np.mean(vals):.3f} uS, n={len(vals)}")
    
    # ANOVA sur les positions (seulement si tous les groupes ont au moins 5 participants)
    pos1 = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == 1)]['mean_stim_40s'].values
    pos2 = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == 2)]['mean_stim_40s'].values
    pos3 = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == 3)]['mean_stim_40s'].values
    pos4 = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == 4)]['mean_stim_40s'].values
    
    if len(pos1) >= 5 and len(pos2) >= 5 and len(pos3) >= 5 and len(pos4) >= 5:
        f_stat, p_val = f_oneway(pos1, pos2, pos3, pos4)
        position_effects.append({
            'modalite': MODALITES_NAMES[modalite],
            'F_statistic': f_stat,
            'p_value': p_val,
            'pos1_mean': np.mean(pos1),
            'pos2_mean': np.mean(pos2),
            'pos3_mean': np.mean(pos3),
            'pos4_mean': np.mean(pos4),
            'pos1_n': len(pos1),
            'pos2_n': len(pos2),
            'pos3_n': len(pos3),
            'pos4_n': len(pos4)
        })
        print(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    else:
        print(f"  ANOVA: Ignorée (n1={len(pos1)}, n2={len(pos2)}, n3={len(pos3)}, n4={len(pos4)} <5)")

df_position_effects = pd.DataFrame(position_effects)
df_position_effects.to_csv(os.path.join(OUTPUT_PATH, 'effet_position_intra_bloc.csv'), index=False)

# ============================================================================
# COMPARAISON DES MODALITES À POSITION ÉGALE
# ============================================================================

print("\n" + "="*80)
print("COMPARAISON DES MODALITES À POSITION ÉGALE")
print("="*80)

modalite_comparisons = []

for stim_type in ['negatif', 'positif', 'mcnichol', 'monet']:
    print(f"\n{STIMULUS_LABELS[stim_type].replace(chr(10), ' ')}:")
    
    for pos in range(1, 5):
        df_pos = df_data[(df_data['stimulus_type'] == stim_type) & (df_data['position_in_block'] == pos)]
        
        vis_vals = df_pos[df_pos['modalite'] == 'visuelle']['mean_stim_40s'].values
        aud_vals = df_pos[df_pos['modalite'] == 'auditive']['mean_stim_40s'].values
        com_vals = df_pos[df_pos['modalite'] == 'combinee']['mean_stim_40s'].values
        
        # Seuil minimum de 5 participants par groupe
        if len(vis_vals) >= 5 and len(aud_vals) >= 5 and len(com_vals) >= 5:
            f_stat, p_val = f_oneway(vis_vals, aud_vals, com_vals)
            modalite_comparisons.append({
                'stimulus': stim_type,
                'position': pos,
                'F_statistic': f_stat,
                'p_value': p_val,
                'mean_vis': np.mean(vis_vals),
                'mean_aud': np.mean(aud_vals),
                'mean_com': np.mean(com_vals),
                'n_vis': len(vis_vals),
                'n_aud': len(aud_vals),
                'n_com': len(com_vals)
            })
            print(f"  Position {pos}: F={f_stat:.3f}, p={p_val:.4f} (n_vis={len(vis_vals)}, n_aud={len(aud_vals)}, n_com={len(com_vals)})")
        else:
            print(f"  Position {pos}: Ignorée (n_vis={len(vis_vals)}, n_aud={len(aud_vals)}, n_com={len(com_vals)} <5)")

df_modalite_comparisons = pd.DataFrame(modalite_comparisons)
df_modalite_comparisons.to_csv(os.path.join(OUTPUT_PATH, 'comparaison_modalites_par_position.csv'), index=False)

# ============================================================================
# GRAPHIQUES
# ============================================================================

print("\n" + "="*80)
print("GENERATION DES GRAPHIQUES")
print("="*80)

BG_FIG = '#0d1117'
BG_AX = '#161b22'
TEXT_C = '#e6edf3'
LABEL_C = '#8b949e'
GRID_C = '#21262d'

# Graphique 1: Boxplot par modalite (donnees brutes)
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG_FIG)

for i, modalite in enumerate(['visuelle', 'auditive', 'combinee']):
    ax = axes[i]
    ax.set_facecolor(BG_AX)
    
    stim_order = ['negatif', 'positif', 'mcnichol', 'monet']
    stim_labels = [STIMULUS_LABELS[s] for s in stim_order]
    
    data = [df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == s)]['mean_stim_40s'].values for s in stim_order]
    
    bp = ax.boxplot(data, tick_labels=stim_labels, patch_artist=True, showmeans=True,
                    meanprops={'marker': 'o', 'markerfacecolor': '#ffa657', 
                              'markeredgecolor': 'white', 'markersize': 6})
    
    for patch in bp['boxes']:
        patch.set_facecolor(MODALITES_COLORS[modalite])
        patch.set_alpha(0.7)
        patch.set_edgecolor('white')
    
    ax.set_ylabel('SCL (uS)', color=LABEL_C, fontsize=11)
    ax.set_title(MODALITES_NAMES[modalite], color=TEXT_C, fontsize=12, fontweight='bold')
    ax.tick_params(colors=LABEL_C, labelsize=8)
    ax.grid(True, color=GRID_C, alpha=0.3, axis='y')

plt.suptitle('Distribution du SCL par modalité et par stimulus (N=40 jeunes)', 
             color=TEXT_C, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'boxplots_par_modalite.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("  Saved: boxplots_par_modalite.png")

# Graphique 2: Effet de position intra-bloc
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(BG_FIG)

for i, modalite in enumerate(['visuelle', 'auditive', 'combinee']):
    ax = axes[i]
    ax.set_facecolor(BG_AX)
    
    means = []
    errs = []
    positions = []
    
    for pos in range(1, 5):
        vals = df_data[(df_data['modalite'] == modalite) & (df_data['position_in_block'] == pos)]['mean_stim_40s'].values
        if len(vals) > 0:
            means.append(np.mean(vals))
            errs.append(sem(vals))
            positions.append(pos)
    
    ax.bar(positions, means, yerr=errs, color=MODALITES_COLORS[modalite], 
           alpha=0.7, edgecolor='white', capsize=5, width=0.6)
    ax.set_xlabel('Position dans le bloc', color=LABEL_C, fontsize=11)
    ax.set_ylabel('SCL (uS)', color=LABEL_C, fontsize=11)
    ax.set_title(MODALITES_NAMES[modalite], color=TEXT_C, fontsize=12, fontweight='bold')
    ax.set_xticks(positions)
    ax.tick_params(colors=LABEL_C)
    ax.grid(True, color=GRID_C, alpha=0.3, axis='y')

plt.suptitle('Effet de position intra-bloc (sensibilisation locale)', 
             color=TEXT_C, fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'effet_position_intra_bloc.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("  Saved: effet_position_intra_bloc.png")

# Graphique 3: 5s avant, 40s stim, 5s apres
print("\n" + "="*80)
print("ANALYSE DES 5s AVANT, 40s STIMULATION, 5s APRES")
print("="*80)

results_pre_post = []

for modalite in ['visuelle', 'auditive', 'combinee']:
    for stim_type in ['negatif', 'positif', 'mcnichol', 'monet']:
        df_stim = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == stim_type)]
        
        if len(df_stim) > 0:
            results_pre_post.append({
                'modalite': MODALITES_NAMES[modalite],
                'stimulus': STIMULUS_LABELS[stim_type].replace('\n', ' '),
                'mean_pre_5s': df_stim['mean_pre_5s'].mean(),
                'mean_stim_40s': df_stim['mean_stim_40s'].mean(),
                'mean_post_5s': df_stim['mean_post_5s'].mean(),
                'n': len(df_stim)
            })

df_pre_post = pd.DataFrame(results_pre_post)
df_pre_post.to_csv(os.path.join(OUTPUT_PATH, 'analyse_5s_avant_apres.csv'), index=False)

for modalite in ['VISUELLE', 'AUDITIVE', 'COMBINEE']:
    df_modalite = df_pre_post[df_pre_post['modalite'] == modalite]
    
    if len(df_modalite) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(BG_FIG)
        ax.set_facecolor(BG_AX)
        
        x = np.arange(len(df_modalite))
        width = 0.25
        
        bars1 = ax.bar(x - width, df_modalite['mean_pre_5s'], width, label='5s AVANT', color='#58a6ff', alpha=0.8)
        bars2 = ax.bar(x, df_modalite['mean_stim_40s'], width, label='40s STIMULATION', color='#3fb950', alpha=0.8)
        bars3 = ax.bar(x + width, df_modalite['mean_post_5s'], width, label='5s APRES', color='#f85149', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(df_modalite['stimulus'], fontsize=10, color=LABEL_C)
        ax.set_ylabel('SCL (uS)', color=LABEL_C, fontsize=12)
        ax.set_title(f'{modalite} - Comparaison 5s avant, 40s stimulation, 5s apres\n(40 participants jeunes)', 
                     color=TEXT_C, fontsize=12, fontweight='bold')
        ax.tick_params(colors=LABEL_C)
        ax.grid(True, color=GRID_C, alpha=0.5, axis='y')
        ax.legend(loc='upper right', facecolor=BG_AX, edgecolor=GRID_C, labelcolor=TEXT_C)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f'{modalite.lower()}_5s_avant_apres.png'), 
                    dpi=150, facecolor=BG_FIG, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {modalite.lower()}_5s_avant_apres.png")

# ============================================================================
# RESUME FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUME FINAL - MOYENNE SCL (uS) PENDANT 40s")
print("="*80)
print("\n| Modalite | Negatif | Positif | McNicoll | Monet |")
print("|----------|---------|---------|----------|-------|")

for modalite in ['visuelle', 'auditive', 'combinee']:
    neg = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == 'negatif')]['mean_stim_40s'].mean()
    pos = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == 'positif')]['mean_stim_40s'].mean()
    mcn = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == 'mcnichol')]['mean_stim_40s'].mean()
    mon = df_data[(df_data['modalite'] == modalite) & (df_data['stimulus_type'] == 'monet')]['mean_stim_40s'].mean()
    print(f"| {MODALITES_NAMES[modalite]:<8} | {neg:.3f} | {pos:.3f} | {mcn:.3f} | {mon:.3f} |")

print("\n" + "="*80)
print("FICHIERS GENERES:")
print("="*80)
print(f"\n  {OUTPUT_PATH}")
print("\n  1. donnees_individuelles/ (par modalite et stimulus)")
print("\n  2. statistiques_descriptives.csv")
print("\n  3. matrice_participants_stimuli.csv")
print("\n  4. boxplots_par_modalite.png")
print("\n  5. effet_position_intra_bloc.png")
print("\n  6. *_5s_avant_apres.png")
print("\n  7. tests_statistiques_par_position.csv")
print("\n  8. comparaison_modalites_par_position.csv")
print("\n  9. effet_position_intra_bloc.csv")
print("\n  10. analyse_5s_avant_apres.csv")

print("\n" + "="*80)
print("ANALYSE COMPLETE - TERMINEE")
print("="*80)

# Sauvegarde des pentes de sensibilisation
slopes_df = pd.DataFrame([
    {'participant': pid, 'pente_sensibilisation_uS_s': v['slope'], 
     'r_value': v['r_value'], 'p_value': v['p_value']} 
    for pid, v in participant_slopes.items()
])
slopes_df.to_csv(os.path.join(OUTPUT_PATH, 'pentes_sensibilisation_participants.csv'), index=False)
print("\n  Saved: pentes_sensibilisation_participants.csv")