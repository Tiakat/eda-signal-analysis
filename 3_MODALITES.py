# -*- coding: utf-8 -*-
"""
ANALYSE DES 3 MODALITES - VERSION CORRIGEE AVEC CONTREBALANCEMENT
==================================================================
- Fusion des versions _2 avec les versions principales
- PRISE EN COMPTE DU PLAN CONTREBALANCE (Groupes A/B et blocks)
- Graphiques clairs avec tous les participants (40)
- Comparaisons statistiques avec correction pour l'effet de position
- CORRECTION: Timestamps réels pour les pentes de sensibilisation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, sem, linregress
from tqdm import tqdm

# ============================================================================
# PATHS
# ============================================================================

EDA_FOLDER = r'C:/Users/katia/Desktop/output-tnc-cleaned/jeune_lowfilter_labeled'
OUTPUT_PATH = r'C:/Users/katia/Desktop/output-tnc-cleaned/comparison_analysis/modalites_analysis_corrige'
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================================
# PARAMETERS
# ============================================================================

SAMPLING_RATE = 128
PRE_SEC = 5
POST_SEC = 5

# ============================================================================
# LISTE DES PARTICIPANTS JEUNES (40)
# ============================================================================

JEUNES_PARTICIPANTS = ['004', '005', '008', '009', '010', '011', '012', '016', 
                       '018', '020', '022', '024', '025b', '028', '029', '031', 
                       '032', '033', '034', '035', '036', '037', '038', '039', 
                       '041', '042', '043', '044', '046', '047b', '049', '053', 
                       '054', '055', '057', '058', '060', '061', '062', '064']

# ============================================================================
# PLAN CONTREBALANCE - COMPLET (basé sur votre premier script qui a bien fonctionné)
# ============================================================================

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
    print(f"⚠️  Attention: Participant {pid} non trouvé, utilisation défaut")
    return {'group': 'A', 'block_vis': 1, 'block_aud': 1, 'block_bim': 1}

# ============================================================================
# ORDRES DES STIMULI PAR BLOCK (inchangé)
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
# MAPPING DES NOMS DE FICHIERS
# ============================================================================

FILE_TO_STANDARD = {
    'Fleurs': 'FleursVis',
    'Monet': 'MonetVis',
    'regata': 'RefPosVis',
    'Mc_nicoll': 'McNicollVis',
    'Bataille': 'RefNegVis',
    'Audio_monet': 'MonetAud',
    'Audio_mc_nicoll': 'McNicollAud',
    'Musique__test_EN': 'RefNegAud',
    'routine_127_EP': 'RefPosAud',
    'routine_128_EN': 'RefPosAud',  # Corrigé pour éviter double comptage
    'Audio___tableau_monet': 'MonetMix',
    'Audio___tableau_Mc_nicoll': 'McNicollMix',
    'routine_127_EP___tableau_ragata': 'RefPosMix',
    'routine_128_EN__tableau_bataill': 'RefNegMix'
}

STANDARD_TO_MODALITE = {
    'FleursVis': 'image',
    'MonetVis': 'image', 'RefPosVis': 'image', 'McNicollVis': 'image', 'RefNegVis': 'image',
    'MonetAud': 'music', 'RefPosAud': 'music', 'McNicollAud': 'music', 'RefNegAud': 'music',
    'MonetMix': 'combined', 'RefPosMix': 'combined', 'McNicollMix': 'combined', 'RefNegMix': 'combined'
}

VERSION_MAPPING = {
    'Musique__test_EN_2': 'Musique__test_EN',
    'Audio___tableau_monet_2': 'Audio___tableau_monet',
    'Audio___tableau_Mc_nicoll_2': 'Audio___tableau_Mc_nicoll'
}

MODALITES_NAMES = {
    'image': 'IMAGE',
    'music': 'MUSIQUE',
    'combined': 'COMBINE'
}

MODALITES_COLORS = {
    'image': '#58a6ff',
    'music': '#3fb950',
    'combined': '#f85149'
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

def extract_stimulus_mean_with_context(df, t, gsr, stim_name, pid):
    original_name = VERSION_MAPPING.get(stim_name, stim_name)
    start, end = find_stimulus_indices(df, original_name)
    
    if start is None:
        for variant in VERSION_MAPPING.keys():
            if VERSION_MAPPING[variant] == original_name:
                start, end = find_stimulus_indices(df, variant)
                if start is not None:
                    break
    
    if start is None:
        return None, None, None
    
    stim_gsr = gsr[start:end+1]
    if len(stim_gsr) == 0:
        return None, None, None
    
    mean_val = np.mean(stim_gsr)
    timestamp_median = np.median(t[start:end+1])
    
    standard_name = FILE_TO_STANDARD.get(original_name, original_name)
    if standard_name not in STANDARD_TO_MODALITE:
        return mean_val, timestamp_median, None
    
    modalite = STANDARD_TO_MODALITE[standard_name]
    
    position_in_block = None
    if standard_name != 'FleursVis':
        pinfo = get_participant_info(pid)
        group = pinfo['group']
        
        if modalite == 'image':
            if group == 'A':
                order = ORDERS_VIS_A.get(pinfo['block_vis'], ORDERS_VIS_A[1])
            else:
                order = ORDERS_VIS_B.get(pinfo['block_vis'], ORDERS_VIS_B[1])
            if standard_name in order:
                position_in_block = order.index(standard_name) + 1
        elif modalite == 'music':
            if group == 'A':
                order = ORDERS_AUD_A.get(pinfo['block_aud'], ORDERS_AUD_A[1])
            else:
                order = ORDERS_AUD_B.get(pinfo['block_aud'], ORDERS_AUD_B[1])
            if standard_name in order:
                position_in_block = order.index(standard_name) + 1
        else:
            if group == 'A':
                order = ORDERS_BIM_A.get(pinfo['block_bim'], ORDERS_BIM_A[1])
            else:
                order = ORDERS_BIM_B.get(pinfo['block_bim'], ORDERS_BIM_B[1])
            if standard_name in order:
                position_in_block = order.index(standard_name) + 1
    
    return mean_val, timestamp_median, {'modalite': modalite, 'position': position_in_block, 'standard_name': standard_name}

# ============================================================================
# COLLECTE DES DONNEES
# ============================================================================

print("="*80)
print("ANALYSE DES 3 MODALITES - VERSION CORRIGEE")
print("="*80)

print(f"Participants: {len(JEUNES_PARTICIPANTS)}")

all_data_with_context = []

for pid in tqdm(JEUNES_PARTICIPANTS, desc="Traitement participants"):
    data = load_participant_data(pid)
    if data is None:
        continue
    
    df, t, gsr = data
    
    stimuli_seen = []
    for _, row in df.iterrows():
        stim_name = row['stimulus_label']
        if stim_name != 'baseline' and stim_name not in stimuli_seen:
            stimuli_seen.append(stim_name)
            
            mean_val, timestamp, context = extract_stimulus_mean_with_context(df, t, gsr, stim_name, pid)
            if mean_val is not None:
                record = {
                    'participant': pid,
                    'stimulus_raw': stim_name,
                    'SCL_uS': mean_val,
                    'timestamp_sec': timestamp
                }
                if context is not None:
                    record['modalite'] = context['modalite']
                    record['position_in_block'] = context['position']
                    record['standard_name'] = context['standard_name']
                else:
                    record['modalite'] = None
                    record['position_in_block'] = None
                    record['standard_name'] = None
                
                all_data_with_context.append(record)

df_data = pd.DataFrame(all_data_with_context)
df_data = df_data.drop_duplicates(subset=['participant', 'modalite', 'standard_name'])

print(f"\nDonnees collectees: {len(df_data)} observations")
print(f"Participants: {df_data['participant'].nunique()}")

df_data_filtered = df_data[df_data['modalite'].notna()].copy()
print(f"Apres filtrage (sans Fleurs): {len(df_data_filtered)} observations")

# ============================================================================
# ANALYSE DE LA SENSIBILISATION GLOBALE (TIMESTAMP RÉEL - CORRIGÉ)
# ============================================================================

print("\n" + "="*80)
print("ANALYSE DE LA SENSIBILISATION GLOBALE (TIMESTAMP RÉEL)")
print("="*80)

participant_slopes = {}
for pid in JEUNES_PARTICIPANTS:
    df_pid = df_data_filtered[df_data_filtered['participant'] == pid]
    if len(df_pid) >= 5:
        times = df_pid['timestamp_sec'].values
        scls = df_pid['SCL_uS'].values
        if len(times) > 1 and times.max() > times.min():
            # TIMESTAMP RÉEL - pas de normalisation !
            slope, _, _, _, _ = linregress(times, scls)
            participant_slopes[pid] = slope

slopes = list(participant_slopes.values())
print(f"Pente moyenne de sensibilisation: {np.mean(slopes):.6f} µS/s (n={len(slopes)})")
print(f"Écart-type des pentes: {np.std(slopes):.6f}")

# Créer une version ajustée des données (soustraction de la tendance temporelle)
df_data_adjusted = df_data_filtered.copy()
for pid in JEUNES_PARTICIPANTS:
    if pid in participant_slopes:
        slope = participant_slopes[pid]
        df_pid = df_data_filtered[df_data_filtered['participant'] == pid]
        times = df_pid['timestamp_sec'].values
        t_min = times.min() if len(times) > 0 else 0
        for idx, row in df_pid.iterrows():
            adjustment = slope * (row['timestamp_sec'] - t_min)
            df_data_adjusted.loc[idx, 'SCL_uS_adjusted'] = row['SCL_uS'] - adjustment
    else:
        df_data_adjusted.loc[df_data_adjusted['participant'] == pid, 'SCL_uS_adjusted'] = \
            df_data_adjusted.loc[df_data_adjusted['participant'] == pid, 'SCL_uS'].values

# ============================================================================
# COLLECTE DES VALEURS PAR MODALITE
# ============================================================================

print("\n" + "="*80)
print("COLLECTE DES VALEURS PAR MODALITE (APRES CORRECTION)")
print("="*80)

all_values_adjusted = {}
stats_by_modalite = {}

for modalite_key in ['image', 'music', 'combined']:
    df_m = df_data_adjusted[df_data_adjusted['modalite'] == modalite_key]
    values = df_m['SCL_uS_adjusted'].values
    
    all_values_adjusted[modalite_key] = values
    
    stats_by_modalite[modalite_key] = {
        'n': len(values),
        'mean': np.mean(values) if len(values) > 0 else 0,
        'std': np.std(values) if len(values) > 0 else 0,
        'sem': sem(values) if len(values) > 1 else 0,
        'cv': np.std(values) / np.mean(values) if len(values) > 0 and np.mean(values) != 0 else 0
    }
    
    print(f"\n{MODALITES_NAMES[modalite_key]}:")
    print(f"  n = {stats_by_modalite[modalite_key]['n']}")
    print(f"  mean = {stats_by_modalite[modalite_key]['mean']:.4f} uS")
    print(f"  std = {stats_by_modalite[modalite_key]['std']:.4f}")

# ============================================================================
# COMPARAISON STATISTIQUE DES 3 MODALITES
# ============================================================================

print("\n" + "="*80)
print("COMPARAISON STATISTIQUE DES 3 MODALITES")
print("="*80)

f_stat, p_val_anova = f_oneway(
    all_values_adjusted['image'], 
    all_values_adjusted['music'], 
    all_values_adjusted['combined']
)

t_im, p_im = ttest_ind(all_values_adjusted['image'], all_values_adjusted['music'])
t_ic, p_ic = ttest_ind(all_values_adjusted['image'], all_values_adjusted['combined'])
t_mc, p_mc = ttest_ind(all_values_adjusted['music'], all_values_adjusted['combined'])

print(f"\nANOVA: F={f_stat:.3f}, p={p_val_anova:.4f}")
print(f"\nT-tests (corriges pour la sensibilisation):")
print(f"  Image vs Music: t={t_im:.3f}, p={p_im:.4f}")
print(f"  Image vs Combined: t={t_ic:.3f}, p={p_ic:.4f}")
print(f"  Music vs Combined: t={t_mc:.3f}, p={p_mc:.4f}")

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

# Boxplot des 3 modalites
fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AX)

data_to_plot = [all_values_adjusted['image'], all_values_adjusted['music'], all_values_adjusted['combined']]
labels = ['IMAGE', 'MUSIQUE', 'COMBINE']
colors = ['#58a6ff', '#3fb950', '#f85149']

bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, showmeans=True,
                meanprops={'marker': 'o', 'markerfacecolor': '#ffa657', 
                          'markeredgecolor': 'white', 'markersize': 8})

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')

ax.text(0.5, 0.95, f'ANOVA: F={f_stat:.3f}, p={p_val_anova:.4f}', transform=ax.transAxes,
        ha='center', va='top', fontsize=12, color=TEXT_C,
        bbox=dict(boxstyle='round', facecolor=BG_AX, edgecolor=GRID_C))

ax.set_ylabel('SCL ajusté (uS) - sensibilisation soustraite', color=LABEL_C, fontsize=12)
ax.set_title('Comparaison des 3 modalités (correction pour sensibilisation globale)', 
             color=TEXT_C, fontsize=14, fontweight='bold')
ax.tick_params(colors=LABEL_C)
ax.grid(True, color=GRID_C, alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'comparaison_3_modalites_corrige.png'), 
            dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("  -> Saved: comparaison_3_modalites_corrige.png")

# Barres avec erreurs
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AX)

x = np.arange(3)
means = [stats_by_modalite['image']['mean'], 
         stats_by_modalite['music']['mean'], 
         stats_by_modalite['combined']['mean']]
errors = [stats_by_modalite['image']['sem'], 
          stats_by_modalite['music']['sem'], 
          stats_by_modalite['combined']['sem']]

bars = ax.bar(x, means, yerr=errors, color=colors, alpha=0.7, edgecolor='white', capsize=5, width=0.6)

for bar, mean_val, err_val in zip(bars, means, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err_val + 0.01,
            f'{mean_val:.3f}', ha='center', va='bottom', fontsize=11, color=TEXT_C)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12, color=LABEL_C)
ax.set_ylabel('SCL ajusté (uS) - sensibilisation soustraite', color=LABEL_C, fontsize=12)
ax.set_title('Comparaison des 3 modalités (Moyenne ± SEM après correction)', 
             color=TEXT_C, fontsize=14, fontweight='bold')
ax.tick_params(colors=LABEL_C)
ax.grid(True, color=GRID_C, alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'comparaison_3_modalites_bars_corrige.png'), 
            dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close()
print("  -> Saved: comparaison_3_modalites_bars_corrige.png")

# ============================================================================
# SAUVEGARDE
# ============================================================================

print("\n" + "="*80)
print("SAUVEGARDE DES FICHIERS CSV")
print("="*80)

stats_df = []
for modalite_key in ['image', 'music', 'combined']:
    stats_df.append({
        'modalite': MODALITES_NAMES[modalite_key],
        'n_participants': stats_by_modalite[modalite_key]['n'],
        'mean_SCL_ajuste_uS': round(stats_by_modalite[modalite_key]['mean'], 4),
        'std_SCL_ajuste_uS': round(stats_by_modalite[modalite_key]['std'], 4),
        'sem_SCL_ajuste_uS': round(stats_by_modalite[modalite_key]['sem'], 4),
        'cv': round(stats_by_modalite[modalite_key]['cv'], 4)
    })

df_stats = pd.DataFrame(stats_df)
df_stats.to_csv(os.path.join(OUTPUT_PATH, 'statistiques_modalites_corrige.csv'), index=False)
print("  Saved: statistiques_modalites_corrige.csv")

tests_df = pd.DataFrame([
    {'test': 'ANOVA', 'comparaison': 'Image vs Music vs Combined', 'statistic': f_stat, 'p_value': p_val_anova},
    {'test': 't-test', 'comparaison': 'Image vs Music', 'statistic': t_im, 'p_value': p_im},
    {'test': 't-test', 'comparaison': 'Image vs Combined', 'statistic': t_ic, 'p_value': p_ic},
    {'test': 't-test', 'comparaison': 'Music vs Combined', 'statistic': t_mc, 'p_value': p_mc}
])
tests_df.to_csv(os.path.join(OUTPUT_PATH, 'resultats_tests_statistiques_corrige.csv'), index=False)
print("  Saved: resultats_tests_statistiques_corrige.csv")

# Pentes par participant
slopes_df = pd.DataFrame([
    {'participant': pid, 'pente_sensibilisation_uS_s': slope} 
    for pid, slope in participant_slopes.items()
])
slopes_df.to_csv(os.path.join(OUTPUT_PATH, 'pentes_sensibilisation_participants.csv'), index=False)
print("  Saved: pentes_sensibilisation_participants.csv")

# ============================================================================
# RESUME FINAL
# ============================================================================

print("\n" + "="*80)
print("RESUME FINAL")
print("="*80)

for modalite_key in ['image', 'music', 'combined']:
    s = stats_by_modalite[modalite_key]
    print(f"{MODALITES_NAMES[modalite_key]}: mean={s['mean']:.4f} uS, n={s['n']}")

print(f"\nANOVA: F={f_stat:.3f}, p={p_val_anova:.4f}")
print(f"\nImage vs Combined: p={p_ic:.4f}")
print(f"Music vs Combined: p={p_mc:.4f}")

print("\n" + "="*80)
print("ANALYSE COMPLETE - TERMINEE")
print("="*80)