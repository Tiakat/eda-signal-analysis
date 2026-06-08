import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter\004-0-B-2-3.csv"
OUTPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter\plots"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Paramètres
SAMPLING_RATE = 128
LOWCUT = 0.0159      # Hz pour supprimer la dérive
HIGHCUT = 5.0        # Hz

def butter_bandpass(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)

def time_to_seconds(time_str):
    parts = time_str.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    sec_parts = parts[2].split('.')
    seconds = int(sec_parts[0])
    milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000

# ============================================================================
# CHARGEMENT
# ============================================================================

df = pd.read_csv(INPUT_FILE)
df['Time_sec'] = df['Time'].apply(time_to_seconds)
time_sec = df['Time_sec'].values - df['Time_sec'].iloc[0]

signal_raw = df['GSR_raw'].values
signal_lowpass = df['GSR_filtered'].values  # Votre SCL actuel

# Séparation : passe-bande pour extraire les SCR
signal_bandpass = butter_bandpass(signal_raw, LOWCUT, HIGHCUT, SAMPLING_RATE)

# Calcul du SCL tonique (version lissée)
from scipy.ndimage import uniform_filter1d
scl_tonic = uniform_filter1d(signal_raw, size=int(SAMPLING_RATE*10))  # Moyenne sur 10s

# ============================================================================
# GRAPHIQUE UNIQUE : SCL vs SCR
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('SCL (tonique) vs SCR (phasique) - Même participant', fontsize=14)

# Graphique 1 : SCL tonique (votre analyse actuelle)
ax1 = axes[0]
ax1.plot(time_sec, signal_lowpass, 'b-', linewidth=1)
ax1.set_ylabel('SCL (µS)')
ax1.set_title('1. SCL TONIQUE (éveil de fond) - Votre analyse actuelle')
ax1.grid(True, alpha=0.3)

# Graphique 2 : SCR phasique (extrait par passe-bande)
ax2 = axes[1]
ax2.plot(time_sec, signal_bandpass, 'r-', linewidth=0.8)
ax2.set_ylabel('SCR (µS)')
ax2.set_title('2. SCR PHASIQUE (réponses aux stimuli) - Extraits par passe-bande')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.grid(True, alpha=0.3)

# Graphique 3 : Superposition des deux (normalisés)
ax3 = axes[2]
# Normalisation pour superposition
scl_norm = (signal_lowpass - signal_lowpass.min()) / (signal_lowpass.max() - signal_lowpass.min())
scr_norm = (signal_bandpass - signal_bandpass.min()) / (signal_bandpass.max() - signal_bandpass.min())

ax3.plot(time_sec, scl_norm, 'b-', linewidth=1, label='SCL (tonique) - normalisé')
ax3.plot(time_sec, scr_norm, 'r-', linewidth=0.8, label='SCR (phasique) - normalisé')
ax3.set_xlabel('Temps (secondes)')
ax3.set_ylabel('Valeur normalisée')
ax3.set_title('3. Superposition SCL vs SCR (normalisés)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'SCL_vs_SCR_comparison.png'), dpi=150)
plt.show()

# ============================================================================
# EXPLICATION DES DIFFÉRENCES
# ============================================================================

print("\n" + "=" * 70)
print("DIFFÉRENCE ENTRE SCL ET SCR")
print("=" * 70)
print("""
| Caractéristique | SCL (tonique) | SCR (phasique) |
|----------------|---------------|----------------|
| Ce qu'il mesure | Éveil de fond (minutes) | Réponse à un stimulus (secondes) |
| Filtre recommandé | Passe-bas 1 Hz | Passe-bande 0,0159-5 Hz |
| Aspect sur le graphe | Courbe lente, qui dérive | Pics rapides, centrés sur zéro |
| Ce que vous analysez actuellement | ✅ OUI | ❌ NON |
""")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
- Pour le SCL (votre objectif) : votre filtre passe-bas 1 Hz est CORRECT.
- Pour les SCR : il faut un filtre passe-bande 0,0159-5 Hz.
- Les deux ne peuvent pas être analysés avec le même filtre.
- Le graphique ci-dessus montre la différence entre les deux composantes.
""")