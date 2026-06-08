# SENSE Study - Electrodermal Activity (EDA) Analysis Pipeline

Complete analysis pipeline for Electrodermal Activity (EDA) data from the SENSE study, investigating emotional responses to nature-inspired artistic experiences in young adults (18-35 years).

## Study Background

The SENSE study (Studies examining Emotional responses to Nature-inspired artiStic Experience) examines how digital reproductions of Impressionist paintings, combined with plant-based sonification (music generated from plants' own bioelectrical signals), elicit emotional responses across different sensory modalities: visual, auditory, and combined.

**Key references:**
- Galery, Fauvet, Djerroud et al. (2025) - Young adults' emotional responses to nature-inspired art
- Fauvet, Galery, Djerroud et al. (2025) - Older adults' emotional responses
- Jang et al. (2014) - Relationship between affective dimensions and physiological responses

## Methods Overview

### Experimental Design
- Participants: 40 young adults (18-35 years, mean age 26.7 ± 4.6)
- Design: One-arm, open-label, pre-post experimental
- Modalities: Visual-only, Auditory-only, Combined (audiovisual)
- Stimuli per participant: 14 stimulations
- Stimulation duration: 40 seconds per stimulus
- Sampling rate: 128 Hz (Shimmer3 GSR+)

### Stimulus Conditions

| Condition | Artwork | Music |
|-----------|---------|-------|
| Experimental (nature-inspired) | Sunny September by Helen McNicoll | Plant sonification |
| Experimental 2 | Les Saules by Claude Monet | Plant sonification |
| Positive reference | The Regattas at Hampton Court by Alfred Sisley | Classical music (DEAM) |
| Negative reference | Smardan Attack by Nicolae Grigorescu | Classical music (DEAM) |

## Processing Pipeline

### 1. Preprocessing (EDAQA - Kleckner et al., 2018)
Four-rule quality assessment:
- Rule 1: EDA out of range (0.05-60 µS)
- Rule 2: Too-rapid changes (>1.0 µS/s or >0.1 µS/100ms)
- Rule 3: Temperature out of range (30-40°C)
- Rule 4: Transitional data (±5 seconds around invalid segments)

Invalid samples are interpolated to preserve temporal continuity.

### 2. Signal Processing (Privratsky et al., 2020; Staib et al., 2015)

| Component | Filter | Cutoff | Reference |
|-----------|--------|--------|-----------|
| SCL (tonic) | Butterworth low-pass | 1 Hz | Boucsein et al. (2012) |
| SCR (phasic) | Butterworth band-pass | 0.0159-5 Hz | Privratsky et al. (2020) |
| Detrending | Cubic spline | knot/30 sec | - |

### 3. Window Extraction
- Pre-stimulus: 5 seconds (baseline)
- Stimulation: 40 seconds (exact)
- Post-stimulus: 5 seconds (recovery)
- Total window: 50 seconds (6400 samples at 128Hz)

### 4. Statistical Analysis
- Within-subject range normalization: (SCL - min) / (max - min)
- Linear Mixed Models (LMM) for temporal trends
- ANOVA for modality comparisons
- Sensitization correction: Subtraction of participant-specific temporal slope

## Key Results

### Modality Comparison (N=40, after sensitization correction)

| Modality | Mean SCL (µS) | ± SEM | vs Combined |
|----------|---------------|-------|--------------|
| Visual | 2.354 | 0.060 | p = 0.028 |
| Auditory | 2.283 | 0.052 | p = 0.003 |
| Combined | 2.420 | 0.058 | reference |

ANOVA: F = 3.957, p = 0.021

### Perceived Beauty Predicts Valence

| Predictor | β | p-value |
|-----------|-----|---------|
| Perceived beauty | 0.36-0.55 | <0.001 |
| Arts engagement | -1.83 (SCL) | 0.008 |
| Eco-anxiety | not significant | - |

## Repository Structure

```
├── README.md
├── requirements.txt
├── LICENSE
│
├── 01_preprocessing/
│   ├── add_stimulus_labels.py
│   ├── extract_windows.py
│   ├── add_modality_column.py
│   └── edaqa.py
│
├── 02_analysis/
│   ├── correlation_analysis.py
│   ├── modalite_analysis.py
│   ├── position_analysis.py
│   ├── complete_analysis_40_jeunes.py
│   ├── visualize_all_participants.py
│   └── compare_029_049.py
│
├── 03_visualization/
│   ├── generate_plots.py
│   ├── png2_statistical_analysis.py
│   └── generate_correlation_graphs.py
│
├── data/
│   ├── raw/
│   ├── stimulus_markers/
│   ├── processed/
│   ├── windows/
│   └── outputs/
│
└── references.bib
```

## Installation

```bash
git clone https://github.com/yourusername/sense-eda-analysis.git
cd sense-eda-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tqdm>=4.62.0
openpyxl>=3.0.0
```

## Methodological References

### Foundational EDA Standards
- Boucsein et al. (2012) - SPR publication standards for EDA measurement
- Lykken & Venables (1971) - Standardization of conductance units and constant-voltage method
- Braithwaite et al. (2015) - Practical Biopac EDA recording guide
- Dawson et al. (2007) - Comprehensive EDA methods chapter
- Critchley (2002) - Neural basis of EDA

### Quality Assessment
- Kleckner et al. (2018) - EDAQA automated quality assessment
- Chen et al. (2015) - Wavelet-based motion artifact removal

### Model-Based Analysis
- Greco et al. (2014) - cvxEDA convex optimization decomposition
- Staib et al. (2015) - PsPM DCM optimization
- Privratsky et al. (2020) - Filter comparison for fMRI EDA
- Kuhn et al. (2022) - Manyverse comparison of SCR quantification methods

### Normalization and Individual Differences
- Ben-Shakhar (1985) - Within-subject z-score standardization
- Jang et al. (2014) - Affective dimensions versus physiological responses

### Theoretical Framework
- Steiner & Barry (2014) - Dishabituation mechanism
- Spinks & Siddle (1976) - Stimulus information and duration effects

## Usage Examples

### Run complete analysis pipeline

```python
python 01_preprocessing/add_stimulus_labels.py
python 01_preprocessing/extract_windows.py
python 01_preprocessing/add_modality_column.py
python 03_visualization/generate_plots.py
```

### Run statistical analysis

```python
python 02_analysis/modalite_analysis.py
python 02_analysis/position_analysis.py
python 02_analysis/complete_analysis_40_jeunes.py
```

### Generate correlation graphs

```python
python 03_visualization/generate_correlation_graphs.py
```

## Output Files

### Preprocessing Visualizations
Four panels per participant:
1. Raw EDA with EDAQA mask
2. SCL Butterworth low-pass 1 Hz (interpolated)
3. SCL spline-detrended
4. SCR band-pass (0.0159-5 Hz) with detected peaks

### Statistical Analysis
- statistiques_modalites_corrige.csv - Modality statistics
- resultats_tests_statistiques_corrige.csv - ANOVA/t-test results
- matrice_participants_stimuli.csv - Participant by stimulus matrix
- pentes_sensibilisation_participants.csv - Individual sensitization slopes

## Configuration Parameters

```python
SAMPLING_RATE = 128
PRE_STIM_DURATION = 5
STIM_DURATION = 40
POST_STIM_DURATION = 5
SCL_LOW_PASS_CUTOFF = 1.0
SCR_HIGH_PASS_CUTOFF = 0.0159
SCR_LOW_PASS_CUTOFF = 5.0
FILTER_ORDER = 4
EDA_MIN_VALID = 0.05
EDA_MAX_VALID = 60.0
MAX_RATE_CHANGE = 1.0
```

## License

MIT License

Copyright (c) 2025 Kevin Galery, Cordelia Fauvet, Katia Djerroud

## Contact
katia djerroud - katia.djerroud@umontreal.ca
Kevin Galery - kevin.galery.ccsmtl@ssss.gouv.qc.ca
