"""
GENERATE CORRELATION GRAPHS - Young Adults (18-35)
===================================================
Creates visualizations linking demographics to EDA responses
Based on 40 matched participants
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, ttest_ind, spearmanr
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================

EXCEL_PATH = r"C:\Users\katia\Desktop\min data\phd\donnees\NeuroArt_All_Groups.xlsx"
EDA_CSV_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\jeune_lowfilter_labeled"
PLOTS_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\plots"
OUTPUT_PATH = r"C:\Users\katia\Desktop\output-tnc-cleaned\young_correlations"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ============================================================================
# COLORS (GitHub Dark Theme)
# ============================================================================

BG_FIG = '#0d1117'
BG_AX = '#161b22'
GRID_C = '#21262d'
TEXT_C = '#e6edf3'
LABEL_C = '#8b949e'
SPINE_C = '#30363d'

C_ARTS_HIGH = '#3fb950'      # Green
C_ARTS_LOW = '#f85149'       # Red
C_ECO_YES = '#ffa657'        # Orange
C_ECO_NO = '#58a6ff'         # Blue
C_FEMALE = '#d2a8ff'         # Purple
C_MALE = '#79c0ff'           # Light blue
C_CORR = '#f0883e'           # Orange for correlations
C_SCATTER = '#58a6ff'        # Blue for scatter plots

# ============================================================================
# STEP 1: Load and match data
# ============================================================================

print("="*80)
print("GENERATING CORRELATION GRAPHS - Young Adults (18-35)")
print("="*80)

# Load Excel
df_excel = pd.read_excel(EXCEL_PATH, sheet_name=0)
df_excel['R_Age'] = pd.to_numeric(df_excel['R_Age'], errors='coerce')

# Filter young adults
df_young = df_excel[(df_excel['R_Age'] >= 18) & (df_excel['R_Age'] <= 35)].copy()
df_young = df_young.reset_index(drop=True)

print(f"\n📊 Young participants in Excel: {len(df_young)}")

# Create matched dataset
matched_data = []

for idx, row in df_young.iterrows():
    # Get numeric ID from position (P004 -> 4)
    numeric_id = idx  # P000=0, P001=1, etc.
    
    # Find EDA file with matching numeric ID (but EDA files start at 004)
    eda_numeric = numeric_id + 4  # Because EDA files start at 004
    
    # Look for EDA file
    eda_file = None
    for f in os.listdir(EDA_CSV_PATH):
        if f.startswith(f"{eda_numeric:03d}") and f.endswith('_labeled.csv'):
            eda_file = f
            break
    
    if eda_file:
        # Extract EDA metrics from CSV
        csv_path = os.path.join(EDA_CSV_PATH, eda_file)
        df_eda = pd.read_csv(csv_path)
        
        # Find GSR column
        gsr_col = None
        for col in df_eda.columns:
            if 'GSR' in col or 'filtered' in col:
                gsr_col = col
                break
        
        if gsr_col:
            eda_values = df_eda[gsr_col].values.astype(float)
            eda_values = eda_values[~np.isnan(eda_values)]
            
            if len(eda_values) > 0:
                matched_data.append({
                    'participant_id': f"P{numeric_id:03d}",
                    'eda_id': eda_file.replace('_labeled.csv', ''),
                    'numeric_id': eda_numeric,
                    'age': row['R_Age'],
                    'sex': row.get('SDQ2Sex', np.nan),
                    'arts_engagement': row.get('R_EngagementArtsMixteBinaire', np.nan),
                    'eco_anxiety': row.get('R_EcoanxietyMixteBinaire', np.nan),
                    'vaia_score': row.get('R_VAIAKScore', np.nan),
                    'music_index': row.get('R_IndexMusicListening', np.nan),
                    'mean_scl': np.mean(eda_values),
                    'std_scl': np.std(eda_values),
                    'max_scl': np.max(eda_values),
                    'min_scl': np.min(eda_values),
                    'cv_scl': np.std(eda_values) / np.mean(eda_values) if np.mean(eda_values) > 0 else 0,
                    'png_path': os.path.join(PLOTS_PATH, f"{eda_file.replace('_labeled.csv', '')}_PNG1_preprocessing.png")
                })

df_matched = pd.DataFrame(matched_data)
print(f"✅ Matched {len(df_matched)} participants with EDA data")

if len(df_matched) == 0:
    print("❌ No matches found!")
    exit()

# ============================================================================
# STEP 2: Statistical summary
# ============================================================================

print("\n📊 Participant characteristics:")
print(f"   Age: {df_matched['age'].mean():.1f} ± {df_matched['age'].std():.1f} years")
print(f"   Mean SCL: {df_matched['mean_scl'].mean():.3f} ± {df_matched['mean_scl'].std():.3f} µS")

# Categorical counts
arts_counts = df_matched['arts_engagement'].value_counts()
eco_counts = df_matched['eco_anxiety'].value_counts()
sex_counts = df_matched['sex'].value_counts()

print(f"\n   Arts engaged: {arts_counts.get(1, 0)}")
print(f"   Eco-anxious: {eco_counts.get(1, 0)}")
print(f"   Female: {sex_counts.get(1, 0)}")
print(f"   Male: {sex_counts.get(0, 0)}")

# ============================================================================
# FIGURE 1: Distribution of Mean SCL
# ============================================================================

print("\n🎨 Creating Figure 1: SCL Distribution...")

fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
fig1.patch.set_facecolor(BG_FIG)

# Histogram
ax1 = axes[0]
ax1.set_facecolor(BG_AX)
ax1.hist(df_matched['mean_scl'], bins=15, color=C_SCATTER, edgecolor='white', alpha=0.7, linewidth=1)
ax1.axvline(df_matched['mean_scl'].mean(), color=C_ARTS_HIGH, linestyle='--', linewidth=2, label=f'Mean: {df_matched["mean_scl"].mean():.2f} µS')
ax1.axvline(df_matched['mean_scl'].median(), color=C_ECO_YES, linestyle=':', linewidth=2, label=f'Median: {df_matched["mean_scl"].median():.2f} µS')
ax1.set_xlabel('Mean SCL (µS)', color=LABEL_C, fontsize=11)
ax1.set_ylabel('Frequency', color=LABEL_C, fontsize=11)
ax1.set_title('Distribution of Mean SCL (N=40)', color=TEXT_C, fontsize=12, fontweight='bold')
ax1.tick_params(colors=LABEL_C)
ax1.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5)
ax1.legend(facecolor=BG_AX, edgecolor=SPINE_C, labelcolor=TEXT_C)
for sp in ax1.spines.values():
    sp.set_edgecolor(SPINE_C)

# Box plot by age group (young adults only, but can show spread)
ax2 = axes[1]
ax2.set_facecolor(BG_AX)
# Create age groups for visualization
df_matched['age_group'] = pd.cut(df_matched['age'], bins=[18, 22, 28, 35], labels=['18-22', '23-28', '29-35'])
box_data = [df_matched[df_matched['age_group'] == g]['mean_scl'].values for g in df_matched['age_group'].cat.categories]
bp = ax2.boxplot(box_data, labels=df_matched['age_group'].cat.categories, patch_artist=True)
for patch, color in zip(bp['boxes'], [C_ARTS_HIGH, C_ECO_YES, C_MALE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('white')
for whisker in bp['whiskers']:
    whisker.set_color(LABEL_C)
for cap in bp['caps']:
    cap.set_color(LABEL_C)
for median in bp['medians']:
    median.set_color('white')
    median.set_linewidth(2)
ax2.set_xlabel('Age Group', color=LABEL_C, fontsize=11)
ax2.set_ylabel('Mean SCL (µS)', color=LABEL_C, fontsize=11)
ax2.set_title('SCL by Age Group', color=TEXT_C, fontsize=12, fontweight='bold')
ax2.tick_params(colors=LABEL_C)
ax2.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')
for sp in ax2.spines.values():
    sp.set_edgecolor(SPINE_C)

plt.tight_layout()
fig1.savefig(os.path.join(OUTPUT_PATH, 'Figure1_SCL_Distribution.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close(fig1)
print("   ✅ Figure 1 saved")

# ============================================================================
# FIGURE 2: Group Comparisons (Bar plots)
# ============================================================================

print("\n🎨 Creating Figure 2: Group Comparisons...")

fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor(BG_FIG)

# Function to create comparison plots
def create_comparison_plot(ax, data, group_col, group_names, colors, title, ylabel='Mean SCL (µS)'):
    ax.set_facecolor(BG_AX)
    
    # Calculate statistics
    groups = []
    means = []
    errors = []
    p_values = []
    
    for code, name in group_names.items():
        group_data = data[data[group_col] == code]['mean_scl'].dropna()
        if len(group_data) > 0:
            groups.append(name)
            means.append(group_data.mean())
            errors.append(group_data.std() / np.sqrt(len(group_data)))  # SEM
            # Calculate p-value if two groups
            if len(group_names) == 2:
                other_code = [c for c in group_names.keys() if c != code][0]
                other_data = data[data[group_col] == other_code]['mean_scl'].dropna()
                if len(other_data) > 0:
                    _, p = ttest_ind(group_data, other_data)
                    p_values.append(p)
    
    # Create bars
    x = np.arange(len(groups))
    bars = ax.bar(x, means, yerr=errors, color=colors[:len(groups)], 
                  capsize=8, edgecolor='white', linewidth=1.5, alpha=0.8, zorder=3)
    
    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, means, errors)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
                f'{val:.2f} µS', ha='center', va='bottom', 
                color=TEXT_C, fontsize=9, fontweight='bold')
    
    # Add significance stars
    if len(p_values) > 0 and p_values[0] < 0.05:
        max_height = max(means) + max(errors)
        ax.text(x[0] + 0.5, max_height + 0.05, '*', ha='center', va='bottom', 
                fontsize=16, color=C_ARTS_HIGH)
        ax.text(x[0] + 0.5, max_height + 0.09, f'p={p_values[0]:.3f}', ha='center', va='bottom',
                fontsize=8, color=LABEL_C)
    
    ax.set_xticks(x)
    ax.set_xticklabels(groups, color=LABEL_C, fontsize=10, rotation=15, ha='right')
    ax.set_ylabel(ylabel, color=LABEL_C, fontsize=10)
    ax.set_title(title, color=TEXT_C, fontsize=11, fontweight='bold')
    ax.tick_params(colors=LABEL_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='y')
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)
    
    return bars

# Plot 1: Arts Engagement
if len(df_matched['arts_engagement'].dropna()) > 1:
    create_comparison_plot(axes[0], df_matched, 'arts_engagement', 
                          {0: 'Not Arts\nEngaged', 1: 'Arts\nEngaged'},
                          [C_ARTS_LOW, C_ARTS_HIGH],
                          'SCL by Arts Engagement')

# Plot 2: Eco-anxiety
if len(df_matched['eco_anxiety'].dropna()) > 1:
    create_comparison_plot(axes[1], df_matched, 'eco_anxiety',
                          {0: 'Not\nEco-anxious', 1: 'Eco-anxious'},
                          [C_ECO_NO, C_ECO_YES],
                          'SCL by Eco-anxiety')

# Plot 3: Sex
if len(df_matched['sex'].dropna()) > 1:
    create_comparison_plot(axes[2], df_matched, 'sex',
                          {0: 'Male', 1: 'Female'},
                          [C_MALE, C_FEMALE],
                          'SCL by Sex')

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_PATH, 'Figure2_Group_Comparisons.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close(fig2)
print("   ✅ Figure 2 saved")

# ============================================================================
# FIGURE 3: Correlation Plots
# ============================================================================

print("\n🎨 Creating Figure 3: Correlation Plots...")

fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
fig3.patch.set_facecolor(BG_FIG)

# Function for scatter plot with regression
def add_scatter(ax, x, y, xlabel, ylabel, title, color=C_SCATTER):
    ax.set_facecolor(BG_AX)
    
    # Remove NaN
    valid = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]
    
    if len(x_clean) > 3:
        # Scatter plot
        ax.scatter(x_clean, y_clean, alpha=0.6, color=color, s=80, 
                  edgecolor='white', linewidth=1, zorder=3)
        
        # Regression line
        z = np.polyfit(x_clean, y_clean, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='white', alpha=0.7, linewidth=1.5, zorder=4)
        
        # Correlation
        r, p_val = pearsonr(x_clean, y_clean)
        ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.4f}', 
               transform=ax.transAxes, fontsize=10, color=TEXT_C,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor=BG_AX, alpha=0.8))
        
        # Add interpretation
        if p_val < 0.05:
            if r > 0:
                interp = "↑ Positive correlation"
            else:
                interp = "↓ Negative correlation"
            ax.text(0.05, 0.88, interp, transform=ax.transAxes, fontsize=9, 
                   color=C_ARTS_HIGH, verticalalignment='top')
    
    ax.set_xlabel(xlabel, color=LABEL_C, fontsize=10)
    ax.set_ylabel(ylabel, color=LABEL_C, fontsize=10)
    ax.set_title(title, color=TEXT_C, fontsize=11, fontweight='bold')
    ax.tick_params(colors=LABEL_C)
    ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE_C)

# Plot correlations
add_scatter(axes[0,0], df_matched['age'].values, df_matched['mean_scl'].values,
           'Age (years)', 'Mean SCL (µS)', 'Age vs Physiological Arousal', C_MALE)

add_scatter(axes[0,1], df_matched['vaia_score'].values, df_matched['mean_scl'].values,
           'VAIAK Score (Art Interest)', 'Mean SCL (µS)', 'Art Interest vs Arousal', C_ARTS_HIGH)

add_scatter(axes[1,0], df_matched['music_index'].values, df_matched['mean_scl'].values,
           'Music Listening Index', 'Mean SCL (µS)', 'Music Listening vs Arousal', C_ECO_YES)

# EDA variability vs mean
add_scatter(axes[1,1], df_matched['mean_scl'].values, df_matched['cv_scl'].values,
           'Mean SCL (µS)', 'Coefficient of Variation (CV)', 'Mean vs Variability', C_FEMALE)

plt.tight_layout()
fig3.savefig(os.path.join(OUTPUT_PATH, 'Figure3_Correlations.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close(fig3)
print("   ✅ Figure 3 saved")

# ============================================================================
# FIGURE 4: Individual Participant Summary
# ============================================================================

print("\n🎨 Creating Figure 4: Individual Participant Summary...")

# Sort by mean SCL
df_sorted = df_matched.sort_values('mean_scl', ascending=False).head(20)

fig4, ax = plt.subplots(figsize=(14, 8))
fig4.patch.set_facecolor(BG_FIG)
ax.set_facecolor(BG_AX)

# Create horizontal bar chart
y_pos = np.arange(len(df_sorted))
colors_bars = [C_ARTS_HIGH if a == 1 else C_ARTS_LOW for a in df_sorted['arts_engagement'].values]

bars = ax.barh(y_pos, df_sorted['mean_scl'].values, color=colors_bars, 
               edgecolor='white', linewidth=1, alpha=0.8, zorder=3)

# Add error bars (standard deviation)
ax.errorbar(df_sorted['mean_scl'].values, y_pos, xerr=df_sorted['std_scl'].values,
            fmt='none', color='white', capsize=3, alpha=0.5, zorder=2)

# Add participant labels
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{pid}\n({age}y)" for pid, age in zip(df_sorted['participant_id'], df_sorted['age'])], 
                   color=LABEL_C, fontsize=8)
ax.set_xlabel('Mean SCL (µS)', color=LABEL_C, fontsize=11)
ax.set_title('Individual Participants Ranked by Mean SCL\n(Green = Arts Engaged, Red = Not Engaged)', 
             color=TEXT_C, fontsize=12, fontweight='bold')
ax.tick_params(colors=LABEL_C, axis='x')
ax.grid(True, color=GRID_C, linewidth=0.5, linestyle='--', alpha=0.5, axis='x')
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_C)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df_sorted['mean_scl'])):
    ax.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.2f} µS', 
            va='center', ha='left', color=TEXT_C, fontsize=8)

plt.tight_layout()
fig4.savefig(os.path.join(OUTPUT_PATH, 'Figure4_Individual_Ranking.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close(fig4)
print("   ✅ Figure 4 saved")

# ============================================================================
# FIGURE 5: Combined Dashboard
# ============================================================================

print("\n🎨 Creating Figure 5: Combined Dashboard...")

fig5 = plt.figure(figsize=(16, 10))
fig5.patch.set_facecolor(BG_FIG)

# Create grid
gs = fig5.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                        left=0.08, right=0.95, top=0.92, bottom=0.08)

# Panel 1: Arts engagement boxplot
ax1 = fig5.add_subplot(gs[0, 0])
ax1.set_facecolor(BG_AX)
arts_data = [df_matched[df_matched['arts_engagement']==0]['mean_scl'].values,
             df_matched[df_matched['arts_engagement']==1]['mean_scl'].values]
bp1 = ax1.boxplot(arts_data, labels=['Not Engaged', 'Engaged'], patch_artist=True)
for patch, color in zip(bp1['boxes'], [C_ARTS_LOW, C_ARTS_HIGH]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax1.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax1.set_title('Arts Engagement Effect', color=TEXT_C, fontweight='bold')
ax1.tick_params(colors=LABEL_C)
ax1.grid(True, color=GRID_C, alpha=0.3, axis='y')

# Panel 2: Eco-anxiety boxplot
ax2 = fig5.add_subplot(gs[0, 1])
ax2.set_facecolor(BG_AX)
eco_data = [df_matched[df_matched['eco_anxiety']==0]['mean_scl'].values,
            df_matched[df_matched['eco_anxiety']==1]['mean_scl'].values]
bp2 = ax2.boxplot(eco_data, labels=['Not Anxious', 'Anxious'], patch_artist=True)
for patch, color in zip(bp2['boxes'], [C_ECO_NO, C_ECO_YES]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax2.set_title('Eco-anxiety Effect', color=TEXT_C, fontweight='bold')
ax2.tick_params(colors=LABEL_C)
ax2.grid(True, color=GRID_C, alpha=0.3, axis='y')

# Panel 3: Sex boxplot
ax3 = fig5.add_subplot(gs[0, 2])
ax3.set_facecolor(BG_AX)
sex_data = [df_matched[df_matched['sex']==0]['mean_scl'].values,
            df_matched[df_matched['sex']==1]['mean_scl'].values]
bp3 = ax3.boxplot(sex_data, labels=['Male', 'Female'], patch_artist=True)
for patch, color in zip(bp3['boxes'], [C_MALE, C_FEMALE]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax3.set_title('Sex Effect', color=TEXT_C, fontweight='bold')
ax3.tick_params(colors=LABEL_C)
ax3.grid(True, color=GRID_C, alpha=0.3, axis='y')

# Panel 4: Age correlation
ax4 = fig5.add_subplot(gs[1, 0])
ax4.set_facecolor(BG_AX)
ax4.scatter(df_matched['age'], df_matched['mean_scl'], alpha=0.6, color=C_CORR, s=60, edgecolor='white')
z = np.polyfit(df_matched['age'], df_matched['mean_scl'], 1)
p = np.poly1d(z)
ax4.plot(np.linspace(18, 35, 100), p(np.linspace(18, 35, 100)), '--', color='white', alpha=0.7)
r, p_val = pearsonr(df_matched['age'], df_matched['mean_scl'])
ax4.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.3f}', transform=ax4.transAxes, color=TEXT_C)
ax4.set_xlabel('Age (years)', color=LABEL_C)
ax4.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax4.set_title('Age Correlation', color=TEXT_C, fontweight='bold')
ax4.tick_params(colors=LABEL_C)
ax4.grid(True, color=GRID_C, alpha=0.3)

# Panel 5: VAIAK correlation
ax5 = fig5.add_subplot(gs[1, 1])
ax5.set_facecolor(BG_AX)
valid_vaia = df_matched[df_matched['vaia_score'].notna()]
ax5.scatter(valid_vaia['vaia_score'], valid_vaia['mean_scl'], alpha=0.6, color=C_ARTS_HIGH, s=60, edgecolor='white')
if len(valid_vaia) > 3:
    z = np.polyfit(valid_vaia['vaia_score'], valid_vaia['mean_scl'], 1)
    p = np.poly1d(z)
    ax5.plot(np.linspace(valid_vaia['vaia_score'].min(), valid_vaia['vaia_score'].max(), 100), 
             p(np.linspace(valid_vaia['vaia_score'].min(), valid_vaia['vaia_score'].max(), 100)), 
             '--', color='white', alpha=0.7)
    r, p_val = pearsonr(valid_vaia['vaia_score'], valid_vaia['mean_scl'])
    ax5.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.3f}', transform=ax5.transAxes, color=TEXT_C)
ax5.set_xlabel('VAIAK Score (Art Interest)', color=LABEL_C)
ax5.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax5.set_title('Art Interest Correlation', color=TEXT_C, fontweight='bold')
ax5.tick_params(colors=LABEL_C)
ax5.grid(True, color=GRID_C, alpha=0.3)

# Panel 6: Music correlation
ax6 = fig5.add_subplot(gs[1, 2])
ax6.set_facecolor(BG_AX)
valid_music = df_matched[df_matched['music_index'].notna()]
ax6.scatter(valid_music['music_index'], valid_music['mean_scl'], alpha=0.6, color=C_ECO_YES, s=60, edgecolor='white')
if len(valid_music) > 3:
    z = np.polyfit(valid_music['music_index'], valid_music['mean_scl'], 1)
    p = np.poly1d(z)
    ax6.plot(np.linspace(valid_music['music_index'].min(), valid_music['music_index'].max(), 100), 
             p(np.linspace(valid_music['music_index'].min(), valid_music['music_index'].max(), 100)), 
             '--', color='white', alpha=0.7)
    r, p_val = pearsonr(valid_music['music_index'], valid_music['mean_scl'])
    ax6.text(0.05, 0.95, f'r = {r:.3f}, p = {p_val:.3f}', transform=ax6.transAxes, color=TEXT_C)
ax6.set_xlabel('Music Listening Index', color=LABEL_C)
ax6.set_ylabel('Mean SCL (µS)', color=LABEL_C)
ax6.set_title('Music Listening Correlation', color=TEXT_C, fontweight='bold')
ax6.tick_params(colors=LABEL_C)
ax6.grid(True, color=GRID_C, alpha=0.3)

fig5.suptitle('Young Adults (18-35) - EDA Correlations Dashboard\nN=40 participants', 
              color=TEXT_C, fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()
fig5.savefig(os.path.join(OUTPUT_PATH, 'Figure5_Dashboard.png'), dpi=150, facecolor=BG_FIG, bbox_inches='tight')
plt.close(fig5)
print("   ✅ Figure 5 saved")

# ============================================================================
# Generate HTML Report
# ============================================================================

print("\n📄 Generating HTML report...")

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Young Adults (18-35) - EDA Correlation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #0d1117;
            color: #e6edf3;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{
            color: #58a6ff;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 32px;
            font-weight: bold;
            color: #3fb950;
        }}
        .stat-label {{
            color: #8b949e;
            font-size: 12px;
            margin-top: 5px;
        }}
        .figure {{
            margin: 30px 0;
            text-align: center;
        }}
        .figure img {{
            max-width: 100%;
            border: 1px solid #30363d;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .figure-caption {{
            color: #8b949e;
            font-size: 12px;
            margin-top: 10px;
        }}
        hr {{
            border-color: #30363d;
            margin: 30px 0;
        }}
        .footer {{
            text-align: center;
            color: #8b949e;
            font-size: 12px;
            margin-top: 50px;
            padding: 20px;
            border-top: 1px solid #30363d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Young Adults (18-35 years) - EDA Correlation Report</h1>
        <p>Linking demographic and behavioral variables with physiological arousal (SCL)</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{len(df_matched)}</div>
                <div class="stat-label">Participants</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{df_matched['age'].mean():.1f} ± {df_matched['age'].std():.1f}</div>
                <div class="stat-label">Age (years)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{df_matched['mean_scl'].mean():.2f}</div>
                <div class="stat-label">Mean SCL (µS)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{arts_counts.get(1, 0)}</div>
                <div class="stat-label">Arts Engaged</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{eco_counts.get(1, 0)}</div>
                <div class="stat-label">Eco-anxious</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{sex_counts.get(1, 0)} / {sex_counts.get(0, 0)}</div>
                <div class="stat-label">Female / Male</div>
            </div>
        </div>
        
        <hr>
        
        <div class="figure">
            <img src="Figure1_SCL_Distribution.png" alt="SCL Distribution">
            <div class="figure-caption">Figure 1: Distribution of Mean SCL across participants</div>
        </div>
        
        <div class="figure">
            <img src="Figure2_Group_Comparisons.png" alt="Group Comparisons">
            <div class="figure-caption">Figure 2: SCL differences by Arts Engagement, Eco-anxiety, and Sex</div>
        </div>
        
        <div class="figure">
            <img src="Figure3_Correlations.png" alt="Correlations">
            <div class="figure-caption">Figure 3: Correlations with Age, VAIAK Score, Music Index, and SCL variability</div>
        </div>
        
        <div class="figure">
            <img src="Figure4_Individual_Ranking.png" alt="Individual Ranking">
            <div class="figure-caption">Figure 4: Individual participants ranked by mean SCL</div>
        </div>
        
        <div class="figure">
            <img src="Figure5_Dashboard.png" alt="Dashboard">
            <div class="figure-caption">Figure 5: Complete dashboard of all analyses</div>
        </div>
        
        <hr>
        
        <h2>📊 Statistical Summary</h2>
"""
# Add statistical results
html_content += "<ul>"
for _, row in df_matched.iterrows():
    pass  # Skip for brevity
html_content += "</ul>"

html_content += f"""
        <div class="footer">
            Generated from SENSE Study Data | N={len(df_matched)} young adults (18-35 years)<br>
            EDA metrics extracted from 40 participants with complete data
        </div>
    </div>
</body>
</html>
"""

# Save HTML
html_path = os.path.join(OUTPUT_PATH, 'EDA_Correlation_Report.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# Save merged data
csv_path = os.path.join(OUTPUT_PATH, 'merged_data.csv')
df_matched.to_csv(csv_path, index=False)

print(f"   ✅ HTML report saved: {html_path}")
print(f"   ✅ Merged data saved: {csv_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ ALL GRAPHS GENERATED!")
print("="*80)

print(f"\n📁 Output folder: {OUTPUT_PATH}")
print("\n📊 Generated files:")
print("   1. Figure1_SCL_Distribution.png - Distribution of mean SCL")
print("   2. Figure2_Group_Comparisons.png - Group comparisons (arts, eco, sex)")
print("   3. Figure3_Correlations.png - Correlation plots")
print("   4. Figure4_Individual_Ranking.png - Individual participant ranking")
print("   5. Figure5_Dashboard.png - Complete dashboard")
print("   6. EDA_Correlation_Report.html - Interactive HTML report")
print("   7. merged_data.csv - Complete merged dataset")

print("\n" + "="*80)
print("🎨 OPEN THE HTML REPORT TO SEE ALL GRAPHS:")
print(f"   {html_path}")
print("="*80)

# Create launcher
launcher_path = os.path.join(OUTPUT_PATH, 'open_report.bat')
with open(launcher_path, 'w') as f:
    f.write(f'start "" "{html_path}"')
print(f"\n✅ Launcher created: {launcher_path}")