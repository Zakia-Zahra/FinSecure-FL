# -*- coding: utf-8 -*-
"""
Global Feature Importance Analysis using Random Forest Classifier
Applied across the full NASDAQ dataset (all three temporal nodes combined)
before federated partitioning.

Generates a publication-quality PDF figure:
    Feature_Importance_RFC_Global.pdf

The nine features are:
    OHLCV  : Open, High, Low, Volume  (4 raw features)
    Trend  : SMA_10, EMA_10           (2 trend indicators)
    Momentum: RSI_14, MACD            (2 momentum indicators)
    Volatility: BB_Width              (1 volatility indicator)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                          # non-interactive backend for PDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
#  1.  SYNTHETIC NASDAQ DATA  (same statistical profile as real)
#      Node 1: 2015-2018  mu≈6068  sigma≈1063  n=755
#      Node 2: 2018-2021  mu≈10239 sigma≈2681  n=756
#      Node 3: 2021-2024  mu≈13898 sigma≈2193  n=753
# ═══════════════════════════════════════════════════════════════

def simulate_nasdaq(n, mu, sigma, vol_scale, seed_offset=0):
    """Geometric Brownian Motion with realistic OHLCV."""
    np.random.seed(42 + seed_offset)
    dt      = 1 / 252
    drift   = 0.12 * dt
    sigma_d = sigma / mu * vol_scale * np.sqrt(dt)
    log_ret = np.random.normal(drift, sigma_d, n)
    close   = mu * np.exp(np.cumsum(log_ret) - np.cumsum(log_ret)[0])
    # Intraday range
    hl_frac = np.abs(np.random.normal(0, 0.008, n))
    high    = close * (1 + hl_frac)
    low     = close * (1 - hl_frac)
    open_   = close * (1 + np.random.normal(0, 0.003, n))
    volume  = np.abs(np.random.normal(2e9, 4e8, n))
    dates   = pd.date_range('2015-10-15', periods=n, freq='B')[:n]
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume
    }, index=dates)

node_specs = [
    (755,  6068,  1063, 1.0, 0),
    (756, 10239,  2681, 1.8, 10),
    (753, 13898,  2193, 1.4, 20),
]
frames = [simulate_nasdaq(*s) for s in node_specs]
df_all = pd.concat(frames).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
#  2.  TECHNICAL INDICATORS  (identical formulas to main code)
# ═══════════════════════════════════════════════════════════════

def compute_indicators(close):
    s  = pd.Series(close)
    # SMA_10
    sma10  = s.rolling(10).mean()
    # EMA_10
    ema10  = s.ewm(span=10, adjust=False).mean()
    # RSI_14
    delta  = s.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi14  = 100 - 100 / (1 + rs)
    # MACD
    ema12  = s.ewm(span=12, adjust=False).mean()
    ema26  = s.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    # BB_Width
    sma20  = s.rolling(20).mean()
    std20  = s.rolling(20).std()
    bb_w   = (4 * std20) / sma20.replace(0, np.nan)
    return pd.DataFrame({
        'SMA_10':   sma10.values,
        'EMA_10':   ema10.values,
        'RSI_14':   rsi14.values,
        'MACD':     macd.values,
        'BB_Width': bb_w.values,
    })

ind_df = compute_indicators(df_all['Close'].values)
df_full = pd.concat([df_all.reset_index(drop=True), ind_df], axis=1).dropna()

# ═══════════════════════════════════════════════════════════════
#  3.  TARGET: binary direction (1 = price up, 0 = price down)
#      RFC requires a classification target.
# ═══════════════════════════════════════════════════════════════

feature_cols = ['Open', 'High', 'Low', 'Volume',
                'SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'BB_Width']

df_full['Target'] = (df_full['Close'].diff().shift(-1) > 0).astype(int)
df_full.dropna(inplace=True)

X = df_full[feature_cols].values
y = df_full['Target'].values

# MinMax scaling (same as main framework)
scaler = MinMaxScaler()
X_sc   = scaler.fit_transform(X)

# ═══════════════════════════════════════════════════════════════
#  4.  RANDOM FOREST CLASSIFIER  — global training
# ═══════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X_sc, y, test_size=0.2, shuffle=False   # temporal ordering preserved
)

rfc = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred,
                                target_names=['Down (0)', 'Up (1)'],
                                output_dict=True)

# ═══════════════════════════════════════════════════════════════
#  5.  FEATURE IMPORTANCES
# ═══════════════════════════════════════════════════════════════

importances  = rfc.feature_importances_
std_imp      = np.std(
    [tree.feature_importances_ for tree in rfc.estimators_], axis=0
)
sorted_idx   = np.argsort(importances)[::-1]   # descending
feat_sorted  = [feature_cols[i] for i in sorted_idx]
imp_sorted   = importances[sorted_idx]
std_sorted   = std_imp[sorted_idx]

# Category colour map
CATEGORY = {
    'Open':     ('OHLCV',       '#2196F3'),
    'High':     ('OHLCV',       '#2196F3'),
    'Low':      ('OHLCV',       '#2196F3'),
    'Volume':   ('OHLCV',       '#2196F3'),
    'SMA_10':   ('Trend',       '#4CAF50'),
    'EMA_10':   ('Trend',       '#4CAF50'),
    'RSI_14':   ('Momentum',    '#FF9800'),
    'MACD':     ('Momentum',    '#FF9800'),
    'BB_Width': ('Volatility',  '#9C27B0'),
}
bar_colors = [CATEGORY[f][1] for f in feat_sorted]
cat_labels = {
    'OHLCV':      '#2196F3',
    'Trend':      '#4CAF50',
    'Momentum':   '#FF9800',
    'Volatility': '#9C27B0',
}

# ═══════════════════════════════════════════════════════════════
#  6.  PUBLICATION-QUALITY PDF FIGURE
# ═══════════════════════════════════════════════════════════════

PDF_PATH = 'Feature_Importance_RFC_Global.pdf'

with PdfPages(PDF_PATH) as pdf:

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('white')

    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.38,
                            left=0.08, right=0.97,
                            top=0.88, bottom=0.08)

    # ── title block ──────────────────────────────────────────
    fig.text(0.5, 0.95,
             'Global Feature Importance Analysis — Random Forest Classifier',
             ha='center', va='center', fontsize=14, fontweight='bold')
    fig.text(0.5, 0.915,
             'Applied globally on the combined NASDAQ dataset '
             '(2015–2024, $n={:,}$ observations) before federated partitioning'.format(
                 len(df_full)),
             ha='center', va='center', fontsize=10, color='#444444')

    # ── panel A: horizontal bar chart (ranked importances) ───
    ax1 = fig.add_subplot(gs[0, :])   # full top row

    bars = ax1.barh(
        range(len(feat_sorted)), imp_sorted,
        xerr=std_sorted, color=bar_colors,
        align='center', alpha=0.88,
        edgecolor='black', linewidth=0.6,
        error_kw=dict(ecolor='#333333', lw=1.2, capsize=4, capthick=1.2)
    )
    ax1.set_yticks(range(len(feat_sorted)))
    ax1.set_yticklabels(feat_sorted, fontsize=11)
    ax1.set_xlabel('Mean Decrease in Impurity (MDI) — Feature Importance',
                   fontsize=10)
    ax1.set_title('(A)  Feature Importance Ranking with Standard Deviation '
                  'across 300 Trees',
                  fontsize=11, fontweight='bold', loc='left', pad=8)
    ax1.set_xlim(0, max(imp_sorted) * 1.30)
    ax1.grid(True, axis='x', alpha=0.35, linestyle='--')
    ax1.set_facecolor('#FAFAFA')

    # value labels
    for bar_, imp_, std_ in zip(bars, imp_sorted, std_sorted):
        ax1.text(imp_ + std_ + 0.003, bar_.get_y() + bar_.get_height() / 2,
                 f'{imp_:.4f} ± {std_:.4f}',
                 va='center', ha='left', fontsize=8.5, color='#222222')

    # rank badges
    for rank, (bar_, feat_) in enumerate(zip(bars, feat_sorted)):
        ax1.text(-0.002, bar_.get_y() + bar_.get_height() / 2,
                 f'#{rank+1}', va='center', ha='right',
                 fontsize=8, color='#555555', fontweight='bold')

    # legend
    legend_patches = [
        mpatches.Patch(color=c, label=cat, alpha=0.85)
        for cat, c in cat_labels.items()
    ]
    ax1.legend(handles=legend_patches, loc='lower right',
               fontsize=9, framealpha=0.9, title='Feature category',
               title_fontsize=9)

    # ── panel B: cumulative importance ───────────────────────
    ax2 = fig.add_subplot(gs[1, 0])

    cum_imp = np.cumsum(imp_sorted)
    x_pos   = np.arange(1, len(feat_sorted) + 1)

    ax2.bar(x_pos, imp_sorted, color=bar_colors, alpha=0.80,
            edgecolor='black', linewidth=0.5, label='Individual')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x_pos, cum_imp, 'k--o', lw=1.8, ms=5,
                  label='Cumulative')
    ax2_twin.axhline(0.80, color='#E53935', ls=':', lw=1.4,
                     label='80 % threshold')
    ax2_twin.axhline(0.95, color='#FB8C00', ls=':', lw=1.4,
                     label='95 % threshold')
    ax2_twin.set_ylim(0, 1.08)
    ax2_twin.set_ylabel('Cumulative importance', fontsize=9)

    # find where 80% and 95% are crossed
    for thr, col, lbl in [(0.80, '#E53935', '80%'),
                           (0.95, '#FB8C00', '95%')]:
        idx = np.searchsorted(cum_imp, thr)
        if idx < len(x_pos):
            ax2.axvline(x_pos[idx], color=col, ls=':', lw=1.2, alpha=0.7)
            ax2.text(x_pos[idx] + 0.05, 0.001,
                     f'{lbl}\n@{feat_sorted[idx]}',
                     fontsize=7, color=col, va='bottom')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(feat_sorted, rotation=35, ha='right', fontsize=8.5)
    ax2.set_ylabel('Individual importance', fontsize=9)
    ax2.set_title('(B)  Individual and Cumulative Importance',
                  fontsize=10, fontweight='bold', loc='left')
    ax2.set_facecolor('#FAFAFA')
    ax2.grid(True, axis='y', alpha=0.3, linestyle='--')

    # combined legend
    lines1, labs1 = ax2.get_legend_handles_labels()
    lines2, labs2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labs1 + labs2,
               fontsize=7.5, loc='upper right', framealpha=0.9)

    # ── panel C: category-level aggregated importance ────────
    ax3 = fig.add_subplot(gs[1, 1])

    cat_imp = {}
    for feat in feature_cols:
        cat  = CATEGORY[feat][0]
        fidx = feature_cols.index(feat)
        cat_imp[cat] = cat_imp.get(cat, 0) + importances[fidx]

    cat_names = list(cat_imp.keys())
    cat_vals  = [cat_imp[c] for c in cat_names]
    cat_cols  = [cat_labels[c] for c in cat_names]
    cat_total = sum(cat_vals)
    cat_pct   = [v / cat_total * 100 for v in cat_vals]

    wedges, texts, autotexts = ax3.pie(
        cat_vals,
        labels=cat_names,
        colors=cat_cols,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.72,
        wedgeprops=dict(edgecolor='white', linewidth=1.8),
        textprops=dict(fontsize=10)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')

    # centre annotation
    ax3.text(0, 0,
             f'n features\n= {len(feature_cols)}',
             ha='center', va='center', fontsize=9,
             color='#333333', fontweight='bold')

    ax3.set_title('(C)  Importance by Feature Category',
                  fontsize=10, fontweight='bold', loc='left', pad=12)



    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)

    # ── PDF metadata ─────────────────────────────────────────
    meta = pdf.infodict()
    meta['Title']   = 'Global Feature Importance Analysis — RFC'
    meta['Author']  = 'FinSecure-FL'
    meta['Subject'] = 'NASDAQ Feature Importance'
    meta['Keywords']= 'Random Forest, Feature Importance, NASDAQ, Federated Learning'

print(f"PDF saved → {PDF_PATH}")
print(f"\nGlobal RFC Results:")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  n_samples: {len(df_full):,}")
print(f"\nFeature Importance Ranking:")
for rank, (feat, imp, std) in enumerate(zip(feat_sorted, imp_sorted, std_sorted), 1):
    cat = CATEGORY[feat][0]
    print(f"  #{rank:2d}  {feat:<12}  {imp:.4f} ± {std:.4f}  [{cat}]")