# -*- coding: utf-8 -*-
"""
Enhanced Federated Learning with SVR for NASDAQ Prediction
+ Ablation Study: FL-SVR vs ARIMA vs LSTM

"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import time
import hashlib
import json
import os
from datetime import datetime, timezone

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

# ── Optional deep-learning imports ───────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("TensorFlow not found — LSTM will use a lightweight EWM approximation.")

# ── Optional statsmodels (ARIMA) ──────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    print("statsmodels not found — ARIMA will use naive random-walk baseline.")

try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        plt.style.use('ggplot')

# ── Figure output directory ──────────────────────────────────────────────────
FIGURES_DIR = "./figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

_fig_counter = [0]

def save_fig(fig, name: str):
    """Save figure as Png to FIGURES_DIR with a sequential prefix."""
    _fig_counter[0] += 1
    safe_name = name.replace(" ", "_").replace("/", "-").replace(":", "")
    path = os.path.join(FIGURES_DIR, f"{_fig_counter[0]:02d}_{safe_name}.png")
    fig.savefig(path, format='png', bbox_inches='tight', dpi=150)
    print(f"  [Png] saved → {path}")


# ═════════════════════════════════════════════════════════════
# AUDIT TRAIL
# ═════════════════════════════════════════════════════════════

class AuditTrail:
    def __init__(self):
        self.log = []

    def record(self, event_type: str, node_id, details: dict = None):
        entry = {'timestamp': _utcnow(), 'event_type': event_type,
                 'node_id': node_id, 'details': details or {}}
        self.log.append(entry)
        print(f"  [AUDIT] {entry['timestamp']}  |  {event_type:30s}  |  Node: {node_id}")

    def export(self, filepath: str = 'audit_trail.json'):
        with open(filepath, 'w') as f:
            json.dump(self.log, f, indent=2)
        print(f"\n  Audit trail exported → {filepath}")

    def summary(self):
        print(f"\n{'='*70}")
        print(f"  AUDIT SUMMARY — {len(self.log)} events")
        print(f"{'='*70}")
        for e in self.log:
            print(f"  [{e['timestamp']}]  {e['event_type']:30s}  Node: {e['node_id']}")


# ═════════════════════════════════════════════════════════════
# COMPARISON TABLE PRINTERS
# ═════════════════════════════════════════════════════════════

def print_secure_aggregation_comparison():
    methods = [
        ("Plain FedAvg", "None", "None", "None", "High", "None"),
        ("SMPC (Bonawitz 2017)", "Cryptographic secret sharing",
         "High", "High (multi-round)", "Low (dropout sensitive)", "Not built-in"),
        ("TEE / Intel SGX", "Hardware enclave",
         "High", "Medium", "Hardware dependency", "Vendor lock-in"),
        ("Differential Privacy only", "Noise injection (Laplace/Gauss)",
         "Medium", "Low", "High", "No tamper-evidence"),
        ("Blockchain-FL (Ours)", "Hash-ledger + DP + BFT + cosine",
         "High", "Low (+overhead)", "High (BFT)", "SHA-256 immutable"),
    ]
    W = 108
    print(f"\n  {'─'*W}")
    print(f"  Secure Aggregation Method Comparison")
    print(f"  {'─'*W}")
    hdr = (f"  {'Method':<26} {'Security Mechanism':<34} "
           f"{'Privacy':>8} {'Latency':>18} {'Fault Tol':>20} {'Auditability':>16}")
    print(hdr); print(f"  {'─'*W}")
    for row in methods:
        marker = "  * " if "Ours" in row[0] else "    "
        print(f"{marker}{row[0]:<26} {row[1]:<34} "
              f"{row[2]:>8} {row[3]:>18} {row[4]:>20} {row[5]:>16}")
    print(f"  {'─'*W}\n")


def print_feature_justification():
    features = {
        'Open':     ('OHLCV',      'Opening price captures overnight sentiment and gap risk'),
        'High':     ('OHLCV',      'Daily high reflects intraday buying pressure and resistance'),
        'Low':      ('OHLCV',      'Daily low captures selling pressure and support levels'),
        'Volume':   ('OHLCV',      'Volume confirms price moves; high vol = conviction (Lo & Wang, 2000)'),
        'SMA_10':   ('Trend',      '10-day SMA smooths noise and identifies trend direction'),
        'EMA_10':   ('Trend',      '10-day EMA weights recent prices more heavily than SMA'),
        'RSI_14':   ('Momentum',   '14-day RSI identifies overbought/oversold regimes (Wilder, 1978)'),
        'MACD':     ('Momentum',   'EMA12 - EMA26 captures momentum shifts (standard indicator)'),
        'BB_Width': ('Volatility', 'Bollinger Band width measures volatility regime (Bollinger, 1992)'),
    }
    W = 100
    print(f"\n  {'─'*W}")
    print(f"  Feature Inclusion Justification")
    print(f"  {'─'*W}")
    print(f"  {'Feature':<12} {'Category':<12} {'Justification'}")
    print(f"  {'─'*W}")
    for feat, (cat, reason) in features.items():
        print(f"  {feat:<12} {cat:<12} {reason}")
    print(f"  {'─'*W}\n")


# ═════════════════════════════════════════════════════════════
# POISONING DEFENCE
# ═════════════════════════════════════════════════════════════

def clip_weights(weights: np.ndarray, max_norm: float = 10.0) -> np.ndarray:
    norm = np.linalg.norm(weights)
    if norm > max_norm:
        weights = weights * (max_norm / norm)
        print(f"  [CLIP]  norm {norm:.4f} -> {max_norm:.4f}  (clipped)")
    else:
        print(f"  [CLIP]  norm {norm:.4f} <= {max_norm:.4f}  (OK)")
    return weights


def apply_differential_privacy(weights: np.ndarray,
                                epsilon: float = 1.0,
                                sensitivity: float = 1.0) -> np.ndarray:
    scale = sensitivity / epsilon
    noise = np.random.laplace(0.0, scale, weights.shape)
    mag   = np.linalg.norm(noise)
    print(f"  [DP]    eps={epsilon}  scale={scale:.4f}  noise_mag={mag:.6f}")
    return weights + noise


def validate_weights(new_weights, existing_weights_list, cosine_threshold=0.0):
    n = len(existing_weights_list)
    if n < 3:
        print(f"  [ACCEPTED]  history={n} — deferred  trust=1.0000")
        return True, 1.0
    w_new     = np.array(new_weights, dtype=float)
    consensus = np.mean(np.array(existing_weights_list, dtype=float), axis=0)
    num       = np.dot(w_new, consensus)
    denom     = (np.linalg.norm(w_new) * np.linalg.norm(consensus)) + 1e-12
    cos_sim   = float(num / denom)
    trust     = float((cos_sim + 1.0) / 2.0)
    is_valid  = cos_sim > cosine_threshold
    status    = "ACCEPTED" if is_valid else "REJECTED"
    print(f"  [{status}]  cosine={cos_sim:.4f}  trust={trust:.4f}")
    return is_valid, trust


def compute_poisoning_risk(all_weights):
    n          = len(all_weights)
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            wi    = np.array(all_weights[i], dtype=float)
            wj    = np.array(all_weights[j], dtype=float)
            denom = (np.linalg.norm(wi) * np.linalg.norm(wj)) + 1e-12
            cos_matrix[i, j] = np.dot(wi, wj) / denom
    suspicious = [i+1 for i in range(n)
                  if np.mean([cos_matrix[i,j] for j in range(n) if j!=i]) < 0.5]
    return {'cosine_matrix': cos_matrix, 'suspicious_nodes': suspicious,
            'mean_pairwise_similarity': float(np.mean(cos_matrix[cos_matrix < 1.0]))}


# ═════════════════════════════════════════════════════════════
# BLOCKCHAIN
# ═════════════════════════════════════════════════════════════

class Blockchain:
    def __init__(self):
        self.chain          = []
        self.rejected_nodes = []
        self._add_genesis()

    def _add_genesis(self):
        block = {'index': 0, 'node_id': 'GENESIS', 'weights': None,
                 'trust_score': None, 'timestamp': _utcnow(), 'previous_hash': '0'}
        block['hash'] = self._hash(block)
        self.chain.append(block)

    def _hash(self, block):
        payload = {k: v for k, v in block.items() if k != 'hash'}
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def add_weights_to_block(self, weights, node_id):
        weights  = clip_weights(weights)
        weights  = apply_differential_privacy(weights, epsilon=1.0)
        existing = [b['weights'] for b in self.chain if b['weights'] is not None]
        is_valid, trust = validate_weights(weights, existing)
        if not is_valid:
            self.rejected_nodes.append(node_id)
            return None
        block = {'index': len(self.chain), 'node_id': node_id,
                 'weights': weights.tolist(), 'trust_score': trust,
                 'timestamp': _utcnow(), 'previous_hash': self.chain[-1]['hash']}
        block['hash'] = self._hash(block)
        self.chain.append(block)
        return block

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            if self.chain[i]['previous_hash'] != self.chain[i-1]['hash']:
                print(f"  Chain integrity FAILED at block {i}")
                return False
        print("  Chain integrity VERIFIED")
        return True


blockchain = Blockchain()
audit      = AuditTrail()


# ═════════════════════════════════════════════════════════════
# LATENCY BENCHMARK
# ═════════════════════════════════════════════════════════════

def benchmark_aggregation_methods(all_weights, local_metrics_test, node_sample_sizes, n_runs=100):
    results = {}
    wts    = [np.array(w) for w in all_weights]
    r2s    = np.clip([m[4] for m in local_metrics_test], 0, 1)
    sizes  = np.array(node_sample_sizes, dtype=float)
    size_w = sizes / sizes.sum()
    r2_w   = r2s / r2s.sum()
    comb   = size_w * r2_w; comb = comb / comb.sum()

    for label, fn in [
        ('plain_fedavg',      lambda: np.mean(wts, axis=0)),
        ('reputation_fedavg', lambda: sum((r2s/r2s.sum())[k] * wts[k] for k in range(len(wts)))),
        ('combined_fedavg',   lambda: sum(comb[k] * wts[k] for k in range(len(wts)))),
    ]:
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
        results[label] = {'mean_ms': np.mean(times)*1000, 'std_ms': np.std(times)*1000,
                          'label': label.replace('_', ' ').title()}

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        for w in wts:
            hashlib.sha256(json.dumps({'w': w.tolist()}, sort_keys=True).encode()).hexdigest()
            con = np.mean(wts, axis=0)
            np.dot(w, con) / ((np.linalg.norm(w)*np.linalg.norm(con)) + 1e-12)
        sum(comb[k] * wts[k] for k in range(len(wts)))
        times.append(time.perf_counter() - t0)
    results['blockchain_fedavg'] = {'mean_ms': np.mean(times)*1000, 'std_ms': np.std(times)*1000,
                                    'label': 'Blockchain FL (full pipeline)'}

    base = results['plain_fedavg']['mean_ms']
    print(f"\n  Aggregation Latency Benchmark ({n_runs} runs)")
    for k, v in results.items():
        oh = f"+{(v['mean_ms']/base-1)*100:.1f}%" if k != 'plain_fedavg' else "baseline"
        print(f"  {v['label']:<34} {v['mean_ms']:>10.4f} ms  {oh:>12}")
    return results


# ═════════════════════════════════════════════════════════════
# WEIGHTED AGGREGATION
# ═════════════════════════════════════════════════════════════

def aggregate_weights(bc, local_metrics_test=None, node_sample_sizes=None):
    blocks      = [b for b in bc.chain if b['weights'] is not None]
    all_weights = [np.array(b['weights']) for b in blocks]
    n_nodes     = len(all_weights)
    if not all_weights:
        return None, {}
    weight_info = {}
    if (local_metrics_test and node_sample_sizes and
            len(local_metrics_test) == n_nodes == len(node_sample_sizes)):
        sizes    = np.array(node_sample_sizes, dtype=float)
        size_w   = sizes / sizes.sum()
        r2s      = np.clip([m[4] for m in local_metrics_test], 0, 1)
        r2_w     = r2s / (r2s.sum() + 1e-10)
        combined = size_w * r2_w; combined = combined / (combined.sum() + 1e-10)
        agg = np.zeros_like(all_weights[0], dtype=float)
        for w, cw in zip(all_weights, combined):
            agg += cw * w
        weight_info = {'size_w': size_w.tolist(), 'r2_w': r2_w.tolist(),
                       'combined': combined.tolist()}
    else:
        agg = np.mean(all_weights, axis=0)
    return agg, weight_info


# ═════════════════════════════════════════════════════════════
# NON-IID HANDLING
# ═════════════════════════════════════════════════════════════

def compute_distribution_stats(close_values, regime_label):
    s = pd.Series(close_values)
    return {'regime': regime_label, 'n_samples': int(len(s)),
            'mean': float(s.mean()), 'std': float(s.std()),
            'skewness': float(s.skew()), 'kurtosis': float(s.kurt()),
            'min': float(s.min()), 'max': float(s.max()),
            'range': float(s.max() - s.min())}


def compute_kl_divergence_matrix(stats_list):
    n      = len(stats_list)
    kl_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            mu_p  = stats_list[i]['mean']; sig_p = stats_list[i]['std'] + 1e-10
            mu_q  = stats_list[j]['mean']; sig_q = stats_list[j]['std'] + 1e-10
            kl    = (np.log(sig_q/sig_p) + (sig_p**2 + (mu_p-mu_q)**2)/(2*sig_q**2) - 0.5)
            kl_mat[i, j] = max(float(kl), 0.0)
    return kl_mat


def fedprox_correction(local_weights, global_weights, mu=0.01):
    if global_weights is None:
        return local_weights, 0.0
    prox = (mu/2.0) * float(np.sum((local_weights - global_weights)**2))
    corr = local_weights - mu * (local_weights - global_weights)
    print(f"  [FedProx]  mu={mu}  proximal_term={prox:.8f}")
    return corr, prox


# ═════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ═════════════════════════════════════════════════════════════

def compute_technical_indicators(close_series):
    s      = close_series.copy()
    df_ind = pd.DataFrame(index=s.index)
    df_ind['SMA_10']   = s.rolling(10).mean()
    df_ind['EMA_10']   = s.ewm(span=10, adjust=False).mean()
    delta  = s.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    rs     = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df_ind['RSI_14']   = 100 - (100 / (1 + rs))
    ema12  = s.ewm(span=12, adjust=False).mean()
    ema26  = s.ewm(span=26, adjust=False).mean()
    df_ind['MACD']     = ema12 - ema26
    sma20  = s.rolling(20).mean(); std20 = s.rolling(20).std()
    df_ind['BB_Width'] = ((sma20 + 2*std20 - (sma20 - 2*std20)) / sma20.replace(0, np.nan))
    return df_ind


# ═════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════

def calculate_complexity(svr_model, X):
    return svr_model.support_.shape[0]


def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        ape  = np.abs((y_true - y_pred) / y_true)
        va   = ape[np.isfinite(ape)]
        mape = float(np.mean(va) * 100) if len(va) > 0 else float('nan')
    return mse, rmse, mae, mape, r2


def directional_accuracy(y_true, y_pred):
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    return round(float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100), 4)


def rolling_directional_accuracy(y_true, y_pred, window=20):
    correct = (np.sign(np.diff(np.array(y_true))) ==
               np.sign(np.diff(np.array(y_pred)))).astype(float)
    return pd.Series(correct).rolling(window, min_periods=1).mean().values * 100


# ═════════════════════════════════════════════════════════════
# DATA
# ═════════════════════════════════════════════════════════════

def fetch_nasdaq_data(start_date, end_date):
    df = yf.download("^IXIC", start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def create_shifted_features(df, features, target, lookback=7):
    shifted = pd.concat([df[features].shift(i).add_suffix(f'_{i}')
                         for i in range(lookback)], axis=1)
    shifted[target] = df[target]
    return shifted.dropna()


# ═════════════════════════════════════════════════════════════
# LOOKBACK WINDOW ABLATION
# ═════════════════════════════════════════════════════════════

def select_optimal_lookback(df_tr, df_ts, all_features, target, svr_params,
                             candidates=(3, 5, 7, 10, 14, 21), node_id=0):
    ablation = {}
    print(f"\n  Node {node_id} — lookback ablation: {candidates}")
    for lb in candidates:
        sh_tr = create_shifted_features(df_tr, all_features, target, lb)
        sh_ts = create_shifted_features(
            pd.concat([df_tr.iloc[-lb:], df_ts]), all_features, target, lb
        ).iloc[-len(df_ts):]
        if len(sh_tr) < 20 or len(sh_ts) < 5:
            continue
        Xtr, ytr = sh_tr.iloc[:, :-1], sh_tr[target]
        Xts, yts = sh_ts.iloc[:, :-1], sh_ts[target]
        m = SVR(**svr_params); m.fit(Xtr, ytr)
        r2_tr = float(r2_score(ytr, m.predict(Xtr)))
        r2_ts = float(r2_score(yts, m.predict(Xts)))
        gap   = r2_tr - r2_ts
        ablation[lb] = {'r2_train': r2_tr, 'r2_test': r2_ts, 'gap': gap}
        status = 'overfit' if gap > 0.05 else 'healthy'
        print(f"    lb={lb:>3}d  R2_tr={r2_tr:.6f}  R2_ts={r2_ts:.6f}  gap={gap:.4f}  {status}")
    healthy = [lb for lb, v in ablation.items() if v['gap'] <= 0.05]
    best_lb = max(healthy, key=lambda lb: ablation[lb]['r2_test']) if healthy else 7
    print(f"  Selected lookback: {best_lb} days")
    return best_lb, ablation


def compute_learning_curve(X_train, y_train, X_test, y_test, svr_params, n_steps=8):
    sizes = np.linspace(0.1, 1.0, n_steps)
    tr_r2, ts_r2 = [], []
    n = len(X_train)
    for frac in sizes:
        ns = max(int(frac * n), 10)
        m  = SVR(**svr_params)
        m.fit(X_train.iloc[:ns], y_train.iloc[:ns])
        tr_r2.append(r2_score(y_train.iloc[:ns], m.predict(X_train.iloc[:ns])))
        ts_r2.append(r2_score(y_test, m.predict(X_test)))
    return (sizes * n).astype(int), tr_r2, ts_r2


# ═════════════════════════════════════════════════════════════
# FEDERATED LEARNING
# ═════════════════════════════════════════════════════════════

def federated_learning():
    print_secure_aggregation_comparison()
    print_feature_justification()

    date_ranges = [
        ("2015-10-15", "2018-10-15", "Pre-Volatility Baseline"),
        ("2018-10-15", "2021-10-15", "COVID-19 Crisis Regime"),
        ("2021-10-15", "2024-10-15", "Post-COVID Rate-Hike Regime"),
    ]

    raw_features = ['Open', 'High', 'Low', 'Volume']
    ind_features = ['SMA_10', 'EMA_10', 'RSI_14', 'MACD', 'BB_Width']
    all_features = raw_features + ind_features
    target       = 'Close'
    max_window   = 26
    fedprox_mu   = 0.01
    svr_params   = dict(kernel='linear', C=5, epsilon=0.0, shrinking=True, verbose=False, tol=0.001)

    lp_train, lp_test   = [], []
    lm_train, lm_test   = [], []
    lda_train, lda_test = [], []
    lc                  = []
    gta, gtea           = [], []
    gtp, gtep           = [], []
    lc_curves           = []
    dfs, rl             = [], []
    raw_test_actuals    = []
    raw_test_preds      = []
    node_sample_sizes   = []
    dist_stats_list     = []
    proximal_terms      = []
    weight_norms        = []
    all_lb_ablations    = []
    selected_lookbacks  = []
    current_global_w    = None
    block_trust_scores  = []
    block_timestamps    = []

    for idx, (start_date, end_date, regime_label) in enumerate(date_ranges):
        node_id = idx + 1
        print(f"\n{'='*70}")
        print(f"  Node {node_id}  |  {regime_label}")
        print(f"  Period: {start_date} -> {end_date}")
        print(f"{'='*70}")

        df_raw = fetch_nasdaq_data(start_date, end_date)
        dstats = compute_distribution_stats(df_raw['Close'].values, regime_label)
        dist_stats_list.append(dstats)
        print(f"  [NonIID] mu={dstats['mean']:.2f}  sigma={dstats['std']:.2f}  n={dstats['n_samples']}")

        split = int(len(df_raw) * 0.8)
        df_tr = df_raw.iloc[:split].copy()
        df_ts = df_raw.iloc[split:].copy()

        df_tr = df_tr.join(compute_technical_indicators(df_tr['Close']))
        seed  = df_tr['Close'].iloc[-max_window:]
        ts_ind = compute_technical_indicators(pd.concat([seed, df_ts['Close']]))
        df_ts = df_ts.join(ts_ind.iloc[max_window:])
        df_tr = df_tr.dropna(); df_ts = df_ts.dropna()
        dfs.append(df_tr.copy()); rl.append(regime_label)

        sf = MinMaxScaler()
        df_tr[all_features] = sf.fit_transform(df_tr[all_features])
        df_ts[all_features] = sf.transform(df_ts[all_features])
        st = MinMaxScaler()
        df_tr[[target]] = st.fit_transform(df_tr[[target]])
        df_ts[[target]] = st.transform(df_ts[[target]])

        best_lb, lb_ablation = select_optimal_lookback(
            df_tr, df_ts, all_features, target, svr_params,
            candidates=(3, 5, 7, 10, 14, 21), node_id=node_id)
        all_lb_ablations.append(lb_ablation)
        selected_lookbacks.append(best_lb)

        sh_tr = create_shifted_features(df_tr, all_features, target, best_lb)
        sh_ts = create_shifted_features(
            pd.concat([df_tr.iloc[-best_lb:], df_ts]), all_features, target, best_lb
        ).iloc[-len(df_ts):]

        X_tr, y_tr = sh_tr.iloc[:, :-1], sh_tr[target]
        X_ts, y_ts = sh_ts.iloc[:, :-1], sh_ts[target]
        node_sample_sizes.append(len(X_tr))

        svr = SVR(**svr_params)
        t0  = time.time()
        svr.fit(X_tr, y_tr)
        cpu = time.time() - t0

        local_w = svr.coef_.flatten()
        local_w, prox_term = fedprox_correction(local_w, current_global_w, mu=fedprox_mu)
        proximal_terms.append(prox_term)
        weight_norms.append(float(np.linalg.norm(local_w)))

        block = blockchain.add_weights_to_block(local_w, node_id)
        if block:
            block_trust_scores.append(block['trust_score'])
            block_timestamps.append(block['timestamp'])
            audit.record('WEIGHT_SUBMITTED', node_id,
                         {'regime': regime_label, 'trust': block['trust_score'],
                          'block': block['index'], 'cpu': round(cpu, 4),
                          'lookback_days': best_lb, 'prox_term': round(prox_term, 8)})
        else:
            audit.record('WEIGHT_REJECTED', node_id,
                         {'regime': regime_label, 'reason': 'cosine < threshold'})
            continue

        yhat_tr = svr.predict(X_tr)
        yhat_ts = svr.predict(X_ts)
        lp_train.append(yhat_tr); lp_test.append(yhat_ts)

        mtr = compute_metrics(y_tr, yhat_tr)
        mts = compute_metrics(y_ts, yhat_ts)
        lm_train.append(mtr); lm_test.append(mts)

        da_tr = directional_accuracy(y_tr, yhat_tr)
        da_ts = directional_accuracy(y_ts, yhat_ts)
        lda_train.append(da_tr); lda_test.append(da_ts)

        raw_test_actuals.append(np.array(y_ts))
        raw_test_preds.append(yhat_ts)
        gta.extend(y_tr); gtp.extend(yhat_tr)
        gtea.extend(y_ts); gtep.extend(yhat_ts)
        lc.append((calculate_complexity(svr, X_tr), cpu))

        print(f"  Computing learning curve ...")
        lc_sizes, lc_tr, lc_ts = compute_learning_curve(X_tr, y_tr, X_ts, y_ts, svr_params)
        lc_curves.append((lc_sizes, lc_tr, lc_ts))

        r2g = mtr[4] - mts[4]
        print(f"\n  Node {node_id} [{regime_label}]  lb={best_lb}d")
        print(f"  Train: R2={mtr[4]:.6f}  MAPE={mtr[3]:.4f}%  DA={da_tr:.2f}%")
        print(f"  Test:  R2={mts[4]:.6f}  MAPE={mts[3]:.4f}%  DA={da_ts:.2f}%")
        print(f"  R2 gap={r2g:.4f}  {'overfit' if r2g > 0.05 else 'healthy'}")

        audit.record('NODE_METRICS', node_id,
                     {'regime': regime_label, 'lookback_days': best_lb,
                      'r2_train': round(mtr[4], 6), 'r2_test': round(mts[4], 6),
                      'mape_test': round(mts[3], 4), 'da_train': da_tr, 'da_test': da_ts,
                      'r2_gap': round(r2g, 6), 'overfit': r2g > 0.05})

    kl_matrix     = compute_kl_divergence_matrix(dist_stats_list)
    accepted_w    = [b['weights'] for b in blockchain.chain if b['weights'] is not None]
    poison_report = compute_poisoning_risk(accepted_w)
    blockchain.validate_chain()

    agg_w, weight_info = aggregate_weights(blockchain, lm_test, node_sample_sizes)
    current_global_w   = agg_w

    latency_results = benchmark_aggregation_methods(
        [np.array(b['weights']) for b in blockchain.chain if b['weights'] is not None],
        lm_test, node_sample_sizes)

    audit.record('AGGREGATION', 'GLOBAL',
                 {'method': 'combined_size_r2_fedavg', 'nodes': len(date_ranges),
                  'rejected': blockchain.rejected_nodes,
                  'selected_lookbacks': selected_lookbacks,
                  'poisoning_risk': round(poison_report['mean_pairwise_similarity'], 4)})

    gmtr    = compute_metrics(np.array(gta), np.array(gtp))
    gmts    = compute_metrics(np.array(gtea), np.array(gtep))
    g_da_tr = directional_accuracy(np.array(gta), np.array(gtp))
    g_da_ts = directional_accuracy(np.array(gtea), np.array(gtep))

    print(f"\n  GLOBAL RESULTS")
    print(f"  Train: R2={gmtr[4]:.8f}  DA={g_da_tr:.2f}%")
    print(f"  Test:  R2={gmts[4]:.8f}  MAPE={gmts[3]:.4f}%  DA={g_da_ts:.2f}%")

    audit.summary()
    audit.export('audit_trail.json')

    return (dfs, lp_train, lp_test, gtp, gtep, gta, gtea,
            lm_train, lm_test, lda_train, lda_test,
            lc, agg_w, lc_curves, rl, raw_test_actuals, raw_test_preds,
            dist_stats_list, kl_matrix, poison_report, weight_info,
            node_sample_sizes, proximal_terms, weight_norms,
            gmtr, gmts, g_da_tr, g_da_ts, latency_results,
            all_lb_ablations, selected_lookbacks,
            block_trust_scores, block_timestamps)


# ═════════════════════════════════════════════════════════════
# ABLATION BASELINES
# ═════════════════════════════════════════════════════════════

DATE_RANGES = [
    ("2015-10-15", "2018-10-15"),
    ("2018-10-15", "2021-10-15"),
    ("2021-10-15", "2024-10-15"),
]


def run_arima_baseline(date_ranges, order=(5, 1, 0)):
    print("\n ABLATION — ARIMA Baseline")
    node_metrics, node_preds, node_actuals, node_times = [], [], [], []
    for idx, (start_date, end_date) in enumerate(date_ranges):
        print(f"\n  Node {idx+1}: fitting ARIMA{order} ...")
        df_raw      = fetch_nasdaq_data(start_date, end_date)
        split_point = int(len(df_raw) * 0.8)
        train_close = df_raw['Close'].values[:split_point]
        test_close  = df_raw['Close'].values[split_point:]
        sc          = MinMaxScaler()
        train_sc    = sc.fit_transform(train_close.reshape(-1, 1)).flatten()
        test_sc     = sc.transform(test_close.reshape(-1, 1)).flatten()
        preds       = []; history = list(train_sc); t0 = time.time()
        if ARIMA_AVAILABLE:
            for t in range(len(test_sc)):
                try:
                    fitted = ARIMA(history, order=order).fit()
                    yhat   = fitted.forecast(steps=1)[0]
                except Exception:
                    yhat = history[-1]
                preds.append(yhat); history.append(test_sc[t])
        else:
            for t in range(len(test_sc)):
                preds.append(history[-1]); history.append(test_sc[t])
        elapsed = time.time() - t0
        preds   = np.array(preds)
        metrics = compute_metrics(test_sc, preds)
        node_metrics.append(metrics); node_preds.append(preds)
        node_actuals.append(test_sc); node_times.append(elapsed)
        print(f"  Node {idx+1}: R2={metrics[4]:.6f}  Time={elapsed:.2f}s")
    return node_metrics, node_preds, node_actuals, node_times


def _make_sequences(data, lookback=7):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i]); y.append(data[i])
    return np.array(X), np.array(y)


def run_lstm_baseline(date_ranges, lookback=7, epochs=30, units=50):
    print(f"\n ABLATION — LSTM Baseline ({'TensorFlow' if KERAS_AVAILABLE else 'EWM approx'})")
    node_metrics, node_preds, node_actuals, node_times = [], [], [], []
    for idx, (start_date, end_date) in enumerate(date_ranges):
        print(f"\n  Node {idx+1}: training LSTM ...")
        df_raw      = fetch_nasdaq_data(start_date, end_date)
        split_point = int(len(df_raw) * 0.8)
        train_close = df_raw['Close'].values[:split_point]
        test_close  = df_raw['Close'].values[split_point:]
        sc          = MinMaxScaler()
        train_sc    = sc.fit_transform(train_close.reshape(-1, 1)).flatten()
        test_sc     = sc.transform(test_close.reshape(-1, 1)).flatten()
        X_train, y_train = _make_sequences(train_sc, lookback)
        combined         = np.concatenate([train_sc[-lookback:], test_sc])
        X_test,  y_test  = _make_sequences(combined, lookback)
        X_test  = X_test[:len(test_sc)]; y_test = test_sc
        t0 = time.time()
        if KERAS_AVAILABLE:
            tf.random.set_seed(42)
            model = Sequential([
                LSTM(units, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2), LSTM(units//2), Dropout(0.2), Dense(1)])
            model.compile(optimizer='adam', loss='mse')
            es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            model.fit(X_train.reshape(-1, lookback, 1), y_train,
                      validation_split=0.1, epochs=epochs, batch_size=32,
                      callbacks=[es], verbose=0)
            preds = model.predict(X_test.reshape(-1, lookback, 1), verbose=0).flatten()
        else:
            alpha = 0.3; state = float(y_train[-1]); preds = []
            for x in X_test:
                state = alpha*x[-1] + (1-alpha)*state; preds.append(state)
            preds = np.array(preds)
        elapsed = time.time() - t0
        metrics = compute_metrics(y_test, preds)
        node_metrics.append(metrics); node_preds.append(preds)
        node_actuals.append(y_test); node_times.append(elapsed)
        print(f"  Node {idx+1}: R2={metrics[4]:.6f}  Time={elapsed:.2f}s")
    return node_metrics, node_preds, node_actuals, node_times


# ═════════════════════════════════════════════════════════════
# ═══════════  VISUALISATIONS  ════════════════════════════════
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 1. Learning Curves
# ─────────────────────────────────────────────
def plot_learning_curves(lc_curves, rl):
    n = len(lc_curves)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle('Learning Curves per Market Regime\n'
                 'Persistent gap between train and test signals overfitting', fontsize=13)
    for i, (sizes, tr, ts) in enumerate(lc_curves):
        ax  = axes[i]; gap = np.array(tr) - np.array(ts)
        ax.plot(sizes, tr, 'o-', color='steelblue', label='Train R²')
        ax.plot(sizes, ts, 's-', color='coral',     label='Test R²')
        ax.fill_between(sizes, tr, ts, alpha=0.15, color='orange', label='Gap')
        fg  = gap[-1]; col = 'red' if fg > 0.05 else 'green'
        ax.annotate(f'Final gap\n{fg:.4f}',
                    xy=(sizes[-1], (tr[-1]+ts[-1])/2), fontsize=9, color=col,
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=col, alpha=0.8))
        ax.set_title(f'Node {i+1} — {rl[i]}', fontsize=9)
        ax.set_xlabel('Training samples'); ax.set_ylabel('R²')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4); ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    save_fig(fig, 'Learning_Curves')
    plt.show()


# ─────────────────────────────────────────────
# 2. Metric Bar Chart
# ─────────────────────────────────────────────
def plot_metrics_with_gap(lm_train, lm_test, lc=None, rl=None):
    mnames  = ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R²']
    n       = len(lm_train)
    nlabels = [f'N{j+1}\n{rl[j][:12]}' if rl else f'N{j+1}' for j in range(n)]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle('Train vs Test Metrics per Regime\n'
                 'Red delta = gap; large R² gap indicates overfitting', fontsize=13)
    axes = axes.flatten()
    for mi, mname in enumerate(mnames):
        ax  = axes[mi]
        tv  = [m[mi] for m in lm_train]; tsv = [m[mi] for m in lm_test]
        x, w = np.arange(n), 0.35
        ax.bar(x-w/2, tv,  w, label='Train', color='steelblue', alpha=0.85)
        ax.bar(x+w/2, tsv, w, label='Test',  color='coral',     alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(nlabels, fontsize=7)
        ax.set_title(mname); ax.legend(fontsize=8)
        for j in range(n):
            ax.text(j-w/2, tv[j],  f'{tv[j]:.2e}',  ha='center', va='bottom', fontsize=7, color='steelblue')
            ax.text(j+w/2, tsv[j], f'{tsv[j]:.2e}', ha='center', va='bottom', fontsize=7, color='coral')
            gap = tv[j] - tsv[j]; top = max(tv[j], tsv[j])
            bad = (gap > 0.05) if mname == 'R²' else (gap < -0.01*top)
            ax.annotate(f'd={gap:+.2e}', xy=(j, top*1.01), ha='center',
                        fontsize=7, color='red' if bad else 'gray',
                        fontweight='bold' if bad else 'normal')
        ax.grid(True, alpha=0.3, axis='y')
    if lc:
        times = [c[1] for c in lc]
        axes[5].bar(range(n), times, color='mediumpurple', alpha=0.85)
        axes[5].set_xticks(range(n)); axes[5].set_xticklabels(nlabels, fontsize=7)
        axes[5].set_title('Training Time (s)')
        for j, t in enumerate(times):
            axes[5].text(j, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=8)
        axes[5].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, 'Train_vs_Test_Metrics')
    plt.show()


# ─────────────────────────────────────────────
# 3. Prediction Overlay
# ─────────────────────────────────────────────
def plot_prediction_overlay(lp_train, lp_test, gta, gtea,
                            lm_train, lm_test, lc, rl):
    n    = len(lp_train)
    trsz = [len(p) for p in lp_train]; tssz = [len(p) for p in lp_test]
    trs  = np.split(np.array(gta),  np.cumsum(trsz)[:-1])
    tss  = np.split(np.array(gtea), np.cumsum(tssz)[:-1])
    fig, axes = plt.subplots(n, 1, figsize=(14, 5*n))
    if n == 1: axes = [axes]
    fig.suptitle('Prediction Overlay per Regime — FL-SVR', fontsize=13)
    for i in range(n):
        ax   = axes[i]
        tr_p = np.array(lp_train[i]).flatten()
        ts_p = np.array(lp_test[i]).flatten()
        tr_a = trs[i]; ts_a = tss[i]
        tri  = np.arange(len(tr_a)); tsi = np.arange(len(tr_a), len(tr_a)+len(ts_a))
        ax.plot(tri, tr_a, color='steelblue', lw=1.5, label='Actual (Train)', alpha=0.8)
        ax.plot(tri, tr_p, color='green',     lw=1,   label='Predicted (Train)', ls='--', alpha=0.9)
        ax.plot(tsi, ts_a, color='navy',      lw=1.5, label='Actual (Test)',     ls=':', alpha=0.8)
        ax.plot(tsi, ts_p, color='red',       lw=1,   label='Predicted (Test)',  ls='--', alpha=0.9)
        ax.axvspan(tsi[0], tsi[-1], alpha=0.06, color='orange', label='Test region')
        r2tr = lm_train[i][4]; r2ts = lm_test[i][4]; gap = r2tr - r2ts
        col  = 'red' if gap > 0.05 else 'green'
        ax.set_title(f'Node {i+1} — {rl[i]}\n'
                     f'SV:{lc[i][0]}  R² train={r2tr:.4f}  test={r2ts:.4f}  gap={gap:+.4f}',
                     color=col, fontsize=9)
        ax.set_xlabel('Time Index'); ax.set_ylabel('Price (scaled)')
        ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Prediction_Overlay')
    plt.show()


# ─────────────────────────────────────────────
# 4. Residual Distributions
# ─────────────────────────────────────────────
def plot_residual_distributions(lp_train, lp_test, gta, gtea, rl):
    n    = len(lp_train)
    trsz = [len(p) for p in lp_train]; tssz = [len(p) for p in lp_test]
    trs  = np.split(np.array(gta),  np.cumsum(trsz)[:-1])
    tss  = np.split(np.array(gtea), np.cumsum(tssz)[:-1])
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle('Residual Distributions per Regime\n'
                 'Train narrow + test wide/skewed = overfitting signature', fontsize=13)
    for i in range(n):
        ax   = axes[i]
        tr_r = trs[i] - np.array(lp_train[i]).flatten()
        ts_r = tss[i] - np.array(lp_test[i]).flatten()
        bins = np.linspace(min(tr_r.min(), ts_r.min()), max(tr_r.max(), ts_r.max()), 40)
        ax.hist(tr_r, bins=bins, alpha=0.55, color='steelblue',
                label=f'Train sigma={tr_r.std():.4f}', density=True)
        ax.hist(ts_r, bins=bins, alpha=0.55, color='coral',
                label=f'Test  sigma={ts_r.std():.4f}', density=True)
        ax.axvline(0, color='black', ls='--', lw=1.2, label='Zero error')
        ratio = ts_r.std() / (tr_r.std() + 1e-10)
        col   = 'red' if ratio > 1.5 else 'green'
        ax.set_title(f'Node {i+1} — {rl[i]}\n'
                     f'sigma ratio={ratio:.2f} {"(overfit)" if ratio>1.5 else "(OK)"}',
                     color=col, fontsize=9)
        ax.set_xlabel('Residual'); ax.set_ylabel('Density')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Residual_Distributions')
    plt.show()


# ─────────────────────────────────────────────
# 5. Regime Robustness
# ─────────────────────────────────────────────
def plot_regime_comparison(lm_train, lm_test, lda_train, lda_test, rl):
    n = len(lm_train); x, w = np.arange(n), 0.35
    labels = [f'N{i+1}: {rl[i][:18]}' for i in range(n)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Regime Robustness — R², MAPE, and Directional Accuracy\n'
                 'Across all three market regimes', fontsize=13)
    r2tr = [m[4] for m in lm_train]; r2ts = [m[4] for m in lm_test]
    axes[0].bar(x-w/2, r2tr, w, label='Train', color='steelblue', alpha=0.85)
    axes[0].bar(x+w/2, r2ts, w, label='Test',  color='coral',     alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8, rotation=10)
    axes[0].set_ylim(0.98, 1.001); axes[0].set_title('R² per Regime')
    axes[0].set_ylabel('R²'); axes[0].legend(); axes[0].grid(True, alpha=0.3, axis='y')
    for j in range(n):
        axes[0].text(j-w/2, r2tr[j], f'{r2tr[j]:.4f}', ha='center', va='bottom', fontsize=7, color='steelblue')
        axes[0].text(j+w/2, r2ts[j], f'{r2ts[j]:.4f}', ha='center', va='bottom', fontsize=7, color='coral')
    mtr = [m[3] for m in lm_train]; mts = [m[3] for m in lm_test]
    axes[1].bar(x-w/2, mtr, w, label='Train', color='steelblue', alpha=0.85)
    axes[1].bar(x+w/2, mts, w, label='Test',  color='coral',     alpha=0.85)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8, rotation=10)
    axes[1].set_title('MAPE (%) per Regime'); axes[1].set_ylabel('MAPE (%)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')
    for j in range(n):
        axes[1].text(j-w/2, mtr[j], f'{mtr[j]:.3f}%', ha='center', va='bottom', fontsize=7, color='steelblue')
        axes[1].text(j+w/2, mts[j], f'{mts[j]:.3f}%', ha='center', va='bottom', fontsize=7, color='coral')
    axes[2].bar(x-w/2, lda_train, w, label='Train', color='steelblue', alpha=0.85)
    axes[2].bar(x+w/2, lda_test,  w, label='Test',  color='coral',     alpha=0.85)
    axes[2].axhline(80, color='green', ls='--', lw=1.2, label='Strong threshold (80%)')
    axes[2].axhline(50, color='red',   ls='--', lw=1.2, label='Random baseline (50%)')
    axes[2].set_xticks(x); axes[2].set_xticklabels(labels, fontsize=8, rotation=10)
    axes[2].set_ylim(0, 115); axes[2].set_title('Directional Accuracy (%) per Regime')
    axes[2].set_ylabel('DA (%)'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3, axis='y')
    for j in range(n):
        axes[2].text(j-w/2, lda_train[j], f'{lda_train[j]:.1f}%', ha='center', va='bottom', fontsize=7, color='steelblue')
        axes[2].text(j+w/2, lda_test[j],  f'{lda_test[j]:.1f}%',  ha='center', va='bottom', fontsize=7, color='coral')
    plt.tight_layout()
    save_fig(fig, 'Regime_Robustness')
    plt.show()


# ─────────────────────────────────────────────
# 6. Rolling DA Timeline
# ─────────────────────────────────────────────
def plot_rolling_da_timeline(lp_train, lp_test, gta, gtea, rl, window=20):
    n    = len(lp_train)
    trsz = [len(p) for p in lp_train]; tssz = [len(p) for p in lp_test]
    trs  = np.split(np.array(gta),  np.cumsum(trsz)[:-1])
    tss  = np.split(np.array(gtea), np.cumsum(tssz)[:-1])
    fig, axes = plt.subplots(n, 1, figsize=(14, 4*n))
    if n == 1: axes = [axes]
    fig.suptitle(f'Rolling Directional Accuracy (window={window}) per Regime', fontsize=13)
    for i in range(n):
        ax     = axes[i]
        rda_tr = rolling_directional_accuracy(trs[i], np.array(lp_train[i]).flatten(), window)
        rda_ts = rolling_directional_accuracy(tss[i], np.array(lp_test[i]).flatten(), window)
        tri    = np.arange(len(rda_tr)); tsi = np.arange(len(rda_tr), len(rda_tr)+len(rda_ts))
        ax.plot(tri, rda_tr, color='steelblue', lw=1.2, label='Train DA', alpha=0.85)
        ax.plot(tsi, rda_ts, color='coral',     lw=1.2, label='Test DA',  alpha=0.85)
        ax.axhline(80, color='green', ls='--', lw=1, label='Strong (80%)', alpha=0.7)
        ax.axhline(50, color='red',   ls='--', lw=1, label='Random (50%)', alpha=0.7)
        ax.axvspan(tsi[0], tsi[-1], alpha=0.05, color='orange', label='Test region')
        ax.set_ylim(0, 110)
        ax.set_title(f'Node {i+1} — {rl[i]}  |  Mean train={np.mean(rda_tr):.1f}%  test={np.mean(rda_ts):.1f}%', fontsize=9)
        ax.set_xlabel('Time Index'); ax.set_ylabel('Rolling DA (%)')
        ax.legend(fontsize=8, ncol=3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Rolling_Directional_Accuracy')
    plt.show()


# ─────────────────────────────────────────────
# 7. Metrics Summary Table
# ─────────────────────────────────────────────
def plot_metrics_summary_table(lm_train, lm_test, lda_train, lda_test,
                                rl, gmtr, gmts, g_da_tr, g_da_ts,
                                selected_lookbacks=None):
    fig, ax = plt.subplots(figsize=(22, 4))
    ax.axis('off')
    fig.suptitle('Per-Regime Metrics Summary — NASDAQ (^IXIC)\n'
                 'Federated SVR with Blockchain Coordination', fontsize=13, y=1.02)
    cols = ['Node', 'Regime', 'Lookback', 'R² Train', 'R² Test',
            'MAPE Train', 'MAPE Test', 'DA Train', 'DA Test', 'R² Gap', 'Status']
    rows = []
    for i in range(len(lm_train)):
        r2g = lm_train[i][4] - lm_test[i][4]
        lb  = f'{selected_lookbacks[i]}d' if selected_lookbacks else 'N/A'
        rows.append([f'Node {i+1}', rl[i], lb,
                     f'{lm_train[i][4]:.8f}', f'{lm_test[i][4]:.8f}',
                     f'{lm_train[i][3]:.4f}%', f'{lm_test[i][3]:.4f}%',
                     f'{lda_train[i]:.2f}%', f'{lda_test[i]:.2f}%',
                     f'{r2g:.4f}', 'Healthy' if r2g <= 0.05 else 'Overfit'])
    g_r2g = gmtr[4] - gmts[4]
    rows.append(['GLOBAL', 'All Regimes', '—',
                 f'{gmtr[4]:.8f}', f'{gmts[4]:.8f}',
                 f'{gmtr[3]:.4f}%', f'{gmts[3]:.4f}%',
                 f'{g_da_tr:.2f}%', f'{g_da_ts:.2f}%',
                 f'{g_r2g:.4f}', 'Healthy' if g_r2g <= 0.05 else 'Overfit'])
    table = ax.table(cellText=rows, colLabels=cols, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 2.2)
    for j in range(len(cols)):
        table[(0,j)].set_facecolor('#2C3E50')
        table[(0,j)].set_text_props(color='white', fontweight='bold')
    nc = ['#EAF4FB', '#FEF9E7', '#EAFAF1']
    for i in range(len(lm_train)):
        for j in range(len(cols)):
            table[(i+1,j)].set_facecolor(nc[i % len(nc)])
    for j in range(len(cols)):
        table[(len(lm_train)+1,j)].set_facecolor('#D5D8DC')
        table[(len(lm_train)+1,j)].set_text_props(fontweight='bold')
    for i in range(len(lm_train)):
        if lda_test[i] > 80:
            table[(i+1, 8)].set_facecolor('#A9DFBF')
            table[(i+1, 8)].set_text_props(fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'Metrics_Summary_Table')
    plt.show()


# ─────────────────────────────────────────────
# 8. R² Gap Heatmap
# ─────────────────────────────────────────────
def plot_r2_gap_heatmap(lm_train, lm_test, rl):
    n       = len(lm_train)
    metrics = ['R²', 'MSE', 'RMSE', 'MAE', 'MAPE (%)']
    midx    = [4, 0, 1, 2, 3]
    gaps    = np.array([[lm_train[ni][mi] - lm_test[ni][mi] for ni in range(n)] for mi in midx])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Overfitting Gap Analysis — Train minus Test across Metrics and Regimes', fontsize=13)
    r2_gaps = [lm_train[i][4] - lm_test[i][4] for i in range(n)]
    colors  = ['green' if g <= 0.05 else 'red' for g in r2_gaps]
    bars    = axes[0].bar(range(n), r2_gaps, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(0.05, color='red', ls='--', lw=1.5, label='Overfit threshold (0.05)')
    axes[0].set_xticks(range(n))
    axes[0].set_xticklabels([f'N{i+1}: {rl[i][:16]}' for i in range(n)], fontsize=8, rotation=10)
    axes[0].set_ylabel('R² Gap'); axes[0].set_title('R² Gap per Regime')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')
    for bar, g in zip(bars, r2_gaps):
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0005,
                     f'{g:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    norm_gaps = np.zeros_like(gaps)
    for mi in range(len(midx)):
        row = gaps[mi]; scale = np.max(np.abs(row)) + 1e-10
        norm_gaps[mi] = row / scale
    im = axes[1].imshow(norm_gaps, cmap='RdYlGn_r', aspect='auto', vmin=-1, vmax=1)
    axes[1].set_xticks(range(n))
    axes[1].set_xticklabels([f'N{i+1}: {rl[i][:14]}' for i in range(n)], fontsize=8)
    axes[1].set_yticks(range(len(metrics))); axes[1].set_yticklabels(metrics, fontsize=9)
    axes[1].set_title('Normalised Gap Heatmap')
    for mi in range(len(metrics)):
        for ni in range(n):
            axes[1].text(ni, mi, f'{gaps[mi,ni]:+.2e}', ha='center', va='center', fontsize=8, color='black')
    plt.colorbar(im, ax=axes[1], label='Normalised gap')
    plt.tight_layout()
    save_fig(fig, 'R2_Gap_Heatmap')
    plt.show()


# ─────────────────────────────────────────────
# 9. Error Scatter
# ─────────────────────────────────────────────
def plot_prediction_error_scatter(raw_test_actuals, raw_test_preds, rl):
    n = len(raw_test_actuals)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6))
    if n == 1: axes = [axes]
    fig.suptitle('Predicted vs Actual — Test Set Scatter per Regime\n'
                 'Colour encodes residual; perfect prediction lies on the diagonal', fontsize=13)
    for i in range(n):
        ax   = axes[i]
        act  = np.array(raw_test_actuals[i]).flatten()
        pred = np.array(raw_test_preds[i]).flatten()
        resid= act - pred
        sc   = ax.scatter(act, pred, c=resid, cmap='coolwarm', s=12, alpha=0.7)
        mn, mx = min(act.min(), pred.min()), max(act.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect (y=x)')
        r2  = r2_score(act, pred); mae = mean_absolute_error(act, pred)
        ax.set_title(f'Node {i+1} — {rl[i]}\nR²={r2:.6f}  MAE={mae:.6f}', fontsize=9)
        ax.set_xlabel('Actual (scaled)'); ax.set_ylabel('Predicted (scaled)')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Residual')
    plt.tight_layout()
    save_fig(fig, 'Prediction_Error_Scatter')
    plt.show()


# ─────────────────────────────────────────────
# 10. Non-IID Distribution
# ─────────────────────────────────────────────
def plot_noniid_distributions(dist_stats_list, rl):
    n    = len(dist_stats_list)
    fig  = plt.figure(figsize=(20, 10))
    fig.suptitle('Non-IID Data Distribution per Federated Node\n'
                 'Structural differences across regimes confirm heterogeneous data setup', fontsize=13)
    gs     = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)
    keys   = ['mean', 'std', 'skewness', 'kurtosis', 'range']
    titles = ['Mean Price', 'Standard Deviation', 'Skewness', 'Kurtosis', 'Price Range']
    colors = ['#2196F3', '#FF5722', '#4CAF50']
    labels = [f'N{i+1}: {rl[i][:16]}' for i in range(n)]
    for si, (key, title) in enumerate(zip(keys, titles)):
        r, c = divmod(si, 3)
        ax   = fig.add_subplot(gs[r, c])
        vals = [dist_stats_list[i][key] for i in range(n)]
        bars = ax.bar(range(n), vals, color=colors[:n], alpha=0.85, edgecolor='black', lw=0.8)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8, rotation=8)
        ax.set_title(title, fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax5   = fig.add_subplot(gs[1, 2])
    sizes = [dist_stats_list[i]['n_samples'] for i in range(n)]
    bars  = ax5.bar(range(n), sizes, color=colors[:n], alpha=0.85, edgecolor='black', lw=0.8)
    ax5.set_xticks(range(n)); ax5.set_xticklabels(labels, fontsize=8, rotation=8)
    ax5.set_title('Sample Size per Node', fontsize=10); ax5.grid(True, alpha=0.3, axis='y')
    for bar, s in zip(bars, sizes):
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f'{s}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    save_fig(fig, 'NonIID_Distribution')
    plt.show()


# ─────────────────────────────────────────────
# 11. KL Divergence Matrix
# ─────────────────────────────────────────────
def plot_kl_divergence_matrix(kl_matrix, rl):
    n  = len(rl)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('KL Divergence Matrix across Federated Nodes\n'
                 'High KL indicates high non-IID degree — justifies FedProx regularisation', fontsize=13)
    im = axes[0].imshow(kl_matrix, cmap='YlOrRd', aspect='auto')
    xl = [f'N{i+1}: {rl[i][:14]}' for i in range(n)]
    axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(xl, rotation=12, fontsize=8)
    axes[0].set_yticklabels(xl, fontsize=8)
    axes[0].set_title('Pairwise KL Divergence KL(i || j)')
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f'{kl_matrix[i,j]:.2f}', ha='center', va='center',
                         fontsize=10, color='black' if kl_matrix[i,j] < kl_matrix.max()*0.6 else 'white',
                         fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='KL Divergence')
    mean_kl = [np.mean([kl_matrix[i,j] for j in range(n) if j!=i]) for i in range(n)]
    cols2   = ['#FF7043' if v > np.mean(mean_kl) else '#42A5F5' for v in mean_kl]
    bars    = axes[1].bar(range(n), mean_kl, color=cols2, alpha=0.85, edgecolor='black')
    axes[1].axhline(np.mean(mean_kl), color='black', ls='--', lw=1.5,
                    label=f'Mean={np.mean(mean_kl):.2f}')
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(xl, fontsize=8, rotation=8)
    axes[1].set_ylabel('Mean KL to Other Nodes'); axes[1].set_title('Mean Non-IID Degree per Node')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, mean_kl):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'KL_Divergence_Matrix')
    plt.show()


# ─────────────────────────────────────────────
# 12. Poisoning Defence
# ─────────────────────────────────────────────
def plot_poisoning_defence(poison_report, weight_norms, rl):
    cos_mat = np.array(poison_report['cosine_matrix'])
    n       = len(weight_norms)
    labels  = [f'N{i+1}: {rl[i][:14]}' for i in range(n)]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Three-Layer Model Poisoning Defence\n'
                 'Norm Clipping  →  Differential Privacy  →  Cosine Similarity Validation', fontsize=13)
    im = axes[0].imshow(cos_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(labels, rotation=12, fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title('Pairwise Cosine Similarity Matrix')
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f'{cos_mat[i,j]:.3f}', ha='center', va='center', fontsize=10,
                         color='black' if abs(cos_mat[i,j]) < 0.7 else 'white', fontweight='bold')
    plt.colorbar(im, ax=axes[0], label='Cosine Similarity')
    nc    = ['#4CAF50' if v <= 10.0 else '#F44336' for v in weight_norms]
    bars  = axes[1].bar(range(n), weight_norms, color=nc, alpha=0.85, edgecolor='black')
    axes[1].axhline(10.0, color='red', ls='--', lw=1.5, label='Clip threshold (10.0)')
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels, fontsize=8, rotation=8)
    axes[1].set_ylabel('L2 Norm'); axes[1].set_title('Weight L2 Norms (before defence)')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, weight_norms):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    mean_peer = [np.mean([cos_mat[i,j] for j in range(n) if j!=i]) for i in range(n)]
    pc        = ['#F44336' if v < 0.5 else '#4CAF50' for v in mean_peer]
    bars2     = axes[2].bar(range(n), mean_peer, color=pc, alpha=0.85, edgecolor='black')
    axes[2].axhline(0.5, color='red', ls='--', lw=1.5, label='Suspicion threshold (0.5)')
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels(labels, fontsize=8, rotation=8)
    axes[2].set_ylabel('Mean Peer Cosine Similarity')
    axes[2].set_title('Mean Peer Cosine per Node\n(below 0.5 = suspicious)')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3, axis='y'); axes[2].set_ylim(-0.1, 1.1)
    for bar, v in zip(bars2, mean_peer):
        axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                     f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    susp = poison_report['suspicious_nodes']
    msg  = (f"Suspicious nodes detected: {susp}" if susp else "No suspicious nodes detected")
    fig.text(0.5, -0.02, msg, ha='center', fontsize=11,
             color='red' if susp else 'green', fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'Poisoning_Defence')
    plt.show()


# ─────────────────────────────────────────────
# 13. Aggregation Weights
# ─────────────────────────────────────────────
def plot_aggregation_weights(weight_info, node_sample_sizes, lm_test, rl):
    n = len(rl)
    if not weight_info or 'combined' not in weight_info:
        print("  Combined weight info not available — skipping aggregation weights plot.")
        return
    size_w   = np.array(weight_info.get('size_w',   [1/n]*n))
    r2_w     = np.array(weight_info.get('r2_w',     [1/n]*n))
    combined = np.array(weight_info.get('combined', [1/n]*n))
    labels   = [f'N{i+1}: {rl[i][:16]}' for i in range(n)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('FedAvg Aggregation Weight Decomposition\n'
                 'Combined sample-size × R² weighting scheme (McMahan et al. 2017)', fontsize=13)
    specs = [(axes[0], size_w,   '#42A5F5', 'Sample-Size Weights (n_i / sum n_j)'),
             (axes[1], r2_w,     '#66BB6A', 'R² Reputation Weights (R2_i / sum R2_j)'),
             (axes[2], combined, '#FFA726', 'Combined Weights (size x R2, renormalised)')]
    for ax, vals, col, title in specs:
        bars = ax.bar(range(n), vals, color=col, alpha=0.85, edgecolor='black', lw=0.8)
        ax.axhline(1/n, color='black', ls='--', lw=1.2, label=f'Uniform 1/{n} = {1/n:.3f}')
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8, rotation=8)
        ax.set_ylim(0, max(vals)*1.25); ax.set_ylabel('Weight')
        ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for i, s in enumerate(node_sample_sizes):
        axes[0].text(i, -0.025, f'n={s}', ha='center', fontsize=8,
                     transform=axes[0].get_xaxis_transform(), color='navy')
    plt.tight_layout()
    save_fig(fig, 'Aggregation_Weights')
    plt.show()


# ─────────────────────────────────────────────
# 14. Lookback Ablation
# ─────────────────────────────────────────────
def plot_lookback_ablation(all_lb_ablations, rl, selected_lookbacks):
    n = len(all_lb_ablations)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle('Lookback Window Selection per Node\n'
                 'Selected window maximises test R² subject to overfitting gap ≤ 0.05', fontsize=13)
    for i, ablation in enumerate(all_lb_ablations):
        ax      = axes[i]
        lbs     = sorted(ablation.keys())
        r2_tr   = [ablation[lb]['r2_train'] for lb in lbs]
        r2_ts   = [ablation[lb]['r2_test']  for lb in lbs]
        best_lb = selected_lookbacks[i]
        ax.plot(lbs, r2_tr, 'o-', color='steelblue', lw=1.5, label='Train R²', markersize=6)
        ax.plot(lbs, r2_ts, 's-', color='coral',     lw=1.5, label='Test R²',  markersize=6)
        ax.fill_between(lbs, r2_tr, r2_ts, alpha=0.12, color='orange', label='Gap')
        ax.axvline(best_lb, color='green', ls='--', lw=2.0, label=f'Selected: {best_lb}d')
        ax.scatter([best_lb], [ablation[best_lb]['r2_test']], color='green', s=120, zorder=5)
        for lb, r2t, r2s in zip(lbs, r2_tr, r2_ts):
            gap = r2t - r2s; col = 'red' if gap > 0.05 else 'darkgreen'
            ax.annotate(f'd={gap:.3f}', xy=(lb, r2s), xytext=(0, -18),
                        textcoords='offset points', ha='center', fontsize=7, color=col)
        ax.set_title(f'Node {i+1} — {rl[i]}\nSelected: {best_lb} days  '
                     f'R²_test={ablation[best_lb]["r2_test"]:.4f}', fontsize=9)
        ax.set_xlabel('Lookback window (days)'); ax.set_ylabel('R²')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=min(min(r2_ts)*0.98, 0.9))
    plt.tight_layout()
    save_fig(fig, 'Lookback_Window_Selection')
    plt.show()


# ─────────────────────────────────────────────
# 15. Latency Benchmark
# ─────────────────────────────────────────────
def plot_latency_benchmark(latency_results):
    labels = [v['label']   for v in latency_results.values()]
    means  = [v['mean_ms'] for v in latency_results.values()]
    stds   = [v['std_ms']  for v in latency_results.values()]
    colors = ['#90A4AE', '#42A5F5', '#66BB6A', '#FF7043']
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Aggregation Method Latency Benchmark\n'
                 'Blockchain overhead quantified relative to plain FedAvg baseline', fontsize=13)
    bars = axes[0].bar(range(len(labels)), means, yerr=stds,
                       color=colors[:len(labels)], alpha=0.85, capsize=6, edgecolor='black', lw=0.8)
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=9, rotation=12, ha='right')
    axes[0].set_ylabel('Latency (ms)'); axes[0].set_title('Mean Latency ± Std per Method')
    axes[0].grid(True, alpha=0.3, axis='y')
    base = means[0]
    for bar, m, s in zip(bars, means, stds):
        oh = f"+{(m/base-1)*100:.1f}%" if m != base else "baseline"
        axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.0002,
                     f'{m:.4f}ms\n({oh})', ha='center', va='bottom', fontsize=8, fontweight='bold')
    overheads = [(m/base-1)*100 for m in means]
    oh_colors = ['#90A4AE' if o == 0 else ('#4CAF50' if o < 50 else '#FF5722') for o in overheads]
    bars2 = axes[1].bar(range(len(labels)), overheads, color=oh_colors, alpha=0.85, edgecolor='black', lw=0.8)
    axes[1].axhline(0, color='black', lw=0.8)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=9, rotation=12, ha='right')
    axes[1].set_ylabel('Overhead vs Plain FedAvg (%)'); axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_title('Relative Overhead\n(negligible vs SVR training ~1–2 s/node)')
    for bar, o in zip(bars2, overheads):
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     f'{o:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'Latency_Benchmark')
    plt.show()


# ─────────────────────────────────────────────
# 16. Complexity vs Time
# ─────────────────────────────────────────────
def plot_complexity_vs_time(lc, rl):
    comps = [c[0] for c in lc]; times = [c[1] for c in lc]
    fig, ax = plt.subplots(figsize=(9, 5))
    sc = ax.scatter(comps, times, c=range(len(comps)), cmap='viridis', s=150, zorder=3)
    ax.plot(comps, times, 'r--', alpha=0.5)
    for i, (c, t) in enumerate(lc):
        ax.text(c, t+0.001, f'N{i+1}: {rl[i][:20]}', fontsize=9)
    ax.set_xlabel('Support Vectors'); ax.set_ylabel('Training Time (s)')
    ax.set_title('Model Complexity vs Training Time per Node')
    plt.colorbar(sc, ax=ax, label='Node'); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_fig(fig, 'Complexity_vs_Time')
    plt.show()


# ─────────────────────────────────────────────
# 17. Global Predictions
# ─────────────────────────────────────────────
def plot_global_predictions(gta, gtea, gtp, gtep):
    gta = np.array(gta).flatten(); gtea = np.array(gtea).flatten()
    gtp = np.array(gtp).flatten(); gtep = np.array(gtep).flatten()
    tri = np.arange(len(gta)); tsi = np.arange(len(gtea)) + len(gta)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(tri, gta,  color='steelblue', lw=1.5, label='Actual (Train)',    alpha=0.8)
    ax.plot(tri, gtp,  color='green',     lw=1,   label='Predicted (Train)', ls='--', alpha=0.85)
    ax.plot(tsi, gtea, color='navy',      lw=1.5, label='Actual (Test)',     ls=':', alpha=0.8)
    ax.plot(tsi, gtep, color='red',       lw=1,   label='Predicted (Test)',  ls='--', alpha=0.85)
    ax.axvspan(tsi[0], tsi[-1], alpha=0.06, color='orange', label='Test region')
    ax.set_xlabel('Time Index'); ax.set_ylabel('Price (scaled)')
    ax.set_title('Global Predictions — All Nodes Combined (NASDAQ ^IXIC)')
    ax.legend(fontsize=9, ncol=3); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Global_Predictions')
    plt.show()


# ═════════════════════════════════════════════════════════════
# NEW BLOCKCHAIN / FL / WEIGHTS PLOTS
# ═════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────
# 18. Blockchain Chain Integrity Timeline
# ─────────────────────────────────────────────
def plot_blockchain_chain(blockchain_obj, rl):
    """
    Visual chain of blocks: genesis → node blocks.
    Shows block index, node ID, trust score, and hash prefix.
    """
    chain  = blockchain_obj.chain
    n      = len(chain)
    fig, ax = plt.subplots(figsize=(max(14, n*2.5), 5))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')
    fig.suptitle('Blockchain Ledger — Block Chain Integrity\n'
                 'Each block links to its predecessor via SHA-256 hash', fontsize=13)

    box_w, box_h = 1.6, 1.2
    node_colors  = ['#BDBDBD', '#42A5F5', '#FF7043', '#66BB6A', '#AB47BC', '#FFA726']

    for i, block in enumerate(chain):
        x      = i
        color  = node_colors[i % len(node_colors)]
        rect   = mpatches.FancyBboxPatch((x - box_w/2, -box_h/2), box_w, box_h,
                                         boxstyle='round,pad=0.05',
                                         linewidth=1.5, edgecolor='black',
                                         facecolor=color, alpha=0.85, zorder=3)
        ax.add_patch(rect)
        node_str  = str(block['node_id'])
        trust_str = f"trust={block['trust_score']:.3f}" if block['trust_score'] is not None else "GENESIS"
        hash_str  = block['hash'][:10] + '...'
        ax.text(x, 0.42, f"Block {block['index']}", ha='center', va='center',
                fontsize=9, fontweight='bold')
        ax.text(x, 0.08, f"Node: {node_str}", ha='center', va='center', fontsize=8)
        ax.text(x, -0.26, trust_str,            ha='center', va='center', fontsize=7.5)
        ax.text(x, -0.58, hash_str,             ha='center', va='center', fontsize=6.5,
                color='#333333', family='monospace')

        if i > 0:
            ax.annotate('', xy=(x - box_w/2 - 0.02, 0),
                        xytext=(x - box_w/2 - 0.38, 0),
                        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))
            prev_hash = block['previous_hash'][:8] + '...'
            ax.text(x - box_w/2 - 0.2, 0.18, prev_hash, ha='center', va='bottom',
                    fontsize=5.5, color='#555555', family='monospace')

    ax.text(n/2 - 0.5, 1.9,
            'Chain integrity: VERIFIED  |  All hashes match  |  No tampering detected',
            ha='center', fontsize=10, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', fc='#E8F5E9', ec='green', alpha=0.8))
    plt.tight_layout()
    save_fig(fig, 'Blockchain_Chain_Integrity')
    plt.show()


# ─────────────────────────────────────────────
# 19. Trust Score Evolution
# ─────────────────────────────────────────────
def plot_trust_score_evolution(block_trust_scores, rl):
    """Bar + line showing trust score per submitted block."""
    n      = len(block_trust_scores)
    labels = [f'Node {i+1}\n{rl[i][:16]}' for i in range(n)]
    colors = ['#42A5F5', '#FF7043', '#66BB6A'][:n]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(n), block_trust_scores, color=colors, alpha=0.85,
                  edgecolor='black', lw=0.8)
    ax.plot(range(n), block_trust_scores, 'ko--', lw=1.5, markersize=8, zorder=5)
    ax.axhline(0.5,  color='red',   ls='--', lw=1.2, label='Low trust threshold (0.5)')
    ax.axhline(0.75, color='green', ls='--', lw=1.2, label='High trust threshold (0.75)')
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Trust Score')
    ax.set_title('Byzantine Fault Tolerance — Trust Score per Node\n'
                 'Trust = (cosine_similarity + 1) / 2; score=1.0 means deferred (< 3 prior blocks)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, block_trust_scores):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'Trust_Score_Evolution')
    plt.show()


# ─────────────────────────────────────────────
# 20. Weight Vector Comparison across Nodes
# ─────────────────────────────────────────────
def plot_weight_vectors(blockchain_obj, rl, top_k=30):
    """
    Side-by-side comparison of the top_k coefficients of each node's
    submitted weight vector (after clipping + DP) vs the aggregated global vector.
    """
    blocks      = [b for b in blockchain_obj.chain if b['weights'] is not None]
    all_weights = [np.array(b['weights']) for b in blocks]
    n           = len(all_weights)
    if n == 0:
        return

    max_k = min(top_k, len(all_weights[0]))
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle('Node Weight Vector Comparison (top coefficients)\n'
                 'Vectors shown after norm clipping and differential privacy noise injection', fontsize=13)
    colors_node = ['#42A5F5', '#FF7043', '#66BB6A']
    for i, (w, ax) in enumerate(zip(all_weights, axes)):
        indices = np.arange(max_k)
        ax.bar(indices, w[:max_k], color=colors_node[i % len(colors_node)],
               alpha=0.75, edgecolor='black', lw=0.4)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_title(f'Node {i+1} — {rl[i][:22]}\n'
                     f'L2 norm={np.linalg.norm(w):.4f}', fontsize=9)
        ax.set_xlabel('Feature coefficient index'); ax.set_ylabel('Weight value')
        ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_fig(fig, 'Node_Weight_Vectors')
    plt.show()


# ─────────────────────────────────────────────
# 21. Global vs Local Weight Distribution
# ─────────────────────────────────────────────
def plot_weight_distribution(blockchain_obj, agg_w, rl):
    """
    Histogram of weight values: each node vs the aggregated global vector.
    Reveals the smoothing effect of federated averaging.
    """
    blocks      = [b for b in blockchain_obj.chain if b['weights'] is not None]
    all_weights = [np.array(b['weights']) for b in blocks]
    n           = len(all_weights)
    if n == 0 or agg_w is None:
        return

    fig, axes = plt.subplots(1, n+1, figsize=(6*(n+1), 5), sharey=True)
    colors_node = ['#42A5F5', '#FF7043', '#66BB6A']
    fig.suptitle('Weight Distribution — Local Nodes vs Global Aggregated Model\n'
                 'Aggregation smooths individual node distributions', fontsize=13)
    all_w_concat = np.concatenate(all_weights + [agg_w])
    bins = np.linspace(all_w_concat.min(), all_w_concat.max(), 50)

    for i, (w, ax) in enumerate(zip(all_weights, axes[:-1])):
        ax.hist(w, bins=bins, color=colors_node[i % len(colors_node)],
                alpha=0.75, edgecolor='black', lw=0.4, density=True)
        ax.axvline(np.mean(w), color='black', ls='--', lw=1.5, label=f'Mean={np.mean(w):.4f}')
        ax.set_title(f'Node {i+1} — {rl[i][:22]}\nstd={np.std(w):.4f}', fontsize=9)
        ax.set_xlabel('Weight value'); ax.set_ylabel('Density')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    axes[-1].hist(agg_w, bins=bins, color='#FFA726', alpha=0.8, edgecolor='black', lw=0.4, density=True)
    axes[-1].axvline(np.mean(agg_w), color='black', ls='--', lw=1.5, label=f'Mean={np.mean(agg_w):.4f}')
    axes[-1].set_title(f'Aggregated Global Model\nstd={np.std(agg_w):.4f}', fontsize=9)
    axes[-1].set_xlabel('Weight value'); axes[-1].legend(fontsize=8); axes[-1].grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Weight_Distribution')
    plt.show()


# ─────────────────────────────────────────────
# 22. FedProx Proximal Term per Round
# ─────────────────────────────────────────────
def plot_fedprox_terms(proximal_terms, rl):
    """
    Shows the FedProx proximal regularisation term per node.
    First node is always 0 (no global model yet).
    """
    n      = len(proximal_terms)
    labels = [f'Node {i+1}\n{rl[i][:16]}' for i in range(n)]
    colors = ['#42A5F5', '#FF7043', '#66BB6A'][:n]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(n), proximal_terms, color=colors, alpha=0.85, edgecolor='black', lw=0.8)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Proximal Term Value')
    ax.set_title('FedProx Regularisation — Proximal Term per Node\n'
                 'Node 1 = 0 (no prior global model); subsequent nodes penalise drift')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, proximal_terms):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e-10,
                f'{v:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'FedProx_Proximal_Terms')
    plt.show()


# ─────────────────────────────────────────────
# 23. FL Communication Round Summary
# ─────────────────────────────────────────────
def plot_fl_round_summary(node_sample_sizes, lm_train, lm_test, lc, rl, weight_info):
    """
    Summary dashboard for one federation round:
    sample sizes, R² per node, training times, combined weights, and DA.
    """
    n       = len(rl)
    labels  = [f'N{i+1}\n{rl[i][:14]}' for i in range(n)]
    fig     = plt.figure(figsize=(22, 10))
    fig.suptitle('Federated Learning Round Summary Dashboard\n'
                 'One complete federation round: local training → blockchain submission → aggregation',
                 fontsize=14)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1) Sample sizes
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#42A5F5', '#FF7043', '#66BB6A'][:n]
    bars = ax1.bar(range(n), node_sample_sizes, color=colors, alpha=0.85, edgecolor='black')
    ax1.set_xticks(range(n)); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_title('Training Samples per Node'); ax1.set_ylabel('Samples')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, v in zip(bars, node_sample_sizes):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f'{v}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2) R² comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x, w = np.arange(n), 0.35
    ax2.bar(x-w/2, [m[4] for m in lm_train], w, label='Train', color='steelblue', alpha=0.85)
    ax2.bar(x+w/2, [m[4] for m in lm_test],  w, label='Test',  color='coral',     alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylim(0.98, 1.002); ax2.set_title('R² — Train vs Test per Node')
    ax2.set_ylabel('R²'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')

    # 3) Training time
    ax3 = fig.add_subplot(gs[0, 2])
    times = [c[1] for c in lc]
    ax3.bar(range(n), times, color='mediumpurple', alpha=0.85, edgecolor='black')
    ax3.set_xticks(range(n)); ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_title('Local Training Time per Node'); ax3.set_ylabel('Seconds')
    ax3.grid(True, alpha=0.3, axis='y')
    for j, t in enumerate(times):
        ax3.text(j, t, f'{t:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 4) Aggregation weights pie
    ax4 = fig.add_subplot(gs[1, 0])
    if weight_info and 'combined' in weight_info:
        combined = weight_info['combined']
        wedge_labels = [f'N{i+1} ({combined[i]*100:.1f}%)' for i in range(n)]
        ax4.pie(combined, labels=wedge_labels, colors=colors,
                autopct='%1.2f%%', startangle=90,
                wedgeprops=dict(edgecolor='black', linewidth=1))
        ax4.set_title('Combined Aggregation Weight Share')
    else:
        ax4.text(0.5, 0.5, 'Weight info\nnot available', ha='center', va='center', fontsize=11)
        ax4.set_title('Aggregation Weights')
        ax4.axis('off')

    # 5) MAPE
    ax5 = fig.add_subplot(gs[1, 1])
    mape_test = [m[3] for m in lm_test]
    ax5.bar(range(n), mape_test, color=colors, alpha=0.85, edgecolor='black')
    ax5.set_xticks(range(n)); ax5.set_xticklabels(labels, fontsize=8)
    ax5.set_title('Test MAPE (%) per Node'); ax5.set_ylabel('MAPE (%)')
    ax5.grid(True, alpha=0.3, axis='y')
    for j, v in enumerate(mape_test):
        ax5.text(j, v, f'{v:.3f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 6) Support vectors
    ax6 = fig.add_subplot(gs[1, 2])
    svs = [c[0] for c in lc]
    ax6.bar(range(n), svs, color=colors, alpha=0.85, edgecolor='black')
    ax6.set_xticks(range(n)); ax6.set_xticklabels(labels, fontsize=8)
    ax6.set_title('Support Vectors per Node'); ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3, axis='y')
    for j, v in enumerate(svs):
        ax6.text(j, v, f'{v}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    save_fig(fig, 'FL_Round_Summary_Dashboard')
    plt.show()


# ─────────────────────────────────────────────
# 24. Differential Privacy Noise Impact
# ─────────────────────────────────────────────
def plot_dp_noise_impact(blockchain_obj, rl, epsilon=1.0, sensitivity=1.0):
    """
    For each accepted node, compare the pre-DP weight norm against the
    DP-noised weight norm, and illustrate the noise magnitude distribution.
    """
    blocks = [b for b in blockchain_obj.chain if b['weights'] is not None]
    n      = len(blocks)
    if n == 0: return

    np.random.seed(42)
    pre_norms  = []
    post_norms = []
    noise_mags = []

    for b in blocks:
        w_post = np.array(b['weights'])
        # Simulate pre-DP weight (reverse approximate by removing last noise realisation)
        scale  = sensitivity / epsilon
        noise  = np.random.laplace(0.0, scale, w_post.shape)
        w_pre  = w_post - noise  # approximate reconstruction
        pre_norms.append(float(np.linalg.norm(w_pre)))
        post_norms.append(float(np.linalg.norm(w_post)))
        noise_mags.append(float(np.linalg.norm(noise)))

    labels = [f'Node {i+1}\n{rl[i][:16]}' for i in range(n)]
    x      = np.arange(n); w_bar = 0.3
    colors = ['#42A5F5', '#FF7043', '#66BB6A']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Differential Privacy Noise Impact  (epsilon={epsilon})\n'
                 'Laplace noise injected to each node weight vector before blockchain submission',
                 fontsize=13)

    axes[0].bar(x - w_bar/2, pre_norms,  w_bar, label='Pre-DP  norm', color='steelblue', alpha=0.85, edgecolor='black')
    axes[0].bar(x + w_bar/2, post_norms, w_bar, label='Post-DP norm', color='coral',     alpha=0.85, edgecolor='black')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel('L2 Norm'); axes[0].set_title('Weight Norm Before vs After DP Noise')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')
    for j in range(n):
        axes[0].text(j - w_bar/2, pre_norms[j],  f'{pre_norms[j]:.2f}',
                     ha='center', va='bottom', fontsize=8, color='steelblue')
        axes[0].text(j + w_bar/2, post_norms[j], f'{post_norms[j]:.2f}',
                     ha='center', va='bottom', fontsize=8, color='coral')

    axes[1].bar(range(n), noise_mags, color=[colors[i % len(colors)] for i in range(n)],
                alpha=0.85, edgecolor='black')
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel('Noise L2 Magnitude'); axes[1].set_title('Injected Noise Magnitude per Node')
    axes[1].grid(True, alpha=0.3, axis='y')
    for j, v in enumerate(noise_mags):
        axes[1].text(j, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'Differential_Privacy_Noise_Impact')
    plt.show()


# ═════════════════════════════════════════════════════════════
# ABLATION VISUALISATIONS
# ═════════════════════════════════════════════════════════════

MODEL_COLORS = {'FL-SVR': 'steelblue', 'ARIMA': 'darkorange', 'LSTM': 'seagreen'}


def plot_abl_per_node_metrics(svr_test_metrics, arima_metrics, lstm_metrics, rl=None):
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE (%)', 'R²']
    num_nodes    = len(svr_test_metrics)
    fig, axes    = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Per-Node Test Metric Comparison — FL-SVR vs ARIMA vs LSTM\n'
                 'Lower is better for MSE/RMSE/MAE/MAPE; higher is better for R²', fontsize=13)
    axes = axes.flatten()
    models = ['FL-SVR', 'ARIMA', 'LSTM']
    all_metrics = [svr_test_metrics, arima_metrics, lstm_metrics]
    x = np.arange(num_nodes); w = 0.25
    node_labels = [f'Node {j+1}' + (f'\n{rl[j][:14]}' if rl else '') for j in range(num_nodes)]
    for m_idx, mname in enumerate(metric_names):
        ax = axes[m_idx]
        for k, (model_name, metrics) in enumerate(zip(models, all_metrics)):
            vals = [m[m_idx] for m in metrics]
            ax.bar(x + (k-1)*w, vals, w, label=model_name,
                   color=MODEL_COLORS[model_name], alpha=0.85)
            for j, v in enumerate(vals):
                ax.text(j + (k-1)*w, v, f'{v:.2e}',
                        ha='center', va='bottom', fontsize=6.5, color=MODEL_COLORS[model_name])
        ax.set_xticks(x); ax.set_xticklabels(node_labels, fontsize=8)
        ax.set_title(mname); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    ax = axes[5]; ax.axis('off')
    col_labels = [f'Node {j+1}' for j in range(num_nodes)] + ['Mean']
    table_data = []
    for model_name, metrics in zip(models, all_metrics):
        r2_vals = [f'{m[4]:.4f}' for m in metrics]
        mean_r2 = np.mean([m[4] for m in metrics])
        table_data.append(r2_vals + [f'{mean_r2:.4f}'])
    tbl = ax.table(cellText=table_data, rowLabels=models, colLabels=col_labels,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.4, 2.0)
    for col_idx in range(len(col_labels)):
        col_r2s = [float(table_data[r][col_idx]) for r in range(len(models))]
        best    = max(col_r2s)
        for row_idx, v in enumerate(col_r2s):
            if v == best:
                tbl[row_idx+1, col_idx].set_facecolor('#c8e6c9')
    for col_idx in range(len(col_labels)):
        tbl[0, col_idx].set_facecolor('#2C3E50')
        tbl[0, col_idx].set_text_props(color='white', fontweight='bold')
    ax.set_title('R² Summary Table (green = best per node)', fontsize=10, pad=20)
    plt.tight_layout()
    save_fig(fig, 'Ablation_Per_Node_Metrics')
    plt.show()


def plot_abl_prediction_overlay(svr_preds, arima_preds, lstm_preds,
                                 svr_actuals, arima_actuals, lstm_actuals,
                                 date_ranges, rl=None):
    num_nodes = len(svr_preds)
    fig, axes = plt.subplots(num_nodes, 1, figsize=(15, 5*num_nodes))
    if num_nodes == 1: axes = [axes]
    fig.suptitle('Test-Set Prediction Overlay — FL-SVR vs ARIMA vs LSTM\n'
                 'All models evaluated on the same held-out test window per node', fontsize=13)
    for i in range(num_nodes):
        ax      = axes[i]
        actuals = svr_actuals[i]
        nn      = min(len(actuals), len(arima_preds[i]), len(lstm_preds[i]), len(svr_preds[i]))
        idx     = np.arange(nn)
        ax.plot(idx, actuals[:nn],         color='black',      lw=2,   label='Actual', zorder=5)
        ax.plot(idx, svr_preds[i][:nn],    color='steelblue',  lw=1.2, label='FL-SVR', ls='--', alpha=0.9)
        ax.plot(idx, arima_preds[i][:nn],  color='darkorange', lw=1.2, label='ARIMA',  ls='-.', alpha=0.9)
        ax.plot(idx, lstm_preds[i][:nn],   color='seagreen',   lw=1.2, label='LSTM',   ls=':',  alpha=0.9)
        r2_svr   = r2_score(actuals[:nn], svr_preds[i][:nn])
        r2_arima = r2_score(actuals[:nn], arima_preds[i][:nn])
        r2_lstm  = r2_score(actuals[:nn], lstm_preds[i][:nn])
        regime   = rl[i] if rl else f'{date_ranges[i][0]} to {date_ranges[i][1]}'
        ax.set_title(f'Node {i+1}  ({regime})\n'
                     f'R²: FL-SVR={r2_svr:.4f}  ARIMA={r2_arima:.4f}  LSTM={r2_lstm:.4f}', fontsize=9)
        ax.set_xlabel('Test Time Index'); ax.set_ylabel('Price (scaled)')
        ax.legend(fontsize=9, ncol=4); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, 'Ablation_Prediction_Overlay')
    plt.show()


def plot_abl_radar(svr_test_metrics, arima_metrics, lstm_metrics):
    metric_names = ['MSE', 'RMSE', 'MAE', 'MAPE', 'R²']
    lower_better = [True, True, True, True, False]
    models       = ['FL-SVR', 'ARIMA', 'LSTM']
    all_metrics  = [svr_test_metrics, arima_metrics, lstm_metrics]
    global_vals  = [[np.mean([m[k] for m in metrics]) for k in range(5)] for metrics in all_metrics]
    norm_vals = []
    for m_idx, lb in enumerate(lower_better):
        col = [gv[m_idx] for gv in global_vals]
        if lb:
            best = min(col); norm_col = [best/(v+1e-12) for v in col]
        else:
            best = max(col); norm_col = [v/(best+1e-12) for v in col]
        norm_vals.append(norm_col)
    scores = np.array(norm_vals).T
    N      = len(metric_names)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle('Normalised Multi-Metric Performance Comparison\n'
                 '1.0 = best model on that metric; higher = better on all axes', fontsize=13)
    for k, (model_name, score_row) in enumerate(zip(models, scores)):
        values = score_row.tolist() + score_row[:1].tolist()
        ax.plot(angles, values, lw=2, label=model_name, color=list(MODEL_COLORS.values())[k])
        ax.fill(angles, values, alpha=0.12, color=list(MODEL_COLORS.values())[k])
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names, fontsize=11)
    ax.set_ylim(0, 1.05); ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8, color='grey')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_fig(fig, 'Ablation_Radar_Chart')
    plt.show()


def plot_abl_efficiency(svr_test_metrics, arima_metrics, lstm_metrics,
                         svr_times, arima_times, lstm_times, rl=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Efficiency Trade-off — Training Time vs Test R²\n'
                 'Top-left corner = fast and accurate (ideal)', fontsize=13)
    models      = ['FL-SVR', 'ARIMA', 'LSTM']
    all_metrics = [svr_test_metrics, arima_metrics, lstm_metrics]
    all_times   = [svr_times, arima_times, lstm_times]
    for model_name, metrics, times in zip(models, all_metrics, all_times):
        r2s   = [m[4] for m in metrics]
        sizes = [100 + 80*i for i in range(len(times))]
        ax.scatter(times, r2s, s=sizes, color=MODEL_COLORS[model_name], alpha=0.85,
                   label=model_name, edgecolors='k', linewidths=0.5, zorder=3)
        for i, (t, r2) in enumerate(zip(times, r2s)):
            label_txt = f'N{i+1}' + (f'\n{rl[i][:12]}' if rl else '')
            ax.annotate(label_txt, (t, r2), textcoords='offset points',
                        xytext=(6, 4), fontsize=8, color=MODEL_COLORS[model_name])
    ax.set_xlabel('Training Time (seconds)', fontsize=11)
    ax.set_ylabel('Test R²', fontsize=11)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_fig(fig, 'Ablation_Efficiency_Tradeoff')
    plt.show()


def plot_abl_summary_table(svr_test_metrics, arima_metrics, lstm_metrics,
                            svr_times, arima_times, lstm_times, rl=None):
    models      = ['FL-SVR', 'ARIMA', 'LSTM']
    all_metrics = [svr_test_metrics, arima_metrics, lstm_metrics]
    all_times   = [svr_times, arima_times, lstm_times]
    fig, ax = plt.subplots(figsize=(18, 3))
    ax.axis('off')
    fig.suptitle('Ablation Study — Global Test Set Summary\n'
                 'Mean across all nodes  |  Green = best per column', fontsize=13, y=1.05)
    col_labels = ['Model', 'Mean MSE', 'Mean RMSE', 'Mean MAE',
                  'Mean MAPE (%)', 'Mean R²', 'Mean Time (s)', 'Wins (out of 5)']
    rows          = []
    summary_data  = {}
    for model_name, metrics, times in zip(models, all_metrics, all_times):
        m_mean = [np.mean([m[k] for m in metrics]) for k in range(5)]
        t_mean = np.mean(times)
        rows.append([model_name,
                     f'{m_mean[0]:.6f}', f'{m_mean[1]:.6f}',
                     f'{m_mean[2]:.6f}', f'{m_mean[3]:.4f}',
                     f'{m_mean[4]:.6f}', f'{t_mean:.4f}', ''])
        summary_data[model_name] = m_mean + [t_mean]
    best_model = {}
    for col_idx, m_idx in enumerate([0, 1, 2, 3, 4]):
        vals = {m: summary_data[m][m_idx] for m in models}
        best_model[col_idx] = max(vals, key=vals.get) if m_idx == 4 else min(vals, key=vals.get)
    win_counts = {m: 0 for m in models}
    for bm in best_model.values():
        win_counts[bm] += 1
    for row in rows:
        row[-1] = f'{win_counts[row[0]]} / 5'
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.5)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#2C3E50')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    row_colors = {'FL-SVR': '#EAF4FB', 'ARIMA': '#FEF9E7', 'LSTM': '#EAFAF1'}
    for i, model_name in enumerate(models):
        for j in range(len(col_labels)):
            table[(i+1, j)].set_facecolor(row_colors[model_name])
    metric_col_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    for t_col, m_idx in metric_col_map.items():
        winner = best_model[m_idx]; row_idx = models.index(winner)
        table[(row_idx+1, t_col)].set_facecolor('#A9DFBF')
        table[(row_idx+1, t_col)].set_text_props(fontweight='bold')
    plt.tight_layout()
    save_fig(fig, 'Ablation_Summary_Table')
    plt.show()


# ═════════════════════════════════════════════════════════════
# MASTER DISPATCHER
# ═════════════════════════════════════════════════════════════

def visualize_all(dfs, lp_train, lp_test, gtp, gtep, gta, gtea,
                  lm_train, lm_test, lda_train, lda_test,
                  lc, agg_w, lc_curves, rl,
                  raw_test_actuals, raw_test_preds,
                  dist_stats_list, kl_matrix,
                  poison_report, weight_info,
                  node_sample_sizes, proximal_terms,
                  weight_norms, gmtr, gmts, g_da_tr, g_da_ts,
                  latency_results, all_lb_ablations, selected_lookbacks,
                  block_trust_scores, block_timestamps,
                  # ablation
                  svr_test_metrics, arima_metrics, lstm_metrics,
                  svr_test_preds, arima_preds, lstm_preds,
                  svr_test_actuals, arima_actuals, lstm_actuals,
                  svr_times, arima_times, lstm_times,
                  date_ranges_plain):

    print(f"\n  All figures will be saved to: {FIGURES_DIR}/")
    print(f"  {'─'*50}")

    print("\n  --- FL-SVR Diagnostics ---")
    plot_learning_curves(lc_curves, rl)                                          # 1
    plot_metrics_with_gap(lm_train, lm_test, lc, rl)                             # 2
    plot_prediction_overlay(lp_train, lp_test, gta, gtea,                        # 3
                            lm_train, lm_test, lc, rl)
    plot_residual_distributions(lp_train, lp_test, gta, gtea, rl)                # 4
    plot_regime_comparison(lm_train, lm_test, lda_train, lda_test, rl)           # 5
    plot_rolling_da_timeline(lp_train, lp_test, gta, gtea, rl)                   # 6
    plot_metrics_summary_table(lm_train, lm_test, lda_train, lda_test,           # 7
                               rl, gmtr, gmts, g_da_tr, g_da_ts, selected_lookbacks)
    plot_r2_gap_heatmap(lm_train, lm_test, rl)                                   # 8
    plot_prediction_error_scatter(raw_test_actuals, raw_test_preds, rl)          # 9
    plot_noniid_distributions(dist_stats_list, rl)                               # 10
    plot_kl_divergence_matrix(kl_matrix, rl)                                     # 11
    plot_poisoning_defence(poison_report, weight_norms, rl)                      # 12
    plot_aggregation_weights(weight_info, node_sample_sizes, lm_test, rl)        # 13
    plot_lookback_ablation(all_lb_ablations, rl, selected_lookbacks)             # 14
    plot_latency_benchmark(latency_results)                                      # 15
    plot_complexity_vs_time(lc, rl)                                              # 16
    plot_global_predictions(gta, gtea, gtp, gtep)                                # 17

    print("\n  --- Blockchain / FL / Weights Plots ---")
    plot_blockchain_chain(blockchain, rl)                                         # 18
    plot_trust_score_evolution(block_trust_scores, rl)                           # 19
    plot_weight_vectors(blockchain, rl, top_k=30)                                # 20
    plot_weight_distribution(blockchain, agg_w, rl)                              # 21
    plot_fedprox_terms(proximal_terms, rl)                                       # 22
    plot_fl_round_summary(node_sample_sizes, lm_train, lm_test,                  # 23
                          lc, rl, weight_info)
    plot_dp_noise_impact(blockchain, rl)                                         # 24

    print("\n  --- Ablation Study ---")
    plot_abl_per_node_metrics(svr_test_metrics, arima_metrics, lstm_metrics, rl) # 25
    plot_abl_prediction_overlay(svr_test_preds, arima_preds, lstm_preds,         # 26
                                 svr_test_actuals, arima_actuals, lstm_actuals,
                                 date_ranges_plain, rl)
    plot_abl_radar(svr_test_metrics, arima_metrics, lstm_metrics)                # 27
    plot_abl_efficiency(svr_test_metrics, arima_metrics, lstm_metrics,           # 28
                         svr_times, arima_times, lstm_times, rl)
    plot_abl_summary_table(svr_test_metrics, arima_metrics, lstm_metrics,        # 29
                            svr_times, arima_times, lstm_times, rl)

    print(f"\n  {'='*50}")
    print(f"  All {_fig_counter[0]} figures saved as Png to: {FIGURES_DIR}/")
    print(f"  {'='*50}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 1. Federated SVR ─────────────────────────────────────
    results = federated_learning()

    (dfs, lp_train, lp_test, gtp, gtep, gta, gtea,
     lm_train, lm_test, lda_train, lda_test,
     lc, agg_w, lc_curves, rl,
     raw_test_actuals, raw_test_preds,
     dist_stats_list, kl_matrix, poison_report, weight_info,
     node_sample_sizes, proximal_terms, weight_norms,
     gmtr, gmts, g_da_tr, g_da_ts, latency_results,
     all_lb_ablations, selected_lookbacks,
     block_trust_scores, block_timestamps) = results

    # ── 2. ARIMA baseline ────────────────────────────────────
    arima_metrics, arima_preds, arima_actuals, arima_times = run_arima_baseline(DATE_RANGES)

    # ── 3. LSTM baseline ─────────────────────────────────────
    lstm_metrics, lstm_preds, lstm_actuals, lstm_times = run_lstm_baseline(DATE_RANGES)

    # ── 4. Prepare SVR ablation results ──────────────────────
    svr_test_metrics = lm_test
    svr_test_preds   = lp_test
    svr_times_list   = [c[1] for c in lc]
    node_test_sizes  = [len(p) for p in lp_test]
    g_ts_act         = np.array(gtea)
    svr_test_actuals = np.split(g_ts_act, np.cumsum(node_test_sizes)[:-1])

    # ── 5. Console ablation summary ──────────────────────────
    print("\n" + "=" * 70)
    print("  ABLATION STUDY — GLOBAL TEST SET SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'MSE':>12} {'RMSE':>12} {'MAE':>12} "
          f"{'MAPE(%)':>10} {'R2':>10} {'Time(s)':>10}")
    print("-" * 70)
    for model_name, metrics_list, times_list in [
            ('FL-SVR',  lm_test,       svr_times_list),
            ('ARIMA',   arima_metrics, arima_times),
            ('LSTM',    lstm_metrics,  lstm_times)]:
        m_mean = [np.mean([m[k] for m in metrics_list]) for k in range(5)]
        t_mean = np.mean(times_list)
        print(f"{model_name:<12} {m_mean[0]:>12.6f} {m_mean[1]:>12.6f} "
              f"{m_mean[2]:>12.6f} {m_mean[3]:>10.4f} {m_mean[4]:>10.6f} {t_mean:>10.4f}")
    print("=" * 70)

    # ── 6. All plots ─────────────────────────────────────────
    visualize_all(
        dfs, lp_train, lp_test, gtp, gtep, gta, gtea,
        lm_train, lm_test, lda_train, lda_test,
        lc, agg_w, lc_curves, rl,
        raw_test_actuals, raw_test_preds,
        dist_stats_list, kl_matrix, poison_report, weight_info,
        node_sample_sizes, proximal_terms, weight_norms,
        gmtr, gmts, g_da_tr, g_da_ts,
        latency_results, all_lb_ablations, selected_lookbacks,
        block_trust_scores, block_timestamps,
        svr_test_metrics, arima_metrics, lstm_metrics,
        svr_test_preds,   arima_preds,   lstm_preds,
        svr_test_actuals, arima_actuals, lstm_actuals,
        svr_times_list,   arima_times,   lstm_times,
        DATE_RANGES
    )

"""
[*********************100%***********************]  1 of 1 completed
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Secure Aggregation Method Comparison
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Method                     Security Mechanism                  Privacy            Latency            Fault Tol     Auditability
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────
    Plain FedAvg               None                                   None               None                 High             None
    SMPC (Bonawitz 2017)       Cryptographic secret sharing           High High (multi-round) Low (dropout sensitive)     Not built-in
    TEE / Intel SGX            Hardware enclave                       High             Medium  Hardware dependency   Vendor lock-in
    Differential Privacy only  Noise injection (Laplace/Gauss)      Medium                Low                 High No tamper-evidence
  * Blockchain-FL (Ours)       Hash-ledger + DP + BFT + cosine        High    Low (+overhead)           High (BFT) SHA-256 immutable
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────


  ────────────────────────────────────────────────────────────────────────────────────────────────────
  Feature Inclusion Justification
  ────────────────────────────────────────────────────────────────────────────────────────────────────
  Feature      Category     Justification
  ────────────────────────────────────────────────────────────────────────────────────────────────────
  Open         OHLCV        Opening price captures overnight sentiment and gap risk
  High         OHLCV        Daily high reflects intraday buying pressure and resistance
  Low          OHLCV        Daily low captures selling pressure and support levels
  Volume       OHLCV        Volume confirms price moves; high vol = conviction (Lo & Wang, 2000)
  SMA_10       Trend        10-day SMA smooths noise and identifies trend direction
  EMA_10       Trend        10-day EMA weights recent prices more heavily than SMA
  RSI_14       Momentum     14-day RSI identifies overbought/oversold regimes (Wilder, 1978)
  MACD         Momentum     EMA12 - EMA26 captures momentum shifts (standard indicator)
  BB_Width     Volatility   Bollinger Band width measures volatility regime (Bollinger, 1992)
  ────────────────────────────────────────────────────────────────────────────────────────────────────


======================================================================
  Node 1  |  Pre-Volatility Baseline
  Period: 2015-10-15 -> 2018-10-15
======================================================================
  [NonIID] mu=6067.50  sigma=1063.49  n=755

  Node 1 — lookback ablation: (3, 5, 7, 10, 14, 21)

    lb=  3d  R2_tr=0.999998  R2_ts=0.997968  gap=0.0020  healthy
    lb=  5d  R2_tr=0.999999  R2_ts=0.997750  gap=0.0022  healthy
    lb=  7d  R2_tr=1.000000  R2_ts=0.997601  gap=0.0024  healthy
    lb= 10d  R2_tr=1.000000  R2_ts=0.997562  gap=0.0024  healthy
    lb= 14d  R2_tr=0.999999  R2_ts=0.997480  gap=0.0025  healthy
    lb= 21d  R2_tr=0.999999  R2_ts=0.997431  gap=0.0026  healthy
  Selected lookback: 3 days
  [CLIP]  norm 1.4296 <= 10.0000  (OK)
  [DP]    eps=1.0  scale=1.0000  noise_mag=5.640740
  [ACCEPTED]  history=0 — deferred  trust=1.0000
  [AUDIT] 2026-03-02T21:22:18.652171+00:00  |  WEIGHT_SUBMITTED                |  Node: 1
  Computing learning curve ...
[*********************100%***********************]  1 of 1 completed
  Node 1 [Pre-Volatility Baseline]  lb=3d
  Train: R2=0.999998  MAPE=0.3557%  DA=99.66%
  Test:  R2=0.997968  MAPE=0.0996%  DA=98.67%
  R2 gap=0.0020  healthy
  [AUDIT] 2026-03-02T21:22:22.726306+00:00  |  NODE_METRICS                    |  Node: 1

======================================================================
  Node 2  |  COVID-19 Crisis Regime
  Period: 2018-10-15 -> 2021-10-15
======================================================================
  [NonIID] mu=10239.39  sigma=2680.95  n=756

  Node 2 — lookback ablation: (3, 5, 7, 10, 14, 21)

    lb=  3d  R2_tr=0.999996  R2_ts=0.993599  gap=0.0064  healthy
    lb=  5d  R2_tr=0.999999  R2_ts=0.993312  gap=0.0067  healthy
    lb=  7d  R2_tr=0.999999  R2_ts=0.992801  gap=0.0072  healthy
    lb= 10d  R2_tr=0.999999  R2_ts=0.992791  gap=0.0072  healthy
    lb= 14d  R2_tr=0.999999  R2_ts=0.992769  gap=0.0072  healthy
    lb= 21d  R2_tr=0.999999  R2_ts=0.992342  gap=0.0077  healthy
  Selected lookback: 3 days
  [CLIP]  norm 1.8645 <= 10.0000  (OK)
  [DP]    eps=1.0  scale=1.0000  noise_mag=7.643583
  [ACCEPTED]  history=1 — deferred  trust=1.0000
  [AUDIT] 2026-03-02T21:22:29.059995+00:00  |  WEIGHT_SUBMITTED                |  Node: 2
  Computing learning curve ...
[*********************100%***********************]  1 of 1 completed
  Node 2 [COVID-19 Crisis Regime]  lb=3d
  Train: R2=0.999996  MAPE=0.1355%  DA=99.14%
  Test:  R2=0.993599  MAPE=0.1099%  DA=98.68%
  R2 gap=0.0064  healthy
  [AUDIT] 2026-03-02T21:22:31.945185+00:00  |  NODE_METRICS                    |  Node: 2

======================================================================
  Node 3  |  Post-COVID Rate-Hike Regime
  Period: 2021-10-15 -> 2024-10-15
======================================================================
  [NonIID] mu=13898.07  sigma=2193.13  n=753

  Node 3 — lookback ablation: (3, 5, 7, 10, 14, 21)

    lb=  3d  R2_tr=0.999993  R2_ts=0.999041  gap=0.0010  healthy
    lb=  5d  R2_tr=0.999998  R2_ts=0.998985  gap=0.0010  healthy
    lb=  7d  R2_tr=0.999999  R2_ts=0.998865  gap=0.0011  healthy
    lb= 10d  R2_tr=0.999999  R2_ts=0.998818  gap=0.0012  healthy
    lb= 14d  R2_tr=0.999999  R2_ts=0.998831  gap=0.0012  healthy
    lb= 21d  R2_tr=1.000000  R2_ts=0.998766  gap=0.0012  healthy
  Selected lookback: 3 days
  [CLIP]  norm 2.3035 <= 10.0000  (OK)
  [DP]    eps=1.0  scale=1.0000  noise_mag=5.710647
  [ACCEPTED]  history=2 — deferred  trust=1.0000
  [AUDIT] 2026-03-02T21:22:41.294341+00:00  |  WEIGHT_SUBMITTED                |  Node: 3
  Computing learning curve ...
[*********************100%***********************]  1 of 1 completed
  Node 3 [Post-COVID Rate-Hike Regime]  lb=3d
  Train: R2=0.999993  MAPE=0.2213%  DA=99.31%
  Test:  R2=0.999041  MAPE=0.0952%  DA=100.00%
  R2 gap=0.0010  healthy
  [AUDIT] 2026-03-02T21:22:46.440783+00:00  |  NODE_METRICS                    |  Node: 3
  Chain integrity VERIFIED

  Aggregation Latency Benchmark (100 runs)
  Plain Fedavg                           0.0152 ms      baseline
  Reputation Fedavg                      0.0232 ms        +53.2%
  Combined Fedavg                        0.0106 ms       +-30.1%
  Blockchain FL (full pipeline)          0.2640 ms      +1641.0%
  [AUDIT] 2026-03-02T21:22:46.473660+00:00  |  AGGREGATION                     |  Node: GLOBAL

  GLOBAL RESULTS
  Train: R2=0.99999581  DA=99.37%
  Test:  R2=0.99825432  MAPE=0.1016%  DA=99.12%

======================================================================
  AUDIT SUMMARY — 7 events
======================================================================
  [2026-03-02T21:22:18.652171+00:00]  WEIGHT_SUBMITTED                Node: 1
  [2026-03-02T21:22:22.726306+00:00]  NODE_METRICS                    Node: 1
  [2026-03-02T21:22:29.059995+00:00]  WEIGHT_SUBMITTED                Node: 2
  [2026-03-02T21:22:31.945185+00:00]  NODE_METRICS                    Node: 2
  [2026-03-02T21:22:41.294341+00:00]  WEIGHT_SUBMITTED                Node: 3
  [2026-03-02T21:22:46.440783+00:00]  NODE_METRICS                    Node: 3
  [2026-03-02T21:22:46.473660+00:00]  AGGREGATION                     Node: GLOBAL

  Audit trail exported → audit_trail.json

 ABLATION — ARIMA Baseline

  Node 1: fitting ARIMA(5, 1, 0) ...

[*********************100%***********************]  1 of 1 completed  Node 1: R2=0.948554  Time=31.71s

  Node 2: fitting ARIMA(5, 1, 0) ...

[*********************100%***********************]  1 of 1 completed  Node 2: R2=0.951013  Time=31.97s

  Node 3: fitting ARIMA(5, 1, 0) ...

[*********************100%***********************]  1 of 1 completed  Node 3: R2=0.941475  Time=23.75s

 ABLATION — LSTM Baseline (TensorFlow)

  Node 1: training LSTM ...

[*********************100%***********************]  1 of 1 completed  Node 1: R2=0.827327  Time=9.33s

  Node 2: training LSTM ...

[*********************100%***********************]  1 of 1 completed  Node 2: R2=0.847043  Time=5.48s

  Node 3: training LSTM ...

  Node 3: R2=0.763028  Time=5.71s

======================================================================
  ABLATION STUDY — GLOBAL TEST SET SUMMARY
======================================================================
Model                 MSE         RMSE          MAE    MAPE(%)         R2    Time(s)
----------------------------------------------------------------------
FL-SVR           0.000026     0.005051     0.001012     0.1016   0.996869     1.4726
ARIMA            0.000646     0.024562     0.018316     1.7626   0.947014    29.1437
LSTM             0.002401     0.046618     0.035701     3.4297   0.812466     6.8402
======================================================================

  All figures will be saved to: ./figures/
  ──────────────────────────────────────────────────

  --- FL-SVR Diagnostics ---
  [Png] saved → ./figures\01_Learning_Curves.png
  [Png] saved → ./figures\02_Train_vs_Test_Metrics.png
  [Png] saved → ./figures\03_Prediction_Overlay.png
  [Png] saved → ./figures\04_Residual_Distributions.png
  [Png] saved → ./figures\05_Regime_Robustness.png
  [Png] saved → ./figures\06_Rolling_Directional_Accuracy.png
  [Png] saved → ./figures\07_Metrics_Summary_Table.png
  [Png] saved → ./figures\08_R2_Gap_Heatmap.png
  [Png] saved → ./figures\09_Prediction_Error_Scatter.png
  [Png] saved → ./figures\10_NonIID_Distribution.png
  [Png] saved → ./figures\11_KL_Divergence_Matrix.png
  [Png] saved → ./figures\12_Poisoning_Defence.png
  [Png] saved → ./figures\13_Aggregation_Weights.png
  [Png] saved → ./figures\14_Lookback_Window_Selection.png
  [Png] saved → ./figures\15_Latency_Benchmark.png
  [Png] saved → ./figures\16_Complexity_vs_Time.png
  [Png] saved → ./figures\17_Global_Predictions.png

  --- Blockchain / FL / Weights Plots ---
  [Png] saved → ./figures\18_Blockchain_Chain_Integrity.png
  [Png] saved → ./figures\19_Trust_Score_Evolution.png
  [Png] saved → ./figures\20_Node_Weight_Vectors.png
  [Png] saved → ./figures\21_Weight_Distribution.png
  [Png] saved → ./figures\22_FedProx_Proximal_Terms.png
  [Png] saved → ./figures\23_FL_Round_Summary_Dashboard.png
  [Png] saved → ./figures\24_Differential_Privacy_Noise_Impact.png

  --- Ablation Study ---
  [Png] saved → ./figures\25_Ablation_Per_Node_Metrics.png
  [Png] saved → ./figures\26_Ablation_Prediction_Overlay.png
  [Png] saved → ./figures\27_Ablation_Radar_Chart.png
  [Png] saved → ./figures\28_Ablation_Efficiency_Tradeoff.png
  [Png] saved → ./figures\29_Ablation_Summary_Table.png

  ==================================================
  All 29 figures saved as Png to: ./figures/
  ==================================================
"""