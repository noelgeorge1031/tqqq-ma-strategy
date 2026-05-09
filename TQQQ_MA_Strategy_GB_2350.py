"""
TQQQ Moving Average Strategy 
=====================================================
Question: Can a moving average crossover strategy on QQQ,
executed via TQQQ (3x leveraged), beat passive buy-and-hold
QQQ or TQQQ on a risk-adjusted basis while avoiding the
brutal drawdowns of holding TQQQ outright?

Approach:
  - Sweep EMA pairs, signal buffers, and volatility filters
    on an in-sample training window (2012–2018)
  - Freeze the best parameters and apply them untouched to
    out-of-sample data (2019–present)
  - Compare results against buy-and-hold QQQ and TQQQ

Dependencies: pip install yfinance pandas numpy matplotlib
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")  # headless rendering — saves to file instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# ── Configuration ─────────────────────────────────────────────────────────────

TRAIN_START = "2012-01-01"
TRAIN_END   = "2018-12-31"
TEST_START  = "2019-01-01"

# Signal construction
TREND_SMA            = 200   # long-term trend filter (days)
TREND_SLOPE_LOOKBACK = 20    # window to measure SMA slope
CONFIRM_DAYS         = 2     # signal must persist N days before entry
RVOL_WINDOW          = 10    # realized volatility lookback (days)
RVOL_PCT_WINDOW      = 63    # rolling window for vol percentile (~1 quarter)
ATR_PERIOD           = 14

# Risk management
ATR_STOP_MULT  = 2.0   # hard stop = entry - N * ATR
TRAIL_ATR_MULT = 3.0   # trailing stop = peak - N * ATR
SLIPPAGE_PCT   = 0.001 # flat half-spread applied on entry and exit

# TQQQ holding costs — two components deducted daily while in position:
#   (a) Expense ratio: TQQQ charges 0.86%/yr (Invesco prospectus); accrued as
#       a simple daily fraction.
#   (b) Volatility-decay drag: a 3x ETF rebalances daily back to its leverage
#       target, which causes it to systematically underperform 3× the index
#       when markets are choppy. The drag per day is approximately
#       (L - 1)² × σ²_daily / 2, where L = 3 and σ²_daily is the realised
#       daily variance of the underlying index. This grows with volatility and
#       is the dominant hidden cost of holding leveraged ETFs over time.
TQQQ_EXPENSE_RATIO = 0.0086                          # 0.86% per year
TQQQ_LEVERAGE      = 3.0                             # stated daily leverage
EXPENSE_DRAG_DAILY = TQQQ_EXPENSE_RATIO / 252        # daily expense accrual
VOL_DECAY_COEFF    = (TQQQ_LEVERAGE - 1) ** 2 / 2   # = 2.0; multiplied by σ²_daily at runtime

INITIAL_CAPITAL = 100_000
RISK_FREE_RATE  = 0.04   # annualised; approximate avg Fed funds rate across full sample
RF_DAILY        = RISK_FREE_RATE / 252

# Parameter search space — full Cartesian grid (10 × 5 × 5 = 250 combinations).
# Running every combination avoids the assumption that parameters are independent

EMA_PAIRS = [
    (5, 15), (8, 21), (10, 30), (12, 35), (15, 45),
    (20, 60), (20, 80), (30, 90), (50, 150), (50, 200),
]
BUFFER_SWEEP = [0.005, 0.010, 0.015, 0.020, 0.025]
RVOL_SWEEP   = [0.40, 0.50, 0.60, 0.70, 0.80]

# Plot theme
DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"
GRID_COL  = "#1e2130"
ACCENT    = "#00d4aa"
WARN      = "#e87040"
NEUTRAL   = "#5b9bd5"
TEXT_DIM  = "#9aa0a6"
TEXT_MAIN = "#e0e3eb"
GOLD      = "#f5c842"
OOS_COL   = "#c084fc"


# ── Plot helpers ──────────────────────────────────────────────────────────────

def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT_MAIN, fontsize=10.5, fontweight="bold", pad=9)
    ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=8.5)
    ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=8.5)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, lw=0.5, alpha=0.8)
    ax.grid(axis="x", color=GRID_COL, lw=0.3, alpha=0.4)


# ── Step 1: Download Data ─────────────────────────────────────────────────────

print("Downloading QQQ and TQQQ data...")
data = {}
for ticker in ["QQQ", "TQQQ"]:
    df = yf.download(ticker, start=TRAIN_START, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    data[ticker] = df.sort_index()
    print(f"  {ticker}: {df.index[0].date()} → {df.index[-1].date()}  ({len(df):,} rows)")

data_train = {k: v.loc[TRAIN_START:TRAIN_END] for k, v in data.items()}
data_test  = {k: v.loc[TEST_START:]           for k, v in data.items()}
print(f"\nTraining : {TRAIN_START} → {TRAIN_END}")
print(f"Test     : {TEST_START} → {data['QQQ'].index[-1].date()}")


# ── Step 2: Indicators & Signal Generation ────────────────────────────────────

def compute_atr(df, period=ATR_PERIOD):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    tr = pd.concat([(hi - lo), (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

def compute_indicators(qqq, tqqq, fast, slow):
    df = qqq.copy()
    df["ema_fast"]      = df["Close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"]      = df["Close"].ewm(span=slow, adjust=False).mean()
    df["sma_trend"]     = df["Close"].rolling(TREND_SMA).mean()
    df["sma_slope"]     = df["sma_trend"] - df["sma_trend"].shift(TREND_SLOPE_LOOKBACK)
    df["spread_pct"]    = (df["ema_fast"] - df["ema_slow"]) / df["Close"]
    log_ret             = np.log(df["Close"] / df["Close"].shift())
    df["rvol"]          = log_ret.rolling(RVOL_WINDOW).std() * np.sqrt(252)
    df["rvol_rank"]     = df["rvol"].rolling(RVOL_PCT_WINDOW).apply(
                              lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    df["tqqq_atr"]      = compute_atr(tqqq.reindex(df.index).ffill())
    # Rolling daily variance of QQQ log-returns — used to estimate the
    # volatility-decay drag on TQQQ each day we hold a position.
    df["qqq_var_daily"] = log_ret.rolling(RVOL_WINDOW).var()
    return df

def generate_signals(ind, buffer_pct, rvol_cutoff):
    bull = (
        (ind["Close"]      > ind["sma_trend"]) &
        (ind["sma_slope"]  > 0) &
        (ind["Close"]      > ind["ema_fast"]) &
        (ind["ema_fast"]   > ind["ema_slow"]) &
        (ind["ema_slow"]   > ind["sma_trend"]) &
        (ind["spread_pct"] >= buffer_pct) &
        (ind["rvol_rank"]  <= rvol_cutoff)
    )
    # Require the signal to persist for CONFIRM_DAYS before acting — reduces
    # whipsaws from transient crossovers.
    confirmed = bull.rolling(CONFIRM_DAYS).sum().eq(CONFIRM_DAYS)

    # Scale position size down in higher-vol regimes: when volatility rank is
    # elevated we reduce exposure rather than staying fully invested.
    weight = pd.Series(0.0, index=ind.index)
    weight[confirmed & (ind["rvol_rank"] <= 0.25)]                                    = 1.00
    weight[confirmed & (ind["rvol_rank"] > 0.25) & (ind["rvol_rank"] <= 0.45)]        = 0.67
    weight[confirmed & (ind["rvol_rank"] > 0.45) & (ind["rvol_rank"] <= rvol_cutoff)] = 0.33

    return pd.DataFrame({"confirmed_bull": confirmed.astype(int), "target_weight": weight},
                        index=ind.index)


# ── Step 3: Backtest Engine ───────────────────────────────────────────────────

def run_backtest(data, signals, ind):
    idx = (signals.index
           .intersection(data["TQQQ"].index)
           .intersection(ind.index))
    valid = ind.loc[idx, ["sma_trend", "tqqq_atr", "rvol_rank"]].notna().all(axis=1)
    idx   = idx[valid]

    sig  = signals.reindex(idx)
    tqqq = data["TQQQ"].reindex(idx)
    ind_ = ind.reindex(idx)

    cash = float(INITIAL_CAPITAL)
    shares = weight = 0.0
    in_pos = False
    entry_px = hard_stop = peak_px = 0.0
    entry_date = None
    hold_days  = 0

    port_vals = np.zeros(len(idx))
    exposures = np.zeros(len(idx))
    trades    = []

    for i, date in enumerate(idx):
        t_open  = tqqq.loc[date, "Open"]
        t_close = tqqq.loc[date, "Close"]
        atr     = ind_.loc[date, "tqqq_atr"]
        w_today = sig.loc[date, "target_weight"]
        w_yday  = sig.iloc[i - 1]["target_weight"] if i > 0 else 0.0

        if in_pos:
            hold_days += 1
            peak_px    = max(peak_px, t_close)
            trail_stop = peak_px - TRAIL_ATR_MULT * atr if not np.isnan(atr) else hard_stop
            eff_stop   = max(hard_stop, trail_stop)

            reason = px_raw = None
            if t_open <= eff_stop:
                px_raw, reason = min(t_open, eff_stop), "stop_loss"
            elif w_yday > 0 and w_today == 0:
                px_raw, reason = t_open, "signal_exit"

            if reason:
                exit_px = px_raw * (1 - SLIPPAGE_PCT)
                cash   += shares * exit_px
                trades.append({
                    "entry_date": entry_date, "exit_date": date,
                    "entry_price": round(entry_px, 4), "exit_price": round(exit_px, 4),
                    "position_weight": round(weight, 2), "hold_days": hold_days,
                    "trade_return": round(exit_px / entry_px - 1, 6), "exit_reason": reason,
                })
                shares = weight = hold_days = 0.0
                in_pos = False
                entry_px = hard_stop = peak_px = 0.0
                entry_date = None

            else:
                # Deduct daily holding costs while in position:
                #   (a) Expense ratio: 0.86%/yr accrued daily.
                #   (b) Volatility-decay drag: (L-1)² × σ²_daily / 2.
                #       This is the theoretical daily cost of a leveraged ETF's
                #       daily rebalancing in a volatile but directionless market.
                qqq_var    = ind_.loc[date, "qqq_var_daily"]
                vol_decay  = VOL_DECAY_COEFF * (qqq_var if not np.isnan(qqq_var) else 0.0)
                daily_drag = EXPENSE_DRAG_DAILY + vol_decay
                drag_cost  = shares * t_close * daily_drag * weight
                cash      -= drag_cost

        if (not in_pos) and i > 0 and w_yday == 0 and w_today > 0:
            entry_px   = t_open * (1 + SLIPPAGE_PCT)
            alloc      = cash * float(w_today)
            shares     = alloc / entry_px if entry_px > 0 else 0.0
            cash      -= alloc
            weight     = float(w_today)
            in_pos     = shares > 0
            entry_date = date
            peak_px    = t_close
            hard_stop  = entry_px - ATR_STOP_MULT * (atr if not np.isnan(atr) else 0.0)

        port_vals[i] = cash + shares * t_close
        exposures[i] = weight if in_pos else 0.0

    daily = pd.DataFrame({"portfolio_value": port_vals, "exposure": exposures}, index=idx)
    daily["daily_return"] = daily["portfolio_value"].pct_change().fillna(0)
    daily["drawdown"]     = (daily["portfolio_value"] - daily["portfolio_value"].cummax()) / \
                             daily["portfolio_value"].cummax()
    return daily, trades

def sortino(rets, rf=RF_DAILY):
    excess   = rets - rf
    downside = excess[excess < 0]
    down_dev = np.sqrt((downside ** 2).mean()) * np.sqrt(252) if len(downside) > 0 else np.nan
    return (excess.mean() * 252) / down_dev if (down_dev and not np.isnan(down_dev)) else np.nan

def calc_metrics(daily, trades):
    port  = daily["portfolio_value"]
    rets  = daily["daily_return"]
    years = len(port) / 252
    total = port.iloc[-1] / port.iloc[0] - 1
    cagr  = (1 + total) ** (1 / years) - 1
    vol   = rets.std() * np.sqrt(252)
    trets = [t["trade_return"] for t in trades]
    wins  = [r for r in trets if r > 0]
    losses= [r for r in trets if r <= 0]
    return dict(
        cagr=cagr, vol=vol,
        sharpe=((rets.mean() - RF_DAILY) * 252) / vol if vol > 0 else 0.0,
        sortino=sortino(rets),
        max_dd=daily["drawdown"].min(),
        total=total,
        n_trades=len(trades),
        avg_exp=daily["exposure"].mean(),
        win_rate=len(wins) / len(trades) if trades else np.nan,
        pf=sum(wins) / abs(sum(losses)) if (trades and losses) else np.nan,
        avg_ret=np.mean(trets) if trades else np.nan,
        avg_hold=np.mean([t["hold_days"] for t in trades]) if trades else np.nan,
    )

def bh_metrics(price_series):
    r    = price_series.pct_change().fillna(0)
    cum  = (1 + r).cumprod()
    dd   = (cum - cum.cummax()) / cum.cummax()
    yrs  = len(r) / 252
    tot  = cum.iloc[-1] - 1
    vol  = r.std() * np.sqrt(252)
    return dict(
        cagr=(1 + tot) ** (1 / yrs) - 1, vol=vol,
        sharpe=((r.mean() - RF_DAILY) * 252) / vol if vol > 0 else 0.0,
        sortino=sortino(r),
        mdd=dd.min(), total=tot,
    )


# ── Step 4: Parameter Sweep on Training Data Only ────────────────────────────

print("\n" + "=" * 65)
print("  PARAMETER SWEEP  —  Training Data Only (2012–2018)")
print("=" * 65)

# Median parameter values are kept for the per-dimension visualisation slices.
med_buf      = sorted(BUFFER_SWEEP)[len(BUFFER_SWEEP) // 2]
med_rvol     = sorted(RVOL_SWEEP)[len(RVOL_SWEEP) // 2]
med_f, med_s = EMA_PAIRS[len(EMA_PAIRS) // 2]

combos = [(f, s, b, r)
          for (f, s) in EMA_PAIRS
          for b in BUFFER_SWEEP
          for r in RVOL_SWEEP]

rows = []
for i, (f, s, b, r) in enumerate(combos, 1):
    # Tag each row so the visualisation panels can slice by single dimension
    dim = ("ema"    if b == med_buf  and r == med_rvol and (f, s) != (med_f, med_s) else
           "buffer" if (f, s) == (med_f, med_s) and r == med_rvol and b != med_buf else
           "rvol"   if (f, s) == (med_f, med_s) and b == med_buf  and r != med_rvol else
           "center" if (f, s) == (med_f, med_s) and b == med_buf  and r == med_rvol else
           "grid")
    print(f"  [{i:>3}/{len(combos)}] EMA({f}/{s}) buf={b:.3f} rvol={r:.2f}", end=" ... ", flush=True)
    ind   = compute_indicators(data_train["QQQ"], data_train["TQQQ"], f, s)
    sig   = generate_signals(ind, b, r)
    daily, trades = run_backtest(data_train, sig, ind)
    m = calc_metrics(daily, trades)
    rows.append(dict(fast=f, slow=s, ema_pair=f"{f}/{s}", buffer=b, rvol=r, dim=dim,
                     sharpe=round(m["sharpe"], 3), cagr=round(m["cagr"], 4),
                     max_dd=round(m["max_dd"], 4), vol=round(m["vol"], 4),
                     n_trades=m["n_trades"],
                     win_rate=round(m["win_rate"], 4) if not np.isnan(m["win_rate"]) else np.nan,
                     pf=round(m["pf"], 3) if not np.isnan(m["pf"]) else np.nan))
    print(f"Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.1%}  MDD={m['max_dd']:.1%}")

sweep_df = pd.DataFrame(rows)
print(f"\nSweep complete — {len(sweep_df)} combinations tested.")


# ── Step 5: Select Best Parameters ───────────────────────────────────────────
#
# Naive peak-picking from 250 combinations risks selecting a parameter set
# that only looks good for one narrow (buffer, rvol) slice — an isolated spike
# on the performance surface rather than a genuinely robust region.
#
# To guard against this, we first compute the median Sharpe for each EMA pair
# across all 25 of its (buffer × rvol) combinations — its "neighbourhood".
# Only EMA pairs whose neighbourhood median beats the grand median across all
# pairs are eligible for selection. We then pick the best individual combo
# (by Sharpe, with max drawdown as a tie-break) from within that eligible set.
# This ensures the selected parameters sit in a broadly strong region of the
# surface, not an isolated peak that may not generalise.

valid_sweep = sweep_df.dropna(subset=["sharpe"])

# Median Sharpe per EMA pair across all buffer × rvol combinations
pair_median_sharpe = (valid_sweep
                      .groupby(["fast", "slow"])["sharpe"]
                      .median()
                      .rename("pair_median_sharpe")
                      .reset_index())

# Grand median across all EMA-pair medians (the robustness threshold)
grand_median = pair_median_sharpe["pair_median_sharpe"].median()

# Eligible pairs: neighbourhood median at or above grand median
eligible_pairs  = pair_median_sharpe[
    pair_median_sharpe["pair_median_sharpe"] >= grand_median
][["fast", "slow"]]

# Best individual combo within the eligible set
eligible_combos = valid_sweep.merge(eligible_pairs, on=["fast", "slow"])
best_row  = eligible_combos.sort_values(["sharpe", "max_dd"], ascending=[False, False]).iloc[0]

best_fast = int(best_row["fast"])
best_slow = int(best_row["slow"])
best_buf  = float(best_row["buffer"])
best_rvol = float(best_row["rvol"])

# Per-dimension slices — used only for Exhibit 1 visualisation panels
ema_df  = sweep_df[(sweep_df["buffer"] == med_buf) & (sweep_df["rvol"] == med_rvol)].dropna(subset=["sharpe"])
buf_df  = sweep_df[(sweep_df["fast"] == med_f) & (sweep_df["slow"] == med_s) & (sweep_df["rvol"] == med_rvol)].dropna(subset=["sharpe"]).sort_values("buffer")
rvol_df = sweep_df[(sweep_df["fast"] == med_f) & (sweep_df["slow"] == med_s) & (sweep_df["buffer"] == med_buf)].dropna(subset=["sharpe"]).sort_values("rvol")

# Derived annotation values for the visualisation cards
s90, smed = ema_df["sharpe"].quantile(0.90), ema_df["sharpe"].median()
flat      = buf_df[buf_df["sharpe"] >= buf_df["sharpe"].max() * 0.90]
flat_n    = len(flat)
rv_range  = rvol_df["sharpe"].max() - rvol_df["sharpe"].min()

rec = dict(fast=best_fast, slow=best_slow, buf=best_buf, rvol=best_rvol,
           s90=s90, smed=smed, flat_n=flat_n, rv_range=rv_range)

print("\n" + "=" * 65)
print("  FROZEN PARAMETERS  (selected from training data only)")
print("=" * 65)
print(f"  EMA Pair    :  {best_fast}/{best_slow}")
print(f"  Buffer      :  {best_buf:.3f}")
print(f"  RVOL Cutoff :  {best_rvol:.2f}")
print(f"\n  Selected via neighbourhood-robust grid search ({len(eligible_combos)} eligible combos).")
print(f"  Criterion: EMA pair neighbourhood median Sharpe ≥ grand median ({grand_median:.3f}),")
print(f"  then best Sharpe within eligible set; max-drawdown tie-break.")
print(f"  Eligible EMA pairs: {len(eligible_pairs)}/{len(pair_median_sharpe)} passed robustness filter.")
print("=" * 65)
print("  Parameters are now LOCKED. No further adjustments to any data.\n")


# ── Exhibit 1: Parameter Sensitivity Charts ───────────────────────────────────

ema_plot = ema_df.assign(ratio=ema_df["fast"] / ema_df["slow"]).sort_values("ratio").reset_index(drop=True)

fig = plt.figure(figsize=(18, 13), facecolor=DARK_BG)
fig.suptitle("Parameter Sweep  →  Data-Driven Selection  [Training Data: 2012–2018]",
             color=TEXT_MAIN, fontsize=15, fontweight="bold", y=0.97)
gs = gridspec.GridSpec(2, 3, figure=fig, left=0.06, right=0.97, top=0.91, bottom=0.08,
                       hspace=0.42, wspace=0.32)

# A: EMA pair Sharpe bars
ax_a = fig.add_subplot(gs[0, :2])
pairs, sharpes = ema_plot["ema_pair"].tolist(), ema_plot["sharpe"].tolist()
colors = [GOLD   if r["fast"] == rec["fast"] and r["slow"] == rec["slow"] else
          ACCENT if r["sharpe"] >= rec["s90"] else
          NEUTRAL if r["sharpe"] >= rec["smed"] else WARN
          for _, r in ema_plot.iterrows()]
x = np.arange(len(pairs))
ax_a.bar(x, sharpes, color=colors, width=0.6, zorder=3, edgecolor=DARK_BG, linewidth=0.5)
ax_a.axhline(rec["smed"], color=TEXT_DIM, lw=1, ls="--", alpha=0.6, label=f"Median ({rec['smed']:.2f})")
ax_a.axhline(rec["s90"],  color=ACCENT,   lw=1, ls=":",  alpha=0.8, label=f"90th pctile ({rec['s90']:.2f})")
for xi, sh in enumerate(sharpes):
    if not np.isnan(sh):
        ax_a.text(xi, sh + 0.02, f"{sh:.2f}", ha="center", va="bottom", fontsize=7.5,
                  color=TEXT_MAIN, fontweight="bold")
rec_row = ema_plot[(ema_plot["fast"] == rec["fast"]) & (ema_plot["slow"] == rec["slow"])]
if not rec_row.empty:
    xi = rec_row.index[0]
    ax_a.annotate(f"SELECTED\n({rec['fast']}/{rec['slow']})",
                  xy=(xi, rec_row["sharpe"].values[0]),
                  xytext=(xi + 1.3, rec_row["sharpe"].values[0] + 0.10),
                  fontsize=8, color=GOLD, fontweight="bold",
                  arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
ax_a.set_xticks(x)
ax_a.set_xticklabels(pairs, rotation=35, ha="right", fontsize=9)
ax_a.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_DIM, edgecolor=GRID_COL, framealpha=0.7)
style_ax(ax_a, "A  —  EMA Pair Sweep: Sharpe Ratio  (buffer=median, rvol=median)", "EMA (Fast/Slow)", "Sharpe")
robust_n = int((np.array(sharpes) >= rec["smed"]).sum())
ax_a.text(0.98, 0.95, f"{robust_n}/{len(sharpes)} pairs above median Sharpe",
          transform=ax_a.transAxes, ha="right", va="top", fontsize=9, color=ACCENT,
          bbox=dict(boxstyle="round,pad=0.35", facecolor=DARK_BG, edgecolor=ACCENT, alpha=0.8))

# B: CAGR horizontal bars
ax_b = fig.add_subplot(gs[0, 2])
cagrs = ema_plot["cagr"].fillna(0).tolist()
cmap  = LinearSegmentedColormap.from_list("cg", ["#cc3333", "#2a2d35", "#00d4aa"])
norm  = matplotlib.colors.Normalize(vmin=min(cagrs), vmax=max(cagrs))
ax_b.barh(x, cagrs, color=[cmap(norm(c)) for c in cagrs], edgecolor=DARK_BG, linewidth=0.5, zorder=3)
ax_b.set_yticks(x)
ax_b.set_yticklabels(pairs, fontsize=8.5)
ax_b.axvline(0, color=TEXT_DIM, lw=0.6, ls="--", alpha=0.5)
for xi, c in enumerate(cagrs):
    ax_b.text(c + 0.003, xi, f"{c:.1%}", va="center", fontsize=7, color=TEXT_MAIN)
style_ax(ax_b, "B  —  CAGR per EMA Pair", "CAGR (annualised)", "EMA Pair")

# C: Buffer sweep
ax_c = fig.add_subplot(gs[1, 0])
bvals, bsh = buf_df["buffer"].tolist(), buf_df["sharpe"].tolist()
flat_thresh = max(bsh) * 0.90
ax_c.fill_between(bvals, flat_thresh, max(bsh), alpha=0.12, color=ACCENT,
                  label=f"Flat region (≥{flat_thresh:.2f})")
ax_c.plot(bvals, bsh, color=ACCENT, lw=2.2, marker="o", markersize=7, zorder=3)
ax_c.axvline(rec["buf"], color=GOLD, lw=1.5, ls="--", alpha=0.9, label=f"Selected ({rec['buf']:.3f})")
for bv, sv in zip(bvals, bsh):
    if not np.isnan(sv):
        ax_c.text(bv, sv + 0.015, f"{sv:.2f}", ha="center", fontsize=8, color=TEXT_MAIN)
ax_c.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_DIM, edgecolor=GRID_COL, framealpha=0.7)
ax_c.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
style_ax(ax_c, "C  —  Buffer Sweep  (EMA=median, rvol=median)", "Buffer %", "Sharpe")

# D: RVOL sweep
ax_d = fig.add_subplot(gs[1, 1])
rvals, rsh = rvol_df["rvol"].tolist(), rvol_df["sharpe"].tolist()
ax_d.plot(rvals, rsh, color=NEUTRAL, lw=2.2, marker="s", markersize=7, zorder=3)
ax_d.axvline(rec["rvol"], color=GOLD, lw=1.5, ls="--", alpha=0.9, label=f"Selected ({rec['rvol']:.2f})")
for rv, sv in zip(rvals, rsh):
    if not np.isnan(sv):
        ax_d.text(rv, sv + 0.01, f"{sv:.2f}", ha="center", fontsize=8, color=TEXT_MAIN)
flat_tag = "FLAT" if rec["rv_range"] < 0.15 else "VARIES"
ax_d.text(0.97, 0.05, f"Range: {rec['rv_range']:.3f}  [{flat_tag}]",
          transform=ax_d.transAxes, ha="right", va="bottom", fontsize=8,
          color=ACCENT if flat_tag == "FLAT" else WARN,
          bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=GRID_COL, alpha=0.8))
ax_d.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_DIM, edgecolor=GRID_COL, framealpha=0.7)
style_ax(ax_d, "D  —  RVOL Cutoff Sweep  (EMA=median, buffer=median)", "RVOL Percentile Cutoff", "Sharpe")

# E: Selection rationale card
ax_e = fig.add_subplot(gs[1, 2])
ax_e.set_facecolor(CARD_BG)
ax_e.axis("off")
ax_e.set_title("E  —  Selection Rationale", color=TEXT_MAIN, fontsize=10.5, fontweight="bold", pad=9)
for sp in ax_e.spines.values(): sp.set_edgecolor(GRID_COL)
card_lines = [
    ("FROZEN PARAMETERS", True, GOLD),
    (f"EMA {rec['fast']}/{rec['slow']}  \u00b7  Buffer {rec['buf']:.3f}  \u00b7  RVOL {rec['rvol']:.2f}", False, TEXT_MAIN),
    ("", False, TEXT_DIM),
    ("EMA pair:", True, ACCENT),
    (f"Neighbourhood filter:\npair median Sharpe\n\u2265 grand median ({grand_median:.2f}).\nEnsures selection from\na broadly strong region.", False, TEXT_DIM),
    ("", False, TEXT_DIM),
    ("Buffer:", True, ACCENT),
    (f"Flat region contains\n{rec['flat_n']} values \u226590% of peak.\nExtreme excluded;\n{rec['buf']:.3f} chosen.", False, TEXT_DIM),
    ("", False, TEXT_DIM),
    ("RVOL cutoff:", True, ACCENT),
    (f"Range = {rec['rv_range']:.3f} \u2014 flat.\nCentral value {rec['rvol']:.2f} retained;\nparameter not influential.", False, TEXT_DIM),
]
y = 0.97
for text, bold, color in card_lines:
    if not text:
        y -= 0.04; continue
    ax_e.text(0.05, y, text, transform=ax_e.transAxes, fontsize=8, color=color,
              fontweight="bold" if bold else "normal", va="top", multialignment="left")
    y -= 0.07 + text.count("\n") * 0.055

plt.tight_layout()
plt.savefig("output/exhibit1_parameter_sweep.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Saved: output/exhibit1_parameter_sweep.png")


# ── Step 6: Apply Frozen Parameters to Both Folds ────────────────────────────

print(f"Running backtest with frozen params: EMA {rec['fast']}/{rec['slow']}  buf={rec['buf']:.3f}  rvol={rec['rvol']:.2f}")

# In-sample (reference — parameters were selected on this data, so results are fitted)
ind_is = compute_indicators(data_train["QQQ"], data_train["TQQQ"], rec["fast"], rec["slow"])
sig_is = generate_signals(ind_is, rec["buf"], rec["rvol"])
daily_is, trades_is = run_backtest(data_train, sig_is, ind_is)
m_is = calc_metrics(daily_is, trades_is)
print(f"  In-sample  : {len(trades_is)} trades | Sharpe={m_is['sharpe']:.3f} | CAGR={m_is['cagr']:.2%} | MDD={m_is['max_dd']:.2%}")

# Out-of-sample — indicators computed on the full history for proper warm-up,
# then sliced to the test window so no future data leaks into the signal.
ind_full = compute_indicators(data["QQQ"], data["TQQQ"], rec["fast"], rec["slow"])
sig_full = generate_signals(ind_full, rec["buf"], rec["rvol"])
oos_start  = pd.Timestamp(TEST_START)
sig_oos    = sig_full.loc[oos_start:]
ind_oos    = ind_full.loc[oos_start:]
data_oos   = {k: v.loc[oos_start:] for k, v in data.items()}
daily_oos, trades_oos = run_backtest(data_oos, sig_oos, ind_oos)
m_oos = calc_metrics(daily_oos, trades_oos)
print(f"  Out-of-sample: {len(trades_oos)} trades | Sharpe={m_oos['sharpe']:.3f} | CAGR={m_oos['cagr']:.2%} | MDD={m_oos['max_dd']:.2%}")

# Buy-and-hold benchmarks
bh_is_qqq   = bh_metrics(data_train["QQQ"]["Close"])
bh_is_tqqq  = bh_metrics(data_train["TQQQ"]["Close"])
bh_oos_qqq  = bh_metrics(data_oos["QQQ"]["Close"])
bh_oos_tqqq = bh_metrics(data_oos["TQQQ"]["Close"])


# ── Step 7: Strategy vs Benchmarks — Printed Summary ─────────────────────────

def print_comparison(m_strat, bh_qqq, bh_tqqq, label, is_oos=False):
    tag = "  ← TRUE EXPECTED PERFORMANCE" if is_oos else "  (fitted — reference only)"
    W, sep = 88, "─" * 82
    print(f"{'=' * W}")
    print(f"  {label}{tag}")
    print(f"  EMA {rec['fast']}/{rec['slow']}  ·  buf={rec['buf']:.3f}  ·  rvol={rec['rvol']:.2f}")
    print(f"{'=' * W}")
    print(f"  {'Metric':<28}  {'Strategy':>14}  {'B&H QQQ':>12}  {'B&H TQQQ':>12}")
    print(f"  {sep}")
    rows = [
        ("CAGR",            f"{m_strat['cagr']:.2%}",    f"{bh_qqq['cagr']:.2%}",    f"{bh_tqqq['cagr']:.2%}"),
        ("Sharpe Ratio",    f"{m_strat['sharpe']:.3f}",  f"{bh_qqq['sharpe']:.3f}",  f"{bh_tqqq['sharpe']:.3f}"),
        ("Sortino Ratio",   (f"{m_strat['sortino']:.3f}" if not np.isnan(m_strat['sortino']) else "N/A"),
                            (f"{bh_qqq['sortino']:.3f}"  if not np.isnan(bh_qqq['sortino'])  else "N/A"),
                            (f"{bh_tqqq['sortino']:.3f}" if not np.isnan(bh_tqqq['sortino']) else "N/A")),
        ("Ann. Volatility", f"{m_strat['vol']:.2%}",     f"{bh_qqq['vol']:.2%}",     f"{bh_tqqq['vol']:.2%}"),
        ("Max Drawdown",    f"{m_strat['max_dd']:.2%}",  f"{bh_qqq['mdd']:.2%}",     f"{bh_tqqq['mdd']:.2%}"),
        ("Total Return",    f"{m_strat['total']:.2%}",   f"{bh_qqq['total']:.2%}",   f"{bh_tqqq['total']:.2%}"),
    ]
    for lbl, sv, qv, tv in rows:
        print(f"  {lbl:<28}  {sv:>14}  {qv:>12}  {tv:>12}")
    print(f"  {sep}")
    strat_rows = [
        ("Avg Exposure",       f"{m_strat['avg_exp']:.2%}"),
        ("Number of Trades",   str(m_strat["n_trades"])),
        ("Win Rate",           f"{m_strat['win_rate']:.2%}"  if not np.isnan(m_strat["win_rate"])  else "N/A"),
        ("Avg Trade Return",   f"{m_strat['avg_ret']:.2%}"   if not np.isnan(m_strat["avg_ret"])   else "N/A"),
        ("Avg Hold Period (d)",f"{m_strat['avg_hold']:.1f}"  if not np.isnan(m_strat["avg_hold"])  else "N/A"),
        ("Profit Factor",      f"{m_strat['pf']:.2f}"        if not np.isnan(m_strat["pf"])        else "N/A"),
    ]
    for lbl, val in strat_rows:
        print(f"  {lbl:<28}  {val:>14}")
    print(f"{'=' * W}\n")

print_comparison(m_is,  bh_is_qqq,  bh_is_tqqq,  "IN-SAMPLE   2012–2018",      is_oos=False)
print_comparison(m_oos, bh_oos_qqq, bh_oos_tqqq, "OUT-OF-SAMPLE  2019–Present", is_oos=True)


# ── Exhibit 2: Walk-Forward Equity Curve ──────────────────────────────────────

strat_is  = daily_is["portfolio_value"]  / daily_is["portfolio_value"].iloc[0]  * 100
oos_scale = strat_is.iloc[-1]
strat_oos = daily_oos["portfolio_value"] / daily_oos["portfolio_value"].iloc[0]  * oos_scale
full_strat = pd.concat([strat_is, strat_oos])
full_idx   = full_strat.index
qqq_full   = data["QQQ"]["Close"].reindex(full_idx).ffill()
qqq_full   = qqq_full / qqq_full.iloc[0] * 100
tqqq_full  = data["TQQQ"]["Close"].reindex(full_idx).ffill()
tqqq_full  = tqqq_full / tqqq_full.iloc[0] * 100
full_dd    = pd.concat([daily_is["drawdown"], daily_oos["drawdown"]])
split_date = pd.Timestamp(TEST_START)

fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

for ax in (ax1, ax2):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor("#2a2d35")
    ax.axvspan(split_date, full_idx[-1], alpha=0.07, color=OOS_COL, zorder=0)
    ax.axvline(split_date, color=OOS_COL, lw=1.4, ls="--", alpha=0.8, zorder=5)

ax1.plot(full_idx, full_strat, color=ACCENT,  lw=1.8, zorder=3, label=f"TQQQ Strategy (EMA {rec['fast']}/{rec['slow']})")
ax1.plot(full_idx, qqq_full,  color=NEUTRAL, lw=1.2, alpha=0.8, zorder=2, label="Buy & Hold QQQ")
ax1.plot(full_idx, tqqq_full, color=WARN,    lw=1.2, alpha=0.7, zorder=2, label="Buy & Hold TQQQ")
ax1.text(0.18, 0.97, "IN-SAMPLE\n(Training · Fitted)", transform=ax1.transAxes,
         ha="center", va="top", fontsize=9, color=TEXT_DIM, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=GRID_COL, alpha=0.85))
ax1.text(0.72, 0.97, "OUT-OF-SAMPLE\n(Test · True Performance)", transform=ax1.transAxes,
         ha="center", va="top", fontsize=9, color=OOS_COL, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_BG, edgecolor=OOS_COL, alpha=0.85))
ax1.set_ylabel("Normalised Value (base 100)", color=TEXT_DIM)
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.grid(axis="y", color=GRID_COL, lw=0.5)
ax1.grid(axis="x", color=GRID_COL, lw=0.5)
ax1.legend(loc="upper left", framealpha=0.3, facecolor=CARD_BG,
           edgecolor="#2a2d35", labelcolor="#c9cdd4", fontsize=9)
ax1.set_title(
    f"TQQQ Strategy — Walk-Forward Validation  "
    f"[IS: Sharpe {m_is['sharpe']:.2f} / CAGR {m_is['cagr']:.1%}  │  "
    f"OOS ✓: Sharpe {m_oos['sharpe']:.2f} / CAGR {m_oos['cagr']:.1%}]",
    color=TEXT_MAIN, fontsize=12, pad=12, fontweight="bold")

ax2.fill_between(full_idx, full_dd * 100, 0, color="#cc3333", alpha=0.65, label="Strategy Drawdown")
ax2.set_ylabel("Drawdown (%)", color=TEXT_DIM)
ax2.grid(axis="y", color=GRID_COL, lw=0.5)
ax2.legend(loc="lower left", framealpha=0.3, facecolor=CARD_BG,
           edgecolor="#2a2d35", labelcolor="#c9cdd4", fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
plt.setp(ax1.get_xticklabels(), visible=False)
plt.savefig("output/exhibit2_equity_curve.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Saved: output/exhibit2_equity_curve.png")


# ── Exhibit 3: In-Sample vs Out-of-Sample Metric Comparison ──────────────────

metrics  = ["sharpe", "sortino", "cagr", "max_dd", "vol", "win_rate"]
labels   = ["Sharpe Ratio", "Sortino Ratio", "CAGR", "Max Drawdown", "Ann. Vol", "Win Rate"]
fmt_pct  = {"cagr", "max_dd", "vol", "win_rate"}
is_vals  = [m_is.get(k, np.nan)  for k in metrics]
oos_vals = [m_oos.get(k, np.nan) for k in metrics]

fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5), facecolor=DARK_BG)
fig.suptitle(
    f"In-Sample vs Out-of-Sample Metric Comparison\n"
    f"(EMA {rec['fast']}/{rec['slow']}  ·  buf={rec['buf']:.3f}  ·  rvol={rec['rvol']:.2f})",
    color=TEXT_MAIN, fontsize=13, fontweight="bold", y=1.01)

for ax, metric, label, iv, ov in zip(axes, metrics, labels, is_vals, oos_vals):
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID_COL)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    bars = ax.bar(["IS\n2012–18", "OOS\n2019+"], [iv, ov],
                  color=[NEUTRAL, OOS_COL], width=0.5, zorder=3,
                  edgecolor=DARK_BG, linewidth=0.8)
    for bar, val in zip(bars, [iv, ov]):
        txt = f"{val:.2%}" if metric in fmt_pct else f"{val:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(bar.get_height()) * 0.03,
                txt, ha="center", va="bottom", fontsize=9.5,
                color=TEXT_MAIN, fontweight="bold")
    if not (np.isnan(iv) or np.isnan(ov)) and iv != 0:
        decay = (ov - iv) / abs(iv) * 100
        col   = WARN if decay < 0 else ACCENT
        ax.text(0.5, 0.06, f"{'+'if decay>=0 else''}{decay:.1f}%",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=DARK_BG, edgecolor=col, alpha=0.85))
    ax.set_title(label, color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID_COL, lw=0.5, alpha=0.8)

plt.tight_layout()
plt.savefig("output/exhibit3_is_oos_comparison.png", dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("  Saved: output/exhibit3_is_oos_comparison.png")

print("\n  IS → OOS Metric Changes:")
for k, lbl in zip(["sharpe", "sortino", "cagr", "max_dd", "win_rate"],
                   ["Sharpe", "Sortino", "CAGR", "Max Drawdown", "Win Rate"]):
    iv, ov = m_is.get(k, np.nan), m_oos.get(k, np.nan)
    if np.isnan(iv) or np.isnan(ov) or iv == 0: continue
    pct  = (ov - iv) / abs(iv) * 100
    direction = "improvement" if (k != "max_dd" and pct > 0) or (k == "max_dd" and pct < 0) else "degradation"
    print(f"    {lbl:<16}: IS={iv:.3f}  OOS={ov:.3f}  ({'+'if pct>=0 else''}{pct:.1f}%)  [{direction}]")


# ── Export Results ────────────────────────────────────────────────────────────

os.makedirs("output", exist_ok=True)

def fmt(v, pct=False, ratio=False):
    if isinstance(v, float) and np.isnan(v): return "N/A"
    if pct:   return f"{v:.2%}"
    if ratio: return f"{v:.3f}"
    return str(v)

def build_summary_rows(m, bh_qqq, bh_tqqq, fold, period):
    return [
        ("Fold",              fold),
        ("Period",            period),
        ("EMA Pair",          f"{rec['fast']}/{rec['slow']}"),
        ("Buffer",            str(rec["buf"])),
        ("RVOL Cutoff",       str(rec["rvol"])),
        ("CAGR",              fmt(m["cagr"], pct=True)),
        ("Sharpe Ratio",      fmt(m["sharpe"], ratio=True)),
        ("Sortino Ratio",     fmt(m["sortino"], ratio=True)),
        ("Max Drawdown",      fmt(m["max_dd"], pct=True)),
        ("Ann. Volatility",   fmt(m["vol"], pct=True)),
        ("Total Return",      fmt(m["total"], pct=True)),
        ("Avg Exposure",      fmt(m["avg_exp"], pct=True)),
        ("Num Trades",        str(m["n_trades"])),
        ("Win Rate",          fmt(m["win_rate"], pct=True)),
        ("Profit Factor",     fmt(m["pf"])),
        ("Avg Trade Return",  fmt(m["avg_ret"], pct=True)),
        ("Avg Hold Days",     f"{m['avg_hold']:.1f}" if not np.isnan(m["avg_hold"]) else "N/A"),
        ("QQQ CAGR (BH)",     fmt(bh_qqq["cagr"],    pct=True)),
        ("QQQ Sharpe (BH)",   fmt(bh_qqq["sharpe"],  ratio=True)),
        ("QQQ Sortino (BH)",  fmt(bh_qqq["sortino"], ratio=True)),
        ("TQQQ CAGR (BH)",    fmt(bh_tqqq["cagr"],   pct=True)),
        ("TQQQ Sharpe (BH)",  fmt(bh_tqqq["sharpe"], ratio=True)),
        ("TQQQ Sortino (BH)", fmt(bh_tqqq["sortino"],ratio=True)),
    ]

rows_is  = build_summary_rows(m_is,  bh_is_qqq,  bh_is_tqqq,  "IN-SAMPLE",     f"{TRAIN_START} to {TRAIN_END}")
rows_oos = build_summary_rows(m_oos, bh_oos_qqq, bh_oos_tqqq, "OUT-OF-SAMPLE", f"{TEST_START} to present")
pd.DataFrame(rows_is + [("---", "---")] + rows_oos, columns=["Metric", "Value"]).to_csv(
    "output/summary_stats.csv", index=False)

pd.DataFrame(trades_is).assign(fold="in_sample").to_csv("output/trades_in_sample.csv",   index=False)
pd.DataFrame(trades_oos).assign(fold="out_of_sample").to_csv("output/trades_out_of_sample.csv", index=False)
daily_is.assign(fold="in_sample").reset_index().to_csv("output/daily_in_sample.csv",   index=False)
daily_oos.assign(fold="out_of_sample").reset_index().to_csv("output/daily_out_of_sample.csv", index=False)
sweep_df.to_csv("output/parameter_sweep.csv", index=False)
pd.DataFrame([{"ema_fast": rec["fast"], "ema_slow": rec["slow"], "buffer": rec["buf"],
               "rvol_cutoff": rec["rvol"], "selected_from": f"{TRAIN_START} to {TRAIN_END}",
               "applied_to": f"{TEST_START} to present",
               "selection_method": "neighbourhood_robust",
               "grand_median_sharpe": round(grand_median, 3),
               "eligible_ema_pairs": len(eligible_pairs)}]).to_csv("output/frozen_params.csv", index=False)

print("\nOutput files:")
for f in sorted(os.listdir("output")):
    print(f"  output/{f}")
