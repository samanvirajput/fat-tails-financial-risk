"""
Fat Tails, Volatility Clustering & Risk Underestimation in Stock Returns
========================================================================
"Beyond Normality: A Comparative Statistical Analysis of Fat Tails,
Volatility Clustering, and Risk Underestimation in Stock Market Returns"

Author  : Samanvi Rajput
Affil.  : School of Computer Science and Engineering, VIT Vellore
Target  : arXiv q-fin.ST / stat.AP

Methodology
-----------
1. Moment analysis (skewness, excess kurtosis)
2. Shapiro–Wilk + Kolmogorov–Smirnov normality tests
3. MLE: Normal vs Student-t; AIC-based model selection
4. Ljung–Box test on squared returns (ARCH / volatility clustering)
5. Parametric, empirical, and Student-t Value-at-Risk @ 95/99/99.9%
6. Six publication-quality figures
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    import yfinance as yf
except ImportError:
    sys.exit("Install yfinance: pip install yfinance")

# ── Global aesthetics ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

C = {
    "AAPL": "#1565C0",
    "TSLA": "#B71C1C",
    "normal": "#757575",
    "student": "#E65100",
    "accent": "#2E7D32",
}

TICKERS   = ["AAPL", "TSLA"]
START     = "2015-01-01"
END       = "2024-12-31"
NOTIONAL  = 1_000_000


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════════════════════

def fetch_log_returns(tickers=TICKERS, start=START, end=END) -> pd.DataFrame:
    print("Downloading data via yfinance …")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    ret = np.log(raw / raw.shift(1)).dropna()
    print(f"  {len(ret):,} trading days  "
          f"({ret.index[0].date()} → {ret.index[-1].date()})\n")
    return ret


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MOMENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def moment_analysis(returns: pd.DataFrame) -> dict:
    out = {}
    print("── Moment Analysis " + "─" * 42)
    for t in TICKERS:
        r = returns[t].values
        m = {
            "n":               len(r),
            "mean":            float(np.mean(r)),
            "std":             float(np.std(r, ddof=1)),
            "skewness":        float(stats.skew(r)),
            "excess_kurtosis": float(stats.kurtosis(r)),   # Fisher (excess)
            "min":             float(r.min()),
            "max":             float(r.max()),
        }
        out[t] = m
        print(f"  {t}: n={m['n']:,}  mean={m['mean']:.5f}  σ={m['std']:.5f}  "
              f"skew={m['skewness']:.4f}  kurt={m['excess_kurtosis']:.4f}")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  NORMALITY TESTS
# ══════════════════════════════════════════════════════════════════════════════

def normality_tests(returns: pd.DataFrame) -> dict:
    out = {}
    print("── Normality Tests " + "─" * 42)
    for t in TICKERS:
        r = returns[t].values
        # Shapiro–Wilk (capped at 5 000 for runtime)
        sw_stat, sw_p = stats.shapiro(r[:5000])
        # Kolmogorov–Smirnov
        ks_stat, ks_p = stats.kstest(r, "norm", args=(np.mean(r), np.std(r, ddof=1)))
        out[t] = {"sw_stat": sw_stat, "sw_p": sw_p, "ks_stat": ks_stat, "ks_p": ks_p}
        print(f"  {t}: SW W={sw_stat:.6f} p={sw_p:.2e}  |  "
              f"KS D={ks_stat:.6f} p={ks_p:.2e}")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MLE + AIC
# ══════════════════════════════════════════════════════════════════════════════

def _fit_normal(r: np.ndarray) -> dict:
    mu, sigma = np.mean(r), np.std(r, ddof=1)
    ll = float(np.sum(stats.norm.logpdf(r, mu, sigma)))
    k  = 2
    return {"mu": mu, "sigma": sigma, "log_lik": ll,
            "aic": 2*k - 2*ll, "bic": k*np.log(len(r)) - 2*ll}


def _neg_loglik_t(params, r):
    nu, mu, sigma = params
    if nu <= 2.01 or sigma <= 1e-10:
        return 1e12
    return -float(np.sum(stats.t.logpdf(r, df=nu, loc=mu, scale=sigma)))


def _fit_student_t(r: np.ndarray) -> dict:
    mu0, sig0 = np.mean(r), np.std(r, ddof=1)
    best, best_val = None, np.inf
    for nu0 in [3.0, 5.0, 8.0, 15.0]:
        res = minimize(_neg_loglik_t, x0=[nu0, mu0, sig0], args=(r,),
                       method="Nelder-Mead",
                       options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 20_000})
        if res.fun < best_val:
            best_val, best = res.fun, res
    nu, mu, sigma = best.x
    ll = -best_val
    k  = 3
    return {"nu": nu, "mu": mu, "sigma": sigma, "log_lik": ll,
            "aic": 2*k - 2*ll, "bic": k*np.log(len(r)) - 2*ll}


def fit_distributions(returns: pd.DataFrame) -> dict:
    out = {}
    print("── MLE: Normal vs Student-t " + "─" * 33)
    for t in TICKERS:
        r  = returns[t].values
        nf = _fit_normal(r)
        tf = _fit_student_t(r)
        delta_aic = nf["aic"] - tf["aic"]
        out[t] = {"normal": nf, "student_t": tf, "delta_aic": delta_aic}
        print(f"  {t}: Normal AIC={nf['aic']:.1f}  Student-t AIC={tf['aic']:.1f}  "
              f"ΔAIC={delta_aic:.1f}  ν̂={tf['nu']:.3f}")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 5.  VOLATILITY CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

def volatility_clustering(returns: pd.DataFrame) -> dict:
    out = {}
    print("── Volatility Clustering (Ljung–Box on r²) " + "─" * 18)
    for t in TICKERS:
        r  = returns[t].values
        r2 = r ** 2
        lb = acorr_ljungbox(r2, lags=[10, 20], return_df=True)
        out[t] = {"r2": r2, "lb": lb}
        print(f"  {t}: Q(10)={lb['lb_stat'].iloc[0]:.1f} p={lb['lb_pvalue'].iloc[0]:.2e}  |  "
              f"Q(20)={lb['lb_stat'].iloc[1]:.1f} p={lb['lb_pvalue'].iloc[1]:.2e}")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VALUE AT RISK
# ══════════════════════════════════════════════════════════════════════════════

CLS = [0.95, 0.99, 0.999]


def compute_var(returns: pd.DataFrame, fits: dict, notional: float = NOTIONAL) -> dict:
    out = {}
    print("── Value at Risk " + "─" * 44)
    for t in TICKERS:
        r   = returns[t].values
        nf  = fits[t]["normal"]
        tf  = fits[t]["student_t"]
        res = {}
        for cl in CLS:
            var_n   = -stats.norm.ppf(1 - cl, nf["mu"], nf["sigma"]) * notional
            var_t   = -stats.t.ppf(1 - cl, tf["nu"], tf["mu"], tf["sigma"]) * notional
            var_h   = -float(np.quantile(r, 1 - cl)) * notional
            res[cl] = {"normal": var_n, "student_t": var_t, "historical": var_h,
                       "underestimation": var_h - var_n}
        out[t] = res
        for cl in CLS:
            v = res[cl]
            print(f"  {t} {cl*100:.1f}%: N=${v['normal']:>10,.0f}  "
                  f"t=${v['student_t']:>10,.0f}  "
                  f"Hist=${v['historical']:>10,.0f}  "
                  f"Δ=${v['underestimation']:>+10,.0f}")
    print()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _dollar_fmt(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ── Figure 1: Return distributions ───────────────────────────────────────────

def fig1_distributions(returns, fits):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        "Log Return Distributions with Fitted Densities\n"
        "AAPL & TSLA  (2015 – 2024)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    for ax, t in zip(axes, TICKERS):
        r  = returns[t].values
        nf = fits[t]["normal"]
        tf = fits[t]["student_t"]
        ax.hist(r, bins=120, density=True, color=C[t], alpha=0.35,
                label="Empirical", zorder=2, edgecolor="none")
        x = np.linspace(r.min(), r.max(), 600)
        ax.plot(x, stats.norm.pdf(x, nf["mu"], nf["sigma"]),
                color=C["normal"], lw=2, ls="--",
                label=f"Normal (σ = {nf['sigma']:.4f})")
        ax.plot(x, stats.t.pdf(x, tf["nu"], tf["mu"], tf["sigma"]),
                color=C["student"], lw=2,
                label=f"Student-t (ν̂ = {tf['nu']:.2f})")
        sk = stats.skew(r);  ku = stats.kurtosis(r)
        ax.text(0.97, 0.97,
                f"skew = {sk:.3f}\nexcess kurt = {ku:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85))
        ax.set_title(f"{t}  (n = {len(r):,})")
        ax.set_xlabel("Log Return")
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    plt.savefig("fig1_return_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_return_distributions.png")


# ── Figure 2: Q-Q vs Normal ───────────────────────────────────────────────────

def fig2_qq_normal(returns):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Q-Q Plots vs. Normal Distribution", fontsize=13, fontweight="bold")
    for ax, t in zip(axes, TICKERS):
        r = returns[t].values
        (osm, osr), (slope, intercept, r_val) = stats.probplot(r, dist="norm", fit=True)
        ax.scatter(osm, osr, color=C[t], alpha=0.25, s=5, label="Observed", rasterized=True)
        lo, hi = osm[0], osm[-1]
        ax.plot([lo, hi], [slope*lo + intercept, slope*hi + intercept],
                color="black", lw=1.5, ls="--", label="Normal reference")
        ax.set_title(f"{t} vs. Normal  (R = {r_val:.4f})")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend()
    plt.tight_layout()
    plt.savefig("fig2_qq_normal.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_qq_normal.png")


# ── Figure 3: Q-Q vs Student-t ───────────────────────────────────────────────

def fig3_qq_student_t(returns, fits):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Q-Q Plots vs. Fitted Student-t Distribution",
                 fontsize=13, fontweight="bold")
    for ax, t in zip(axes, TICKERS):
        r  = returns[t].values
        tf = fits[t]["student_t"]
        zs = (r - tf["mu"]) / tf["sigma"]
        (osm, osr), (slope, intercept, r_val) = stats.probplot(
            zs, dist=stats.t, sparams=(tf["nu"],), fit=True)
        ax.scatter(osm, osr, color=C[t], alpha=0.25, s=5, rasterized=True, label="Observed")
        lo, hi = osm[0], osm[-1]
        ax.plot([lo, hi], [slope*lo + intercept, slope*hi + intercept],
                color=C["student"], lw=1.5, ls="--",
                label=f"Student-t ref (ν̂={tf['nu']:.2f})")
        ax.set_title(f"{t} vs. Student-t  (R = {r_val:.4f})")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.legend()
    plt.tight_layout()
    plt.savefig("fig3_qq_student_t.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_qq_student_t.png")


# ── Figure 4: Volatility clustering ──────────────────────────────────────────

def fig4_volatility(returns, vol):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Volatility Clustering & ARCH Effects  (Squared Returns)",
                 fontsize=13, fontweight="bold")
    for col, t in enumerate(TICKERS):
        r   = returns[t].values
        r2  = vol[t]["r2"]
        lb  = vol[t]["lb"]
        idx = returns[t].index

        # Top row: |returns| time series
        ax0 = axes[0, col]
        ax0.fill_between(idx, np.abs(r), color=C[t], alpha=0.55, lw=0)
        ax0.set_title(f"{t}  — |Log Returns|")
        ax0.set_ylabel("|Return|")
        ax0.set_xlabel("")

        # Bottom row: ACF of r²
        ax1 = axes[1, col]
        sm.graphics.tsa.plot_acf(r2, lags=30, alpha=0.05, ax=ax1, color=C[t],
                                  title="")
        q10 = lb["lb_stat"].iloc[0]
        p10 = lb["lb_pvalue"].iloc[0]
        ax1.set_title(f"{t}  ACF(r²)  —  Ljung-Box Q(10) = {q10:.1f}  (p = {p10:.1e})")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.savefig("fig4_volatility_clustering.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig4_volatility_clustering.png")


# ── Figure 5: VaR comparison ─────────────────────────────────────────────────

def fig5_var(var_results):
    cl_labels = ["95%", "99%", "99.9%"]
    x     = np.arange(len(CLS))
    width = 0.26

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Value-at-Risk Comparison  (${NOTIONAL:,} Notional)",
                 fontsize=13, fontweight="bold")

    for ax, t in zip(axes, TICKERS):
        res = var_results[t]
        nv  = [res[cl]["normal"]     for cl in CLS]
        tv  = [res[cl]["student_t"]  for cl in CLS]
        hv  = [res[cl]["historical"] for cl in CLS]

        ax.bar(x - width, nv, width, label="Normal VaR",    color=C["normal"],  alpha=0.85)
        ax.bar(x,          tv, width, label="Student-t VaR", color=C["student"], alpha=0.85)
        ax.bar(x + width,  hv, width, label="Empirical VaR", color=C[t],         alpha=0.85)

        under = res[0.999]["underestimation"]
        ymax  = max(nv[2], tv[2], hv[2])
        ax.annotate(
            f"Gaussian underestimates\nempirical 99.9% VaR\nby ${under:,.0f}",
            xy=(x[2] + width, hv[2]), xytext=(x[2] - 0.45, ymax * 1.08),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.2),
            fontsize=8.5, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff3f3", alpha=0.9),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(cl_labels)
        ax.set_xlabel("Confidence Level")
        ax.set_ylabel(f"VaR  (${NOTIONAL//1_000_000}M notional)")
        ax.set_title(f"{t}")
        ax.legend()
        _dollar_fmt(ax)

    plt.tight_layout()
    plt.savefig("fig5_var_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig5_var_comparison.png")


# ── Figure 6: Summary panel ───────────────────────────────────────────────────

def fig6_summary(moments, fits, var_results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.suptitle("Model Comparison Summary", fontsize=13, fontweight="bold")
    colors = [C["AAPL"], C["TSLA"]]

    # Panel A — excess kurtosis
    ax = axes[0]
    vals = [moments[t]["excess_kurtosis"] for t in TICKERS]
    bars = ax.bar(TICKERS, vals, color=colors, alpha=0.82, width=0.45)
    ax.axhline(0, color="black", lw=1, ls="--", label="Normal = 0")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("Leptokurtosis (H1)")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.06,
                f"{v:.2f}", ha="center", fontweight="bold")

    # Panel B — ΔAIC
    ax = axes[1]
    vals = [fits[t]["delta_aic"] for t in TICKERS]
    bars = ax.bar(TICKERS, vals, color=colors, alpha=0.82, width=0.45)
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_ylabel("ΔAIC  (Normal − Student-t)")
    ax.set_title("Model Selection (H2)\nhigher = Student-t decisively better")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 3,
                f"{v:.0f}", ha="center", fontweight="bold")

    # Panel C — degrees of freedom ν̂
    ax = axes[2]
    vals = [fits[t]["student_t"]["nu"] for t in TICKERS]
    bars = ax.bar(TICKERS, vals, color=colors, alpha=0.82, width=0.45)
    ax.axhline(30, color="grey", lw=1, ls="--", label="ν = 30 (≈ Normal)")
    ax.set_ylabel("Fitted ν̂  (lower = heavier tails)")
    ax.set_title("Heavy-Tail Severity (H2)\nlower ν̂ → fatter tails")
    ax.legend()
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                f"{v:.2f}", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig("fig6_model_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig6_model_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "═" * 62
    print(sep)
    print("  Fat Tails, Volatility Clustering & Risk Underestimation")
    print("  Samanvi Rajput — VIT Vellore  |  arXiv q-fin.ST / stat.AP")
    print(sep + "\n")

    returns = fetch_log_returns()
    moments = moment_analysis(returns)
    _       = normality_tests(returns)
    fits    = fit_distributions(returns)
    vol     = volatility_clustering(returns)
    var_res = compute_var(returns, fits)

    print("── Generating Figures " + "─" * 40)
    fig1_distributions(returns, fits)
    fig2_qq_normal(returns)
    fig3_qq_student_t(returns, fits)
    fig4_volatility(returns, vol)
    fig5_var(var_res)
    fig6_summary(moments, fits, var_res)

    print("\n── Final Summary " + "─" * 45)
    for t in TICKERS:
        m     = moments[t]
        tf    = fits[t]["student_t"]
        da    = fits[t]["delta_aic"]
        under = var_res[t][0.999]["underestimation"]
        lb10  = vol[t]["lb"]["lb_stat"].iloc[0]
        print(f"\n  {t}:")
        print(f"    Excess kurtosis         : {m['excess_kurtosis']:.2f}")
        print(f"    Student-t ν̂             : {tf['nu']:.2f}")
        print(f"    ΔAIC (N vs t)           : {da:.0f}")
        print(f"    Ljung-Box Q(10) on r²   : {lb10:.1f}")
        print(f"    Gaussian VaR underest.  : ${under:,.0f}  @ 99.9%  ($1M notional)")

    print(f"\n  ✓  6 figures written to current directory.")
    print(sep)


if __name__ == "__main__":
    main()
