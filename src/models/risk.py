"""Low / Medium / High degradation-risk categories, with validation.

Definition (rule-based, uses only information available at the current cycle)
  soh_now     causal rolling median of SoH
  fade_rate   SoH lost per cycle (BA) / per EFC (RW) over the recent trailing window
  projected   (soh_now - EOL) / fade_rate  - remaining life if the recent trend continued

  High    soh_now <= EOL + 3 pp           or projected <= T1
  Low     soh_now >  EOL + 10 pp          and projected >  T2
  Medium  otherwise

T1 / T2 are the 33rd / 67th percentiles of `projected` on the training cells, so
the three classes are roughly balanced by construction. These thresholds (and
the 3 pp / 10 pp margins) are project choices for illustration, NOT
established engineering limits.

Validation (leave-one-cell-out): thresholds are learned without the held-out
cell, which is then classified and compared with its *actual* future RUL:
  * median true RUL per class (should fall Low > Medium > High);
  * Spearman correlation between risk level and true RUL (should be negative);
  * agreement with a reference class from true-RUL terciles of the training cells.
The same is reported for an ML alternative that classifies the held-out
predictions of the best RUL model with the same tercile cut-offs.

Usage:
    python -m src.models.risk
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix

from src.config import EOL_SOH, FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.models.train_rul import DATASETS, load
from src.visualization.style import CATEGORICAL, INK_2, RISK_COLORS, apply_style, save

RESULTS = REPORTS_DIR / "results"
LEVELS = ["Low", "Medium", "High"]
MARGIN_HIGH, MARGIN_LOW = 0.03, 0.10


def projected_life(df: pd.DataFrame, cfg: dict) -> pd.Series:
    soh = df[cfg["soh"]]
    rate = -df[cfg["slope"]] * cfg["slope_to_soh_per_unit"]
    proj = ((soh - EOL_SOH) / rate).where(rate > 0, np.inf)
    return proj.where(soh > EOL_SOH, 0.0).fillna(np.inf)


def classify(soh_now: pd.Series, projected: pd.Series, t1: float, t2: float) -> pd.Series:
    high = (soh_now <= EOL_SOH + MARGIN_HIGH) | (projected <= t1)
    low = (soh_now > EOL_SOH + MARGIN_LOW) & (projected > t2)
    return pd.Series(np.select([high, low], ["High", "Low"], "Medium"), index=soh_now.index)


def classes_from_rul(rul, r1, r2) -> np.ndarray:
    return np.select([rul <= r1, rul <= r2], ["High", "Medium"], "Low")


def thresholds(df: pd.DataFrame, target: str) -> dict:
    """Tercile cut-offs of projected life (rule) and of true RUL (reference / ML classes)."""
    finite = df.projected[np.isfinite(df.projected) & (df.projected > 0)]
    t1, t2 = np.percentile(finite, [33.3, 66.7])
    r1, r2 = np.percentile(df[target], [33.3, 66.7])
    return {"t1": float(t1), "t2": float(t2), "r1": float(r1), "r2": float(r2)}


def save_thresholds(name: str, cfg: dict) -> dict:
    """Thresholds from ALL labelled cells, for inference (validation above never used them)."""
    df, _ = load(cfg)
    df["projected"] = projected_life(df, cfg)
    meta = thresholds(df, cfg["target"]) | {
        "dataset": name, "unit": cfg["unit"], "soh_col": cfg["soh"], "slope_col": cfg["slope"],
        "slope_to_soh_per_unit": cfg["slope_to_soh_per_unit"], "eol_soh": EOL_SOH,
        "margin_high": MARGIN_HIGH, "margin_low": MARGIN_LOW,
        "note": "t1/t2: terciles of projected life; r1/r2: terciles of true RUL; project choices, "
                "not engineering limits"}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / f"risk_{cfg['tag']}.json").write_text(json.dumps(meta, indent=2))
    return meta


def validate(name: str, cfg: dict, rul_preds: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df, _ = load(cfg)
    df["projected"] = projected_life(df, cfg)
    target = cfg["target"]
    best_model = (rul_preds[(rul_preds.dataset == name) & (rul_preds.validation == "leave-one-cell-out")]
                  .groupby("model").apply(lambda g: (g.y_true - g.y_pred).abs().mean(), include_groups=False)
                  .drop(["Mean baseline"], errors="ignore").idxmin())
    ml = rul_preds[(rul_preds.dataset == name) & (rul_preds.validation == "leave-one-cell-out")
                   & (rul_preds.model == best_model)]
    out = []
    for cell in sorted(df.cell.unique()):
        tr, te = df[df.cell != cell], df[df.cell == cell].copy()
        th = thresholds(tr, target)
        t1, t2, r1, r2 = th["t1"], th["t2"], th["r1"], th["r2"]
        te["risk_rule"] = classify(te[cfg["soh"]], te.projected, t1, t2)
        te["risk_reference"] = classes_from_rul(te[target].to_numpy(), r1, r2)
        p = ml[ml.cell == cell].set_index(cfg["order"]).y_pred
        te["rul_pred_ml"] = te[cfg["order"]].map(p)
        te["risk_ml"] = classes_from_rul(te.rul_pred_ml.to_numpy(), r1, r2)
        out.append(te)
    res = pd.concat(out)
    lvl = {"Low": 0, "Medium": 1, "High": 2}
    stats = {"dataset": name, "ml_model": best_model, "n": len(res)}
    for method in ("rule", "ml"):
        col = f"risk_{method}"
        rho = spearmanr(res[col].map(lvl), res[target]).statistic
        stats[f"{method}_spearman_vs_true_rul"] = rho
        stats[f"{method}_accuracy_vs_reference"] = accuracy_score(res.risk_reference, res[col])
        med = res.groupby(col)[target].median()
        for k in LEVELS:
            stats[f"{method}_median_true_rul_{k}"] = med.get(k, np.nan)
            stats[f"{method}_share_{k}"] = (res[col] == k).mean()
        cm = pd.DataFrame(confusion_matrix(res.risk_reference, res[col], labels=LEVELS),
                          index=[f"true {k}" for k in LEVELS], columns=[f"pred {k}" for k in LEVELS])
        cm.to_csv(RESULTS / f"risk_{cfg['tag']}_{method}_confusion.csv")
    return res, stats


def plot(res: pd.DataFrame, name: str, cfg: dict) -> None:
    target, unit = cfg["target"], cfg["unit"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    data = [res.loc[res.risk_rule == k, target] for k in LEVELS]
    bp = ax.boxplot(data, tick_labels=LEVELS, widths=0.5, patch_artist=True, showfliers=False)
    for patch, k in zip(bp["boxes"], LEVELS):
        patch.set_facecolor(RISK_COLORS[k])
        patch.set_alpha(0.85)
    for med in bp["medians"]:
        med.set_color("#0b0b0b")
    ax.set_ylabel(f"Actual remaining life ({unit})")
    ax.set_title("Rule-based risk vs actual RUL (held-out cells)")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    cells = sorted(res.cell.unique(), key=lambda c: int("".join(ch for ch in c if ch.isdigit())))[:4]
    for k, cell in enumerate(cells):
        d = res[res.cell == cell].sort_values(cfg["order"])
        ax.plot(d[cfg["order"]], d[cfg["soh"]], color=CATEGORICAL[k], lw=1.2, label=cell)
        for lvl in LEVELS:
            s = d[d.risk_rule == lvl]
            ax.scatter(s[cfg["order"]], s[cfg["soh"]], s=12, color=RISK_COLORS[lvl], zorder=3,
                       label=f"{lvl} risk" if k == 0 else "_nolegend_", edgecolor="#fcfcfb", lw=0.4)
    ax.axhline(EOL_SOH, color=INK_2, lw=0.8, ls="--")
    ax.set_xlabel("cycle" if cfg["tag"] == "ba" else "EFC")
    ax.set_ylabel("SoH (causal rolling median)")
    ax.set_title("Risk level along the SoH trajectory")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle(name, x=0.01, ha="left", color=INK_2)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "risk" / f"risk_{cfg['tag']}.png")


def run() -> None:
    apply_style()
    rul_preds = pd.read_parquet(RESULTS / "rul_predictions.parquet")
    stats = []
    for name in ("Battery Aging", "Randomized Usage"):
        cfg = DATASETS[name]
        res, s = validate(name, cfg, rul_preds)
        res[["cell", cfg["order"], cfg["soh"], "projected", cfg["target"], "risk_rule",
             "risk_reference", "rul_pred_ml", "risk_ml"]].to_csv(
            RESULTS / f"risk_{cfg['tag']}_assignments.csv", index=False)
        plot(res, name, cfg)
        save_thresholds(name, cfg)
        stats.append(s)
    st = pd.DataFrame(stats)
    st.to_csv(RESULTS / "risk_validation.csv", index=False)
    print(st.T.to_string())


if __name__ == "__main__":
    run()
