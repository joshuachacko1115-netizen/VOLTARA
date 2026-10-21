"""Driving-range estimation - a separate, assumption-driven component.

What the NASA data can and cannot support
  * Both datasets are single 18650 cells cycled in a laboratory. They contain no
    vehicle speed, distance, road load, pack configuration, HVAC load or drivetrain
    efficiency, so real-world range CANNOT be estimated or validated from them.
  * They DO contain measured cell energy over life:
      Battery Aging      energy of each 2 A discharge to 2.7 V (Wh)
      Randomized Usage   energy of each 1 A reference discharge (Wh), and for
                         RW13-RW28 the duration of a 15 W constant-power
                         discharge (constant power is closer to a vehicle load
                         than constant current).
  * So this module reports
      1. energy retention  = E(t) / E(new)  - measured, and
      2. range in km       = only through an explicit RangeAssumptions object that
                             the user must supply; the default values are
                             illustrative placeholders, not data about any vehicle.

If the consumption per km is held constant, relative range retention equals
energy retention. That proportionality is itself an assumption (it ignores
temperature, rising resistance at high power, and driving style).

Usage:
    python -m src.models.range_estimation
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import EOL_SOH, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR
from src.visualization.style import CATEGORICAL, INK_2, apply_style, save

RESULTS = REPORTS_DIR / "results"
# A 15 W discharge that ends at 3.2 V but rests back above this voltage stopped on power
# (resistance), not on exhausted energy.
POWER_LIMITED_REST_V = 4.0


@dataclass
class RangeAssumptions:
    """Illustrative placeholders - replace with real vehicle data before drawing conclusions."""
    cells_in_series: int = 96
    cells_in_parallel: int = 30
    usable_window: float = 0.90           # fraction of measured cell energy the BMS allows
    consumption_wh_per_km: float = 150.0  # at the wheels incl. drivetrain losses
    auxiliary_wh_per_km: float = 10.0     # HVAC, electronics
    temperature_factor: float = 1.00     # <1 for cold-weather losses

    @property
    def n_cells(self) -> int:
        return self.cells_in_series * self.cells_in_parallel


def estimate_range_km(cell_energy_wh, a: RangeAssumptions) -> np.ndarray:
    pack_wh = np.asarray(cell_energy_wh, float) * a.n_cells * a.usable_window * a.temperature_factor
    return pack_wh / (a.consumption_wh_per_km + a.auxiliary_wh_per_km)


def energy_tables() -> pd.DataFrame:
    ba = pd.read_parquet(PROCESSED_DIR / "ba_cycles.parquet")
    ba = ba[ba.in_model_cohort & ba.is_valid & ba.energy_wh.notna()].copy()
    ba["ageing"], ba["cell_energy_wh"], ba["source"] = ba.cycle_number, ba.energy_wh, "BA: 2 A to 2.7 V"
    rw = pd.read_parquet(PROCESSED_DIR / "rw_checks.parquet")
    rw = rw[rw.is_valid].copy()
    a = rw.assign(ageing=rw.efc, cell_energy_wh=rw.ref_energy_wh, source="RW: 1 A to 3.2 V")
    b = rw[rw.power_discharge_duration_s.notna()].assign(
        ageing=lambda d: d.efc, cell_energy_wh=lambda d: 15.0 * d.power_discharge_duration_s / 3600,
        source="RW: 15 W constant power",
        power_limited=lambda d: d.power_discharge_rest_v_after > POWER_LIMITED_REST_V)
    out = pd.concat([ba, a, b], ignore_index=True)[["cell", "source", "ageing", "soh", "cell_energy_wh",
                                                     "power_limited"]]
    out["power_limited"] = out.power_limited.fillna(False).astype(bool)
    out = out.dropna(subset=["cell_energy_wh"])
    e0 = out.groupby(["source", "cell"]).cell_energy_wh.transform(lambda s: s.iloc[:3].median())
    out["energy_retention"] = out.cell_energy_wh / e0
    return out


def run(assumptions: RangeAssumptions | None = None) -> None:
    apply_style()
    a = assumptions or RangeAssumptions()
    RESULTS.mkdir(parents=True, exist_ok=True)
    t = energy_tables()
    t["range_km_illustrative"] = estimate_range_km(t.cell_energy_wh, a)
    t.to_csv(RESULTS / "range_energy_retention.csv", index=False)

    # Energy fades slightly faster than capacity because the voltage under load drops too.
    def diff(g):
        return float(np.nanmedian(
            g.energy_retention - g.soh / g.groupby("cell").soh.transform(lambda s: s.iloc[:3].median())))
    rel = t.groupby("source").apply(lambda g: pd.Series({
        "rows": len(g),
        "power_limited_rows": int(g.power_limited.sum()),
        "median_energy_retention_minus_capacity_soh": diff(g),
        "same_excluding_power_limited": diff(g[~g.power_limited]),
    }), include_groups=False)
    rel.to_csv(RESULTS / "range_energy_vs_capacity.csv")

    # Illustrative range at selected SoH levels for a "new" cell of each source.
    new_e = t.groupby("source").apply(lambda g: g.groupby("cell").cell_energy_wh.first().median(),
                                      include_groups=False)
    levels = [1.0, 0.9, EOL_SOH, 0.7]
    table = pd.DataFrame({f"energy retention {int(l * 100)} %": estimate_range_km(new_e * l, a)
                          for l in levels}, index=new_e.index).round(0)
    table.to_csv(RESULTS / "range_illustrative_km.csv")
    pd.Series(asdict(a)).to_csv(RESULTS / "range_assumptions.csv", header=["value"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for k, (src, g) in enumerate(t.groupby("source")):
        x = g.soh / g.groupby("cell").soh.transform(lambda s: s.iloc[:3].median())
        lim_ = g.power_limited
        axes[0].scatter(x[~lim_], g.energy_retention[~lim_], s=6, alpha=0.5, color=CATEGORICAL[k],
                        label=src, lw=0)
        if lim_.any():
            axes[0].scatter(x[lim_], g.energy_retention[lim_], s=10, color=CATEGORICAL[k], marker="x",
                            lw=0.7, label=f"{src}: power-limited (rest V > {POWER_LIMITED_REST_V} V)")
    lim = [0.3, 1.05]
    axes[0].plot(lim, lim, color=INK_2, lw=0.8, ls="--")
    axes[0].set_xlabel("Capacity relative to new")
    axes[0].set_ylabel("Energy relative to new")
    axes[0].set_title("Measured energy retention vs capacity retention")
    axes[0].legend(fontsize=7)
    for k, cell in enumerate(["RW13", "RW17", "RW21", "RW25"]):
        g = t[(t.cell == cell) & (t.source == "RW: 15 W constant power")]
        axes[1].plot(g.ageing, g.range_km_illustrative, color=CATEGORICAL[k], lw=1.4, marker="o", ms=3,
                     label=cell)
    axes[1].set_xlabel("Equivalent full cycles")
    axes[1].set_ylabel("km (illustrative)")
    axes[1].set_title("ILLUSTRATIVE range from 15 W energy\n(hypothetical pack, not validated)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, FIGURES_DIR / "range" / "range_energy_retention.png")

    print("Assumptions (illustrative placeholders):", asdict(a))
    print("\nEnergy retention minus capacity retention (median):\n", rel.round(3).to_string())
    print("\nIllustrative range (km) of the hypothetical pack at given energy retention:\n",
          table.to_string())


if __name__ == "__main__":
    run()
