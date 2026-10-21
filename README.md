# VOLTARA — Battery health, degradation and RUL from NASA data

Machine-learning pipeline for lithium-ion battery **State of Health (SoH)**, **Remaining Useful Life (RUL)**,
degradation analysis, anomaly detection, degradation-risk classes and a clearly separated range-estimation
component. No dashboard or deployment (by design, for now).

Every number below comes from an experiment run on the data in `data/raw/`. Regenerate them with the commands
in *Run it*; the CSV/PNG outputs are in `reports/`.

## 1. What the data actually is

The two files in `data/raw/` are complete ZIP archives (CRC-checked) even though Chrome left them named
`*.crdownload`. The code identifies them by their contents, so you do **not** need to rename them, and the
originals are never modified.

| | Battery Aging (`5. Battery Data Set`) | Randomized Usage (`11. Randomized Battery Usage Data Set`) |
|---|---|---|
| Cells | 34 unique: B0005–B0056 (B0025–28 appear in two inner ZIPs, byte-identical, loaded once) | 28: RW1–RW28 |
| MATLAB layout | `B00xx.cycle[k]` = charge / discharge / impedance, with `.data` signals | `data.step[k]`, ~10⁴–10⁵ steps per cell, each with `comment`, `type` C/D/R, V, I, T |
| Sampling | ~3–19 s per sample | 1 s (random-walk steps) to 60 s |
| Capacity | stored per discharge (`Capacity`, Ah) | not stored: integrated from the 1 A reference discharge |
| Impedance (EIS, Re, Rct) | yes | no |
| Ageing axis | discharge cycle number | no cycle concept → *equivalent full cycles* (EFC) = cumulative discharged Ah ÷ initial capacity |

**The datasets are not interchangeable**: different protocols, different capacity definitions, different
rated capacity, different ageing axis. They are modelled and scored separately; results are only compared
side by side, never pooled.

Things found by inspecting the files (and handled in code):

* Battery Aging discharge fields are `Current_load`/`Voltage_load` (README says `*_charge`); discharge current is negative.
* Cells differ in protocol: constant current at 1/2/4 A, a 4 A square wave (B0025–28), loads that change between
  cycles (B0038–B0044), ambient 4 / 24 / 43–44 °C. B0041–B0056 (4 °C) contain many near-zero-capacity runs
  that the README says were "not fully analysed".
* Temperature drop-outs of about −4094 °C in RW2, RW3, RW18 (masked as missing); impedance fits in B0049–B0052
  that are complex-valued or absurd (e.g. −10¹⁵ Ω, treated as failed fits).
* The stored `Capacity` is documented "to 2.7 V" but cutoffs differ by cell; capacity is recomputed to a common 2.7 V
  and agrees with the stored value to within 0.026 Ah for every cycle of the modelling cells.
* Randomized Usage reference tests changed mid-experiment: RW13–RW28 add a 15 W constant-power discharge.

## 2. Target definitions (project choices)

* **SoH, Battery Aging** = capacity to 2.7 V ÷ 2.0 Ah (the README's rated capacity).
  **SoH, Randomized Usage** = 1 A reference capacity ÷ the cell's first reference capacity (no rated value is given).
* **End of life** = smoothed SoH ≤ 80 %. The NASA READMEs use 70 % or 80 % depending on the group; 80 % is used for all.
  "Smoothed" is a centred 5-point (BA) / 3-point (RW) median, so one noisy cycle cannot define EOL. It looks at the future,
  so it is used only for *labels*, never as a feature.
* **RUL** = cycles (BA) or EFC (RW) until EOL. **Censoring**: cells that never reach 80 % (B0025–B0029, B0031, B0032, RW26–RW28)
  get no RUL label; they are kept aside and only checked against their known lower bound.
* **Modelling cohorts**
  * BA SoH: 11 cells at constant current, 24 or 43 °C (B0005–07, B0018, B0029–34, B0036), 1 357 valid cycles.
  * BA RUL: the 6 of them that reach EOL after ≥ 10 cycles (B0005–07, B0018, B0030, B0036). B0033/B0034 (4 A) sit near 80 % from the start, so
    they have no meaningful trajectory.
  * RW: all 28 cells (25 reach EOL).
  * Excluded from BA modelling, with reasons in `src/config.py`: B0025–28 (square-wave load), B0038–44 (load current changes between
    cycles), B0045–56 (4 °C, unexplained low-capacity runs). They are still used in EDA and anomaly detection.

## 3. Pipeline

```
data/raw/*.zip ──extract──▶ data/processed/extracted/  (+ manifest.csv with MD5s)
      └─ loaders ─▶ data/processed/interim/*.parquet   (tidy tables)
            └─ features ─▶ ba_cycles.parquet, rw_checks.parquet  (features + quality flags + SoH/RUL)
                  └─ models ─▶ models/*.joblib, reports/results/*.csv, reports/figures/*.png
```

| Module | What it does |
|---|---|
| `src/data/extract.py` | finds the two archives by content, extracts the nested ZIPs, writes a manifest, checks duplicates |
| `src/data/loaders.py`, `build_dataset.py` | parse `.mat` into tables; mask impossible temperatures and invalid impedance fits |
| `src/preprocessing/cleaning.py` | short-gap interpolation (≤ 3 samples, interior only), causal moving average, noise estimate, Hampel flags, causal rolling slope |
| `src/features/battery_aging.py` | per-discharge features in groups: condition, charge, early-discharge, partial-curve, impedance, usage, *full* (leakage-prone) |
| `src/features/randomized_usage.py` | per-reference-check features: usage since last check, reference-charge CC/CV, pulse/operational resistance |
| `src/features/targets.py` | quality flags, SoH, EOL, RUL, censoring, causal trend features |
| `src/evaluation/` | metrics (MAE, RMSE, R²) and leave-cell-out / leave-group-out / chronological splitters with a leakage guard |
| `src/models/` | `train_soh.py`, `train_rul.py`, `sequence_model.py` (CNN-LSTM), `censored_rul.py`, `anomaly.py`, `risk.py`, `range_estimation.py` |
| `src/inference/` | `predict.py`: `VoltaraPredictor`, loads `models/{soh,rul,anomaly}_{ba,rw}.joblib` and `risk_{ba,rw}.json` (optionally `rul_*_censoring_aware.joblib`) |
| `src/visualization/eda.py` | all EDA figures |

**Leakage controls.** Imputation (median) and scaling are inside an sklearn `Pipeline` that is refitted inside every fold, so
they only see training cells. Cleaning is per-cell and uses only causal operations for features. Splits are asserted
(`check_split`): no shared cell, or, for the chronological split, no training row later than a test row. Unit tests cover this
(`tests/`). One more check is *deliberate leakage*: each SoH experiment also trains with features that contain the answer
(see below) to show what leakage looks like.

## 4. EDA (`reports/figures/eda/`)

Capacity vs cycle for every cell (`01`, `02`), normalised SoH comparison (`03`), voltage/current/temperature profiles across life
(`04`, `05`), temperature across cycles (`06`), missing values and quality flags (`07`), feature distributions (`08`),
Spearman correlations (`09`), ageing relations (`10`), sensor noise (`11`).

What the plots show:

* Capacity fades roughly linearly at 24 °C / 2 A; much faster at 4 A and 43 °C (B0030 reaches 80 % at cycle 29, B0005–07/18 at 59–85).
  Random-walk cells at 40 °C fade *slower* per EFC than room-temperature cells in this data.
* During discharge, B0005 gets hotter and ends earlier as it ages (peak ~39 → 41 °C), and its voltage curve sags sooner.
* The strongest single relation with SoH is charging behaviour: constant-current charge duration (Spearman ρ = 0.92 BA, 0.96 RW).
  Re and Rct correlate only moderately (ρ ≈ −0.4 to −0.5, BA) because their level differs by cell as much as by age.
* Quality problems are concentrated in the 4 °C cells (`07`): B0052 has 21 of 25 cycles without load.

## 5. Results

All errors below are on **held-out cells** unless stated. SoH errors are in SoH fraction (0.02 = 2 percentage points).

### SoH estimation (leave-one-cell-out)

Feature sets are nested; the question is "how well can SoH be estimated *without* running a capacity test?".

| Dataset | Feature set | Mean baseline MAE | Linear Reg. MAE | Best model | MAE | RMSE | R² |
|---|---|---|---|---|---|---|---|
| Battery Aging | A: charge + impedance + usage | 0.085 | 0.059 | Random Forest | 0.021 | 0.028 | 0.907 |
| | B: + early discharge (≤ 600 s) | 0.085 | 0.032 | Random Forest | 0.022 | 0.027 | 0.912 |
| | C: + partial curve | 0.085 | 0.036 | Random Forest | 0.022 | 0.027 | 0.913 |
| | *leakage check: + full discharge* | 0.085 | 0.005 | Linear Reg. | 0.005 | 0.012 | 0.982 |
| Randomized Usage | A: operational data only | 0.138 | 0.037 | Random Forest | 0.029 | 0.047 | 0.911 |
| | B: + reference charge | 0.138 | 0.032 | Random Forest / GB | 0.025 | 0.041 | 0.933 |
| | C: + pulse resistance | 0.138 | 0.031 | Gradient Boosting | 0.024 | 0.039 | 0.938 |
| | *leakage check: + reference discharge* | 0.138 | 0.006 | Linear Reg. | 0.006 | 0.009 | 0.997 |

* Trees clearly beat linear regression on Battery Aging (it is non-linear across cells). They are only slightly better on Randomized Usage.
* Adding discharge information to the BA features does not help (0.021 → 0.022): charging features already carry the signal.
* **Caveat on BA set A:** `chg_ah` (Ah put in by the preceding full charge) is close to the previous cycle's discharged capacity, because every NASA charge
  starts from a fully discharged cell. It is not future information, but it is a capacity measurement by another route. Removing it raises RF MAE from
  0.021 to 0.030 (R² 0.91 → 0.84, same folds). The CC-charge duration has the same dependence. In a vehicle, charges rarely start from empty, so expect
  the set-A error to be closer to 0.030 unless partial-charge features are used. The saved model is unchanged; this is documented, not fixed.
* The "leakage" rows are not results to report: duration and energy of the discharge are the capacity, so a linear model reproduces SoH almost exactly.
* Harder splits: chronological 60/40 (train on early life, test on late life) gives BA MAE 0.027 with set A (RF, R² 0.68) and 0.022 with sets B/C (linear regression best, R² 0.77–0.81),
  RW 0.039–0.049 (linear regression best, R² 0.65–0.77). Trees cannot extrapolate below the SoH range seen early in life, which is why linear regression wins here.
  Leave-one-*group*-out (an unseen RW experiment type) gives RW MAE 0.039 (GB, B set, R² 0.87).
* Per-cell errors vary widely (RF, set C): BA MAE 0.006 (B0031) to 0.031 (B0034), with low R² for B0032 (0.19), B0036 (0.20) and B0034 (0.37).
  RW (GB, set C) MAE 0.008 (RW18) to 0.099 (RW12, whose capacity curve differs visibly from the other cells in its group).

Saved models: `models/soh_ba.joblib` (Random Forest, set A) and `models/soh_rw.joblib` (Gradient Boosting, set C), each with a `.json` describing features and CV scores.

### RUL prediction (leave-one-cell-out)

| Dataset / cohort | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| **Randomized Usage** (203 rows, 25 cells, RUL in EFC) | Mean baseline | 78.5 | 97.2 | −0.02 |
| | Linear extrapolation of recent trend | 101.1 | 149.2 | −1.41 |
| | Linear Regression | 33.4 | 51.6 | 0.71 |
| | Random Forest | 32.5 | 51.1 | 0.72 |
| | Gradient Boosting | **32.3** | 50.1 | 0.73 |
| | CNN-LSTM (window 4) | 32.4 | **49.1** | **0.74** |
| RW, leave-one-experiment-group-out | Random Forest (best) | 38.5 | 53.7 | 0.69 |
| **Battery Aging, same protocol** (B0005/06/07/18; 279 rows) | Mean baseline | 18.6 | 22.1 | −0.05 |
| | Linear extrapolation | 18.2 | 25.3 | −0.38 |
| | Random Forest | **8.8** | 10.2 | 0.78 |
| | Gradient Boosting | 9.0 | 10.3 | 0.77 |
| | CNN-LSTM (window 10) | 9.4 | 11.8 | 0.70 |
| **Battery Aging, all 6 cells** (486 rows) | Mean baseline | 40.4 | 53.4 | −0.39 |
| | Random Forest | 33.2 | 49.9 | −0.21 |
| | CNN-LSTM | 39.4 | 57.9 | −0.63 |

Honest reading:

* On Randomized Usage, learned models cut the error of the mean baseline by ~60 %; **the four learned models are within noise of each
  other** (MAE 32.3–33.4; fold-to-fold spread ≈ 24). The CNN-LSTM is not better than a linear regression here, so the simpler model is preferable.
  It was trained once per fold with default settings and no tuning.
* The naive trend-extrapolation baseline is *worse than predicting the mean*: SoH curves are not linear, and a short-window slope is noisy.
* Battery Aging RUL **does not generalise across the 6-cell cohort** (R² < 0). B0036 (EOL at cycle 182) and B0030 (4 A, 43 °C, EOL at 29) have lives very
  different from the 2 A cells (59–85), and there is only one example of each condition. Restricting to the four same-protocol cells gives R² 0.78,
  but that is 4 cells; treat it as indicative, not conclusive. Linear Regression diverged on BA RUL (MAE 225), which is why it is not in the table.
* RUL error depends on where the cell is in its life (`reports/results/rul_*_error_by_true_rul.csv`).
* Censored cells: for RW26 and the BA cells the final model predicts RUL ≥ the known lower bound at 100 % of rows (B0029: 92 %), but for RW27 only 17 % and RW28 64 %
  (`rul_censored_cells_check.csv`). Those cells never reached EOL, so this is not scored as an error, but it is a limitation:
  * RW25–28 are one experiment group (40 °C, skewed-high current). Only RW25 is labelled, and it reached EOL at 238 EFC. RW27 is still at SoH 0.82 at 290 EFC
    and its fade *slows* (slope over the last 8 checks: −0.088 → −0.048 SoH per 100 EFC), unlike most training cells.
  * The model therefore under-predicts RW27 by 20–60 EFC through mid-life (e.g. 92 vs ≥ 141 at 149 EFC). Near the end of the record the errors are small.
  * Cause: right-censoring removes exactly the slow-fading cells from the labelled set (selection bias). The censoring-aware experiment below tests a fix.

### Censoring-aware RUL (`src/models/censored_rul.py`)

Never-failed cells (B0029/31/32, RW26–28) are added to training. Each of their rows only says "RUL > bound", where the bound is the life still observed after that row.
The unknown RUL is imputed, then the model is refitted, iterated 3 times. Two imputations are compared:

* **bound:** raise the target to the bound only where the model predicts below it. This uses nothing beyond the bound itself.
* **Buckley-James:** the expected RUL given RUL > bound, from the model's cell-grouped cross-validated residuals.

Every cell is held out once (leave-one-cell-out). Labelled cells are scored with MAE / R². Censored cells are scored by how often the prediction respects the bound
and the mean shortfall below it. Each dataset uses the same model type as its saved RUL model.

| Dataset | Training data | MAE, labelled cells | R² | Censored rows ≥ bound | Mean shortfall |
|---|---|---|---|---|---|
| Randomized Usage (GB) | labelled only (current model) | **32.3** EFC | **0.73** | 62 % | 9.2 EFC |
| | + censored, bound | 33.3 | 0.72 | 73 % | 7.3 |
| | + censored, Buckley-James | 34.2 | 0.71 | **81 %** | **5.9** |
| Battery Aging (RF) | labelled only (current model) | **33.2** cycles | −0.21 | 97 % | 0.02 |
| | + censored, bound | 35.9 | −0.23 | 97 % | 0.01 |
| | + censored, Buckley-James | 45.1 | −0.81 | 100 % | 0 |

Per cell (`rul_censoring_bounds_per_cell.csv`): RW27 goes from 17 % to 42 % of rows above the bound, with its shortfall reduced from 25.5 to 18.2 EFC. RW28 goes from 64 % to 100 %.

What it costs:
* **RW:** labelled-cell MAE rises by 1.9 EFC. That is well inside the fold-to-fold spread (± 24), but it is consistent. 13 of 25 labelled cells improve.
  RW25, the only *labelled* cell in the RW25–28 group, gets much worse (34 → 64 EFC): its never-failed siblings pull the whole group's predictions up,
  and RW25 failed first. Identical test conditions do not give identical lives, and the features cannot tell which sibling will fail early.
* **BA:** Buckley-James borrows its residual spread from the 2 A cells (errors of ±50 cycles) and applies it to the 4 A cells. It imputes long lives,
  so B0030 (4 A, 43 °C, failed at cycle 29) is over-predicted by ~95 cycles, and none of the 6 labelled cells improves. The "bound" variant is nearly neutral (B0030 even improves, 43 → 37).

**Decision: the default RUL models are unchanged** (labelled-only; best on cells with a known RUL). The Buckley-James model is saved as an option
(`models/rul_{ba,rw}_censoring_aware.joblib`, `VoltaraPredictor(..., rul_variant="censoring_aware")`). Use it only for Randomized-Usage-like batteries,
when under-estimating the life of slow-fading cells is the more costly error. It is **not** recommended for Battery Aging.

Not tried: survival models with covariates (e.g. random survival forests, Cox / AFT). `scikit-survival` is not available for this Python version.
With 3 censored cells per dataset, any method's gain is hard to separate from noise.

Most important features (Random Forest, descriptive only): BA SoH — CC charge duration (0.78); RW RUL — rolling SoH (0.59) and elapsed days (0.19).

### Anomaly detection (`reports/results/anomalies_*.csv`, figures in `reports/figures/anomaly/`)

Isolation Forest on per-cell robust z-scores of capacity, early-discharge voltage, temperature and charge-time features.
It flags 136 of 2 780 BA cycles and 53 of 620 RW checks. Each flag lists its three largest deviations (e.g. `capacity ≤−20σ`) and a context note.

* 91 % of the BA anomalies have a recorded, non-fault explanation: running at a different test condition from the cell's main one (86), failing a rule-based quality check
  (45), a long rest before the cycle (9). Only 12 have no recorded explanation. **That is not evidence of faults**: most of the flagged cycles are unusual because the *test* changed.
* Agreement with the rule-based quality flags is partial (45 of 277 flagged-by-rule cycles are also Isolation-Forest anomalies), so the two methods are complementary.
* 32 of the 53 RW anomalies are in a cell's first three checks (possibly a settling-in effect: not verified), and the most frequent deviating features are the reference-discharge temperature, pulse resistance and CC charge time. The other 21 have no recorded explanation.

### Degradation-risk classes (`src/models/risk.py`)

Rule: **High** if SoH ≤ 83 % or the projected remaining life (from the recent fade rate) is in the shortest third of the training cells; **Low** if SoH > 90 % *and* projected life is in the longest third;
otherwise **Medium**. The margins (3 and 10 points) and tercile cut-offs are *project choices, not engineering standards*; thresholds are learned without the held-out cell.

| | Spearman (risk level vs actual RUL) | Accuracy vs RUL-tercile reference | Median actual RUL Low / Med / High |
|---|---|---|---|
| RW, rule | −0.83 | 0.70 | 192 / 96 / 28 EFC |
| RW, ML (RUL model → terciles) | −0.87 | 0.81 | 232 / 126 / 36 EFC |
| BA, rule | −0.60 | 0.48 | 56 / 66 / 16 cycles |
| BA, ML | −0.44 | 0.50 | 62 / 49 / 21 cycles |

Classes order correctly in Randomized Usage. In Battery Aging the Low/Medium ordering is not preserved (56 vs 66 cycles): only 15 % of rows are "Low", and the cohort is small and heterogeneous.
The "reference" is itself defined from RUL, so accuracy against it measures consistency, not correctness against any external standard.

### Driving range (`src/models/range_estimation.py`) — **not validated**

The NASA data contain single laboratory cells: no vehicle speed, distance, road load, pack layout, HVAC or drivetrain data. **Real driving range cannot be estimated or validated from them.** What is measured:
cell energy over life. Median energy retention minus capacity retention: −0.3 pp (BA 2 A), −1.2 pp (RW 1 A), **−40 pp (RW 15 W constant power)**.

The −40 pp mixes two effects. In 148 of the 293 valid 15 W discharges the cell hit 3.2 V while still nearly full: it rests back above 4.0 V afterwards.
These discharges are **power-limited** (resistance), not energy-limited. In RW13 the 15 W energy drops from 2.8 Wh to 0.39 Wh between two checks.
Excluding them, the gap is −14 pp. Both numbers are measured (16 RW cells), not vehicle results. The `power_limited` column in
`range_energy_retention.csv` and the crosses in the figure mark these rows.

Assumptions used for any km figure (`RangeAssumptions`, all placeholders):

| Assumption | Default | Why it matters |
|---|---|---|
| Pack layout | 96 s × 30 p cells of this type | scales cell energy to pack energy |
| Usable window | 0.90 | BMS keeps a reserve at both ends |
| Consumption | 150 Wh/km + 10 Wh/km auxiliary | constant; no speed, temperature or route dependence |
| Temperature factor | 1.00 | no cold-weather loss |
| Proportionality | range ∝ cell energy at the measured discharge rate | ignores rate dependence, which the 15 W data show is large late in life |

Energy retention is referenced to the median of each cell's first three rows. For B0033 those rows are unusually low (SoH 0.62–0.65, then a jump to 0.85),
which gives ratios up to 1.3 in the figure. This is a property of that cell's data, not of the method.

Kilometre figures need a `RangeAssumptions` object (pack layout, usable window, Wh/km, ...). The defaults are **placeholders for a hypothetical pack**, and
`reports/results/range_illustrative_km.csv` should be read as arithmetic, not a prediction. Supplying real vehicle consumption data (e.g. a drive-cycle dataset) would be the next step.

## 6. Limitations and next steps

* The BA RUL cohort has 6 cells; conclusions about generalisation to new conditions are weak. More cells per condition would help most.
* SoH for the Randomized Usage dataset is relative to each cell's own first capacity, so absolute degradation is not comparable with Battery Aging.
* Hyper-parameters were not tuned; the CNN-LSTM used one seed. Tuning on inner folds and repeating seeds is needed before claiming differences of a few percent.
* The impedance features are measured every few cycles and carried forward (`impedance_age_cycles` records the staleness).
* Anomaly "explanations" are heuristics based on recorded context, not diagnoses. The z-scores are computed over each cell's whole history,
  so in RW the first checks (before the cell has aged) are often flagged: they differ from the cell's typical state, not because they are faulty.
* Risk classes for Battery Aging do not order Low > Medium (median true RUL 56 vs 66 cycles); see the risk section.
* No notebooks are included: `notebooks/` is empty. The modules run from the command line, and EDA figures are generated by `src/visualization/eda.py`.

## 7. Run it

Windows (PowerShell), from the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.run_pipeline          # everything, several minutes; add --no-deep to skip the CNN-LSTM
python -m pytest -q                 # 30 tests (7 need the trained models)
```

macOS/Linux: `source .venv/bin/activate`. XGBoost needs `brew install libomp` on macOS; without it, the code falls back to the other models
(the reported results do not use XGBoost because it was not loadable in the environment where they were produced).
PyTorch is only needed for the CNN-LSTM.

Single steps: `python -m src.data.extract`, `src.data.build_dataset`, `src.features.build_features`, `src.visualization.eda`,
`src.models.train_soh`, `src.models.train_rul`, `src.models.censored_rul` (needs `train_rul`), `src.models.anomaly`, `src.models.risk` (needs `train_rul` first), `src.models.range_estimation`.

### Inference (`src/inference/predict.py`)

After the pipeline has saved `models/`, score one battery's history (one row per cycle or reference check, oldest first):

```python
from src.inference.predict import VoltaraPredictor
from src.models.range_estimation import RangeAssumptions

p = VoltaraPredictor("battery_aging")           # or "randomized_usage"
print(p.required_inputs())                      # columns the models use
pred = p.predict(history_df)                    # one output row per input row
pred = p.predict(history_df, range_assumptions=RangeAssumptions(consumption_wh_per_km=170))
p.summary(history_df)                           # latest row as a dict, with notes
```

```bash
python -m src.inference.predict --dataset battery_aging --cell B0005     # demo on a NASA cell
python -m src.inference.predict --dataset randomized_usage --csv my.csv
```

| Output | Measured or modelled |
|---|---|
| `soh_pred` | model (`soh_*.joblib`) |
| `soh_measured`, `soh_source` | measured if a `capacity_ah` column is given; RUL and risk then use it, otherwise they use `soh_pred` and `soh_source` says `predicted` |
| `rul_pred`, `rul_unit`, `rul_note` | model (`rul_*.joblib`, or `rul_*_censoring_aware.joblib` with `rul_variant="censoring_aware"`); 0 once SoH ≤ 80 %; the note says which variant was used and its caveat |
| `risk_rule`, `risk_from_rul_model` | rule with thresholds from `risk_*.json`; RUL terciles |
| `anomaly_flag`, `anomaly_score`, `anomaly_explanation` | Isolation Forest on z-scores within *this* battery's history; needs ≥ 10 rows |
| `data_quality_flags`, `inputs_outside_training_range`, `missing_inputs` | input checks; missing columns are imputed with the training median |
| `energy_retention`, `energy_retention_source` | measured if an `energy_wh` column is given, otherwise a labelled capacity proxy |
| `range_km_assumed` | only with measured energy **and** an explicit `RangeAssumptions`; not validated |

All trend features are causal, so adding later rows never changes earlier predictions (tested).

Processed data is ~1.3 GB (`data/processed/`); the extracted `.mat` files can be deleted after `build_dataset` if space matters.
