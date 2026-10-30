# VOLTARA — Battery health, degradation and RUL from NASA data

[![VOLTARA CI](https://github.com/joshuachacko1115-netizen/VOLTARA/actions/workflows/ci.yml/badge.svg)](https://github.com/joshuachacko1115-netizen/VOLTARA/actions/workflows/ci.yml)

Repository: [joshuachacko1115-netizen/VOLTARA](https://github.com/joshuachacko1115-netizen/VOLTARA).
The first hosted CI run [passed on 2026-10-30](https://github.com/joshuachacko1115-netizen/VOLTARA/actions/runs/41034676855),
including dependency checks, compilation, tests, source packaging and artifact upload.
Dataset/model-dependent tests may skip in CI; the local pretrained-model failures
documented below remain separate from this successful source validation.

## GitHub Actions and Windows setup

This is a Python battery-health application, with a Streamlit dashboard and a
Starlette web app. Run commands from the directory containing this README.
The macOS `.venv` supplied with the project is not usable on Windows.

```powershell
python -m venv .venv-ci
.\.venv-ci\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv-ci\Scripts\python.exe -m pip check
.\.venv-ci\Scripts\python.exe -m compileall -q src dashboard webapp tests scripts
.\.venv-ci\Scripts\python.exe -m pytest -q -ra --junitxml=dist/test-results.xml
.\.venv-ci\Scripts\python.exe scripts/package.py
```

On Linux/macOS use `.venv-ci/bin/python` in place of
`.\.venv-ci\Scripts\python.exe`. `requirements-ci.txt` installs the application
and tests including XGBoost (imported by EV training), without PyTorch or Jupyter. Install `requirements.txt`
instead to include those optional training/notebook dependencies. Dependencies
are bounded where compatibility matters but are not a fully frozen lockfile.
In VS Code select `.venv-ci/Scripts/python.exe` on Windows. The editor
configuration points to `.venv-ci`, and the debugger uses the selected interpreter.

Start either interface with that interpreter:

```powershell
.\.venv-ci\Scripts\python.exe -m streamlit run dashboard/app.py
.\.venv-ci\Scripts\python.exe -m uvicorn webapp.server:app --port 8600
```

Inference needs trained models. Training needs the original datasets under
`data/raw`; see the existing dataset instructions below. Run
`python -m src.run_pipeline --no-deep` using the configured interpreter to train
without the optional CNN-LSTM. Training writes models, data and reports; CI does
not retrain on large external datasets.

### What CI validates and distributes

`.github/workflows/ci.yml` runs on pushes to `main`, pull requests targeting
`main`, and manual dispatch. It uses Python 3.13 on Ubuntu, read-only repository
permissions, dependency caching, dependency checks, Python bytecode compilation,
and the complete existing pytest suite. Tests include valid synthetic inputs,
invalid inputs, causal features, and leakage checks. Existing tests requiring
absent datasets/models skip with reasons (`-ra`); passing CI is not evidence that
those skipped integration tests or full model training passed. No tests have
been weakened to accommodate CI.

After every preceding step succeeds, `scripts/package.py` creates
`dist/voltara-source.zip`. GitHub uploads it with the JUnit report under
**Actions > VOLTARA CI > successful run > Artifacts** for 14 days.
This is a runnable Python source distribution, not a native executable or a
pretrained demo: extract it, install dependencies, and supply datasets/train
models before inference. Packaging includes source, frontend assets, tests,
workflow and requirements using an explicit file allowlist. It excludes data,
models, reports, virtual environments, Git metadata and secrets. No cloud
deployment, GitHub Release or package-registry publication is configured.

The corrected `/data/` ignore pattern ignores only root datasets and permits
`src/data` source modules to be committed. Generated models/reports are ignored
for new files. Previously tracked models/reports remain tracked to preserve
history and existing work; review them before committing. `.gitignore` cannot
remove secrets or binaries already present in history.

### Create and connect the empty repository

Local validation on 2026-10-30 (Windows, Python 3.13): dependency compatibility,
Python compilation, workflow YAML/structural checks and package-content checks
passed. The extracted source package ran **38 passed, 59 skipped**. The original
folder with its supplied models/data ran **77 passed, 20 failed**, including
XGBoost `input stream corrupted` errors when loading saved models and downstream
dashboard/API failures. Matching scikit-learn to the original 1.9.1 environment
did not resolve those errors. No model files or application behavior were changed
to hide these failures. The full pretrained application is therefore not validated.
The isolated incorrect-assertion demonstration correctly exited with status 1,
and the copied test was restored. GitHub-hosted execution succeeded after the
approved push to `main` (see the verified run above). Reports from local validation are
under ignored `dist/` and are not the historical `reports/test_results.txt`.

Create a repository named **VOLTARA** at <https://github.com/new>. Choose your
visibility and leave README, `.gitignore`, and license initialization unchecked.
This project is now connected to the repository linked above as `origin`;
the instructions here describe its initial setup. No existing remote was overwritten.

Before committing, review existing local changes as well as the new CI files:

```powershell
Set-Location D:\VOLTARA\VOLTARA
git status --short
git diff --stat
git diff -- .gitignore README.md
git remote -v
```

The original branch `dashboard` is retained. `main` now tracks `origin/main`.
For future updates, review changes and commit on `main`.
Stage application sources explicitly so new generated binaries are not added:

```powershell
git switch main
git add .gitignore README.md requirements.txt requirements-ci.txt .github scripts src tests dashboard webapp .streamlit/config.toml
git diff --cached --stat
git diff --cached --check
git commit -m "Add GitHub Actions CI and validated source artifact packaging"
```

Review the staged diff before the commit; this includes your existing application
changes, not just CI. Do not use `git add .` to sweep in unrelated artifacts.
After the empty remote has been inspected and connected as `origin`, and only
when you approve publication, the first push command is:

```powershell
git push -u origin main
```

If the remote already contains commits, stop and inspect the histories before
merging; never reset history or force-push. Existing tracked model/report changes
are deliberately not staged by the commands above.

### Verify and demonstrate

After the approved push, open **Actions > VOLTARA CI**, inspect dependency,
compilation and test logs, check the JUnit skipped count, then download and
extract the artifact. Manual **Run workflow** becomes available after this
workflow exists on the default branch. A hosted run must succeed before calling
the GitHub CI/CD integration verified.

For a failure demonstration, create a disposable branch from `main`, change
the expected value in `tests/test_preprocessing.py::test_nan_run_lengths` from
`[0, 2, 2, 0, 1, 0]` to `[0, 0, 0, 0, 0, 0]`, and run that test locally.
With your approval, publish that demo branch and open a pull request targeting
`main`: its test step must fail and artifact upload must not run. Restore the
assertion with a normal commit to show the succeeding run. Do not merge the
intentional failure or change application functionality for the demonstration.

Machine-learning pipeline for lithium-ion battery **State of Health (SoH)**, **Remaining Useful Life (RUL)**,
degradation analysis, anomaly detection, degradation-risk classes and a clearly separated range-estimation
component. No dashboard or deployment (by design, for now).

Every number below comes from an experiment run on the data in `data/raw/`. Regenerate them with the commands
in *Run it*; the CSV/PNG outputs are in `reports/`.

## 1. What the data actually is

*Sections 1–7 cover the two NASA archives. The two synthetic EV pack datasets added later (CSV folders in `data/raw/`) are in section 8.*

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
| `dashboard/` | `app.py` (Streamlit UI), `helpers.py` (input parsing, summaries, charts) |
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
| | Gradient Boosting | 32.3 | 50.1 | 0.73 |
| | XGBoost (saved model since the retrain) | **31.5** | 52.7 | 0.70 |
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

* On Randomized Usage, learned models cut the error of the mean baseline by ~60 %; **the learned models are within noise of each
  other** (MAE 31.5–33.4; fold-to-fold spread ≈ 24). The CNN-LSTM is not better than a linear regression here, so the simpler model is preferable.
  It was trained once per fold with default settings and no tuning.
* The naive trend-extrapolation baseline is *worse than predicting the mean*: SoH curves are not linear, and a short-window slope is noisy.
* Battery Aging RUL **does not generalise across the 6-cell cohort** (R² < 0). B0036 (EOL at cycle 182) and B0030 (4 A, 43 °C, EOL at 29) have lives very
  different from the 2 A cells (59–85), and there is only one example of each condition. Restricting to the four same-protocol cells gives R² 0.78,
  but that is 4 cells; treat it as indicative, not conclusive. Linear Regression diverged on BA RUL (MAE 225), which is why it is not in the table.
* **Tried and not adopted (2026-10-28): discharge-voltage-curve features** for Battery Aging RUL: voltage at 60, 300 and 600 s, time to 3.8 and 3.6 V,
  mean and spread of discharge voltage, duration and energy, raw and as a causal 5-cycle median. With all 6 cells, leave-one-cell-out R² fell from
  −0.21 to between −0.36 and −0.50. B0030 and B0036 carry most of the error (MAE ≈ 40 and 70); each is the only cell under its conditions, so no feature
  can fix that. On the 4 same-protocol cells Gradient Boosting improved slightly (MAE 9.0 → 7.6, R² 0.77 → 0.81), within noise for 4 cells.
  The remedy is more cells per condition, not more features.
* RUL error depends on where the cell is in its life (`reports/results/rul_*_error_by_true_rul.csv`).
* **Model change at the retrain (2026-10-28).** XGBoost now loads in the environment, so it joined the comparison. The selection rule
  (lowest leave-one-cell-out MAE) picks it for Randomized Usage: MAE 31.5 vs 32.3 for Gradient Boosting, but RMSE 52.7 vs 50.1 and R² 0.70 vs 0.73.
  That is a noise-level difference in either direction (fold spread ± 24); `models/rul_rw.joblib` is now XGBoost. Every other number reproduced
  exactly (Battery Aging RUL, all SoH models, anomalies, risk accuracy). The per-cell analysis in this section and the next one was made with
  Gradient Boosting in v0.1; the tables show the retrained numbers.
* Censored cells: for RW26 and the BA cells the final model predicts RUL ≥ the known lower bound at 100 % of rows (B0029: 92 %), but for RW27 only 25 % and RW28 73 %
  (`rul_censored_cells_check.csv`; v0.1 with Gradient Boosting: 17 % and 64 %). Those cells never reached EOL, so this is not scored as an error, but it is a limitation:
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
| Randomized Usage (XGBoost) | labelled only (current model) | **31.5** EFC | 0.70 | 68 % | 9.7 EFC |
| | + censored, bound | 31.7 | **0.71** | 70 % | 8.9 |
| | + censored, Buckley-James | 34.8 | 0.68 | **76 %** | **5.3** |
| Battery Aging (RF) | labelled only (current model) | **33.2** cycles | −0.21 | 97 % | 0.02 |
| | + censored, bound | 35.9 | −0.23 | 97 % | 0.01 |
| | + censored, Buckley-James | 45.1 | −0.81 | 100 % | 0 |

Per cell, v0.1 Gradient Boosting run (current values in `rul_censoring_bounds_per_cell.csv`): RW27 goes from 17 % to 42 % of rows above the bound, with its shortfall reduced from 25.5 to 18.2 EFC. RW28 goes from 64 % to 100 %.

What it costs:
* **RW:** labelled-cell MAE rises by 3.3 EFC with Buckley-James (v0.1 with Gradient Boosting: 1.9). That is well inside the fold-to-fold spread (± 24), but it is consistent. 13 of 25 labelled cells improve.
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
python -m pytest -q                 # 90 tests (most need the trained models and processed data)
```

macOS/Linux: `source .venv/bin/activate`. XGBoost needs `brew install libomp` on macOS; without it, the code falls back to the other models.
v0.1 was produced without XGBoost; since the 2026-10-28 retrain it is part of every comparison (see the RUL section for the one model that changed).
PyTorch runs single-threaded in the CNN-LSTM: with XGBoost loaded in the same process, its multi-threaded kernels deadlocked on macOS.
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

### Dashboard (`dashboard/app.py`) — local demo

```bash
streamlit run dashboard/app.py        # opens http://localhost:8501
```

`python dashboard/app.py` also works (it relaunches itself under Streamlit). In VS Code, *Run and Debug → "VOLTARA dashboard"*
(`.vscode/launch.json`) uses the project's `.venv`. After editing any dashboard file, stop the server (Ctrl+C) and start it again:
Streamlit reloads `app.py` but not the modules it imports.

The look (Inter / Space Grotesk fonts, header, step cards, metric tiles, SoH gauge with the 80 % end-of-life mark) is in `dashboard/style.py`.
It only styles Streamlit's own widgets with CSS, so behaviour and the tests are unchanged. Colours come from the report palette
(`src/visualization/style.py`), and status is always shown with words or icons as well as colour.

* **Left: user inputs.** Choose the battery dataset, then the measurements: an example NASA cell, with a slider to predict "as of" any earlier
  cycle; an uploaded CSV (a blank template can be downloaded); or a blank table to type into. Every value can be edited.
  Under *Options* are the RUL model variant and the vehicle assumptions for km.
* **Middle:** **Predict**.
* **Right: system outputs.** Predicted and measured SoH, RUL, the risk class (colour, icon and text), anomalies, and energy retention / range.
  Measured, modelled and assumed values are labelled as such.
* **Below:** trend charts (SoH, RUL coloured by risk, energy retention), the flagged anomalies, input checks, the full result table (CSV download),
  and model notes.

The dashboard holds no model logic: it calls `VoltaraPredictor.predict` and only formats the result. `tests/test_dashboard.py` runs the app
headlessly (Streamlit `AppTest`) and checks that what it shows equals the inference module's output exactly, for both datasets.
It also covers uploaded CSVs, blank and partial inputs, non-numeric values, unfamiliar values, the censoring-aware option, range assumptions and stale results.

Input handling (in `src/inference/predict.py`, so the CLI behaves the same):
* Non-numeric values, infinite or implausibly large numbers (|x| > 1e9), an empty table, or a missing / negative ageing column give a readable
  error naming the column and example values, not a crash. A file from the other dataset gets an error saying which dataset to switch to.
* Column names are matched ignoring case, spaces and hyphens (`" Cycle Number "` = `cycle_number`). `""`, `-`, `n/a`, `NaN`, `null`, `missing`
  etc. count as missing. Rows that share an ageing value are flagged in `data_quality_flags`.
* Each row reports `soh_input_coverage` and `rul_input_coverage`, the share of model inputs actually measured. `prediction_status` is
  `ok`, `partial inputs` or `insufficient inputs` (below 50 %). With insufficient inputs, the output would mostly describe the training medians
  used for imputation, so the dashboard withholds SoH, RUL and risk instead of showing them.
* Values outside the 1st–99th percentile of the training data are listed per row (`inputs_outside_training_range`).

CSV upload (`dashboard/helpers.py`): comma, semicolon or tab separators; decimal commas (with `;` or tab); UTF-8 with or without BOM, or
Windows-1252; binary or empty files are rejected. The vehicle assumptions for range are bounded to plausible values.

Fixed while building the dashboard:
* The inference module never passed the capacity-based anomaly features (`capacity_ah_2v7`, `cap_resid_prev5`; RW: `soh`, `cap_resid_prev3`)
  to the Isolation Forest, so capacity drops could not be flagged. It now derives them from `capacity_ah` with the same helper as training.
  On valid cycles it flags exactly the pipeline's anomalies (tested).
* CSV floats are parsed exactly (`float_precision="round_trip"`): the default parser changed the last bit of some values, enough to flip a tree split.

Processed data is ~1.3 GB (`data/processed/`); the extracted `.mat` files can be deleted after `build_dataset` if space matters.

## 8. EV pack datasets (synthetic) — added 2026-10-28

Two more datasets sit in `data/raw/` as folders with CSVs. They are found by their column header, not their name (`src/data/ev_loaders.py`).

| | EV Battery Health & Degradation (`ev_degradation`) | EV Battery Health & Telematics (`ev_telematics`) |
|---|---|---|
| Rows / units | 50,000 observations of 2,600 battery packs (7–37 each) | 20,000 trips of 5,374 vehicles (1–22 each) |
| Level | pack, 40–100 kWh, 4 chemistries (NMC, LFP, NCA, LMO) | vehicle, 30–75 kWh, NMC / LFP |
| Targets | SoH %, RUL (cycles to 80 %), failure risk (Low/Med/High), degradation category | SoH % (integer), failure type, maintenance needed, fire risk |
| Data problems | 3–8 % missing in every feature, ~1.5 % injected outliers in 6 sensors | none missing; labels drawn at random from a risk score |

**Both are synthetic** (their READMEs say so). They are pack-level with SoH in % of nominal, so they cannot be pooled with the NASA cells, and
results on them are not evidence about real batteries.

### What was found and done (`src/preprocessing/ev_cleaning.py`, `src/features/ev_features.py`)

* **Leaks removed.** `capacity_fade_pct` is exactly 100 − SoH in every row (checked); with it, SoH MAE would be 0.11 instead of 0.59.
  `Range_Remaining_km × Wh/km ÷ (SoC × capacity)` equals SoH within ±1 %, so the range field re-encodes the target (MAE 0.58 instead of 1.01).
  Both columns are renamed `leak_*` and never used as inputs; tests enforce it.
* **Missing values.** Pack attributes are constant per battery, so 1,900–3,500 missing values per column are filled from the battery's other rows
  (conflicts raise an error). 2,864 missing ages are rebuilt as observation date − manufacture date (the two agree exactly wherever both exist).
  Everything else stays missing and is handled by the models (XGBoost / HistGB natively; median imputation inside the fold for the others).
* **Time order.** Observation ids increase with age inside every battery (0 exceptions), so they order each battery's history.
  Cycle count still goes *down* between consecutive observations 19 % of the time: the readings are noisy by construction.
* **Outliers.** 39 physically impossible values (e.g. 83.7 °C ambient) are set to missing. A causal robust check (each reading against the median of
  the battery's previous 5) flags statistical outliers. The injected outliers can be identified by an artefact (they are written with 11–17 decimals),
  which is used only to audit the detector: precision 0.77, recall 0.21. Most injected values (×0.5–×2) are inside the normal
  observation-to-observation spread. Blanking the flagged values made no measurable difference (dev MAE 0.589 vs 0.590), so they are kept and
  reported as `flag_outlier_*` columns.
* **Features.** Equivalent full cycles (throughput ÷ capacity), cycles per day, days since service. Per battery, causal history only: running means of
  the stress factors (temperature, depth of discharge, fast-charge share, SoC, …), 3-point rolling medians and previous values of resistance and cycles,
  resistance growth versus the first readings, and the resistance-vs-cycles slope so far. Telematics: vehicle age, Wh/km, C-rates, DC-fast-charge share,
  temperature rise, heat history. Some physics features (charge energy ÷ (capacity × DoD)) carried no signal in these generators and were dropped.

### Protocol

20 % of batteries / vehicles (seeded, stratified by chemistry) were set aside **before any modelling** and scored once at the end. On the other 80 %,
5-fold cross-validation grouped by battery / vehicle compared a baseline, linear, random forest, HistGB and XGBoost, plus a 12-trial XGBoost search.
The winner was refitted on all development data. Intervals are 95 % bootstrap over test units. Code: `src/models/train_ev.py`.

### Results on the held-out test set

| Target | Model | Result | Baseline |
|---|---|---|---|
| **SoH, degradation** | XGBoost (tuned) | **MAE 0.56 points** [0.55–0.58], R² **0.989**; 84.6 % within ±1, 98.3 % within ±2 | MAE 5.36 |
| RUL, degradation | Random Forest | MAE 255 cycles [247–264], R² 0.926 | MAE 984 |
| Failure risk (3 classes) | XGBoost (tuned) | **accuracy 95.4 %** [94.7–96.0], balanced 85.7 %, AUC 0.990 | 86.1 % (all "Low") |
| Degradation category | XGBoost | **accuracy 98.5 %** [98.1–98.8], balanced 90.2 %, AUC 0.998 | 93.0 % |
| **SoH, telematics** | XGBoost (tuned) | **MAE 0.99 points** [0.97–1.02], R² **0.930**; 60 % within ±1, 90 % within ±2 | MAE 3.63 |
| Failure type (4 classes) | Logistic regression | accuracy 85.0 %, balanced **34.7 %**, AUC 0.90 | 82.3 % |
| Maintenance required | XGBoost (tuned) | accuracy 86.7 %, balanced **52.2 %**, AUC **0.65** | 87.3 % |
| Fire risk (Medium+High vs Low) | Random Forest | accuracy 88.8 %, balanced **52.6 %**, AUC **0.58** | 89.0 % |
| Energy use (Wh/km), pre-trip inputs | HistGB | MAE 0.38 Wh/km, R² 0.9998 | MAE 35.5 |
| Manual-entry SoH (age, odometer, consumption, capacity) | XGBoost (tuned) | MAE 1.00 (chemistry known); 2.16, R² 0.77 (unknown) | MAE 5.36 |
| Manual-entry RUL | XGBoost (tuned) | MAE 315 cycles, R² 0.88 (known); **R² 0.12 (unknown)** | MAE 984 |

How to read it:
* **There is a ceiling, and the models are at it.** SoH carries ~0.9 points of observation-to-observation noise; RUL jumps by ~290 cycles between consecutive
  observations of the same battery. Even the *true* SoH and RUL predict the risk label only ~95.7 % of the time and the degradation category ~98.2 %.
  Tuning XGBoost improved SoH by 0.01 (0.590 → 0.581, dev CV); the families within 0.02 of each other are equivalent.
* **Feature engineering helped modestly:** degradation SoH dev MAE 0.670 (raw) → 0.621 (static fill + recovered ages) → 0.590 (history features);
  test 0.564. Telematics 1.026 → 1.010. Most important (XGBoost gain): resistance (rolling median 0.31, growth 0.12) and cycles (0.19);
  for telematics the cycle count (0.54 + 0.34 for its history).
* **Telematics risk labels are not learnable:** maintenance and fire-risk accuracy equals always guessing "no" (AUC 0.58–0.65). The dataset README
  says they are random draws from a score. They are saved for completeness; use their probabilities, not their labels.
* **Consumption is a formula in this generator** (R² 0.9998): the model recovers it. It says nothing about real-world consumption.
* SoH error is largest below 80 % (MAE 1.04), where data are sparse; by chemistry, LFP 0.50 to LMO 0.71.
* **Pooling the two EV datasets does not help** (shared features only; `ev_cross_dataset.csv`): degradation test MAE 0.964 → 0.964,
  telematics 1.017 → 1.012. A model from one dataset transfers poorly to the other (MAE 2.87 and 1.69), so each dataset has its own models.
  The NASA models are unaffected by these datasets.

### 90 % prediction intervals (`src/models/ev_intervals.py`)

Each SoH and RUL estimate comes with a range. The ranges use split-conformal calibration on the out-of-fold residuals of the dev units, with the
same GroupKFold folds as training, and are computed separately in 4 bins of the predicted value. They were checked once on the held-out test units:

| Model | Test coverage (target 90 %) | Mean width |
|---|---|---|
| SoH, degradation (full history) | 90.1 % | 2.3 points |
| SoH, telematics | 89.9 % | 3.7 points |
| SoH, manual entry: chemistry known / unknown | 89.9 % / 89.4 % | 4.1 / 8.6 points |
| RUL, degradation | 89.8 % | 983 cycles |
| RUL, manual entry: chemistry known / unknown | 89.7 % / 89.8 % | 1,250 / 3,260 cycles |

The manual-entry RUL range without the chemistry is about 3,300 cycles wide, which is why the dashboards do not show that RUL.
Run with `python -m src.models.ev_intervals` (about 2 min). It writes `interval` into each model's meta and `reports/results/ev_interval_coverage.csv`.
The dashboards show the range under each estimate. The web app no longer shows the telematics failure-type and maintenance
probabilities, which are close to chance on held-out vehicles.

Saved: `models/ev_*.joblib` + `.json` (features, CV and test scores, intervals), `models/ev_outlier_masker.json`. Results: `reports/results/ev_*.csv`,
`ev_cleaning_report.json`; figures: `reports/figures/ev/`. Inference: `src/inference/ev_predict.py` (`EVPredictor`, full models on raw rows),
`src/inference/ev_assess.py` (dashboard assessment).

```bash
python -m src.models.train_ev                       # all EV models, ~20-40 min (python -m src.run_pipeline --only-ev does the same)
python -m src.models.ev_intervals                   # then calibrate the 90 % intervals (~2 min)
python -m src.models.train_ev --only-manual-entry   # just the manual-entry models, from the saved feature table
python -m src.inference.ev_predict --dataset ev_degradation --unit BATT_00001
tail -f logs/ev_train.log                           # follow a background run (logs/ is git-ignored)
```

### Dashboard page "ev assessment" (`dashboard/pages/ev_assessment.py`)

Opens from the sidebar of `streamlit run dashboard/app.py`.
* **Input:** a held-out sample battery or vehicle, an uploaded CSV in either dataset's format, or manual entry: vehicle type (car / scooter / bus),
  capacity, current charge, age, odometer, consumption, chemistry, climate.
* **Output:** SoH with its model error, remaining range, RUL (cycles and years at the vehicle's cycling rate), anomalies (Detected / None / Not assessed),
  degradation-risk badge, the SoH trend and insights.
* **Labelled honestly:** range is capacity × SoH × charge ÷ consumption (arithmetic, not a validated model). Anomalies are rule checks with stated thresholds
  (fast-charge peak > 55 °C, pack > 45 °C, > 8 °C above ambient, voltage ±5 % from nominal, resistance +50 %, SoH drop > 3 points) plus the outlier
  flags. Insights are sentences from fixed rules, not a language model. Manual entry cannot assess anomalies (one reading). Without the chemistry
  it withholds RUL (R² 0.12), and scooters and buses get an extrapolation warning (training packs were 40–100 kWh).
* `tests/test_ev_dashboard.py` checks that the page shows exactly what `ev_assess` returns.

### Web app "EV Intelligence" (`webapp/`)

A standalone site with the same assessment, built as a custom HTML/CSS/JS front end on a small Starlette API.
```bash
python -m uvicorn webapp.server:app --port 8600      # http://localhost:8600
# deep links: /?source=manual&run=1  ·  /?run=1 (first held-out sample)
```
* `webapp/server.py`: `/api/meta`, `/api/sample`, `/api/profile` (uploaded CSV), `/api/assess`, `/api/example.csv`. Every number comes from
  `src.inference.ev_assess`; the server only converts it to JSON.
* For a sample or upload, capacity, age and odometer are shown from the data and locked. The odometer is an estimate, because neither dataset
  records one. Charge and consumption are prefilled from the data and only change the range when edited.
* **Light theme by default**; dark is opt-in with the toggle in the top bar (remembered in that browser). Reports always print light.
* A **Report** button opens the print dialog with a print layout (inputs, results, chart, findings). Choose "Save as PDF".
* `webapp/static/`: input form and output dashboard following the design mock-ups. It adds an SoH gauge, a trend chart (or usable energy for a
  single reading) and the anomaly list. Motion includes a flow-field background, count-ups, a line-draw chart and staggered reveals; all of it
  respects `prefers-reduced-motion`.
* `tests/test_webapp.py` checks that the API returns exactly what `ev_assess` computes for manual entry, both samples and an uploaded CSV.
