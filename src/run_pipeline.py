"""Run the whole pipeline end to end.

    python -m src.run_pipeline              # everything (NASA ~6 min + EV ~20 min on a laptop)
    python -m src.run_pipeline --no-deep    # skip the CNN-LSTM
    python -m src.run_pipeline --only-ev    # only the EV pack datasets (step 9)
"""
import argparse

from src.data import build_dataset
from src.features import build_features
from src.models import (anomaly, censored_rul, ev_intervals, range_estimation, risk, train_ev, train_rul,
                        train_soh)
from src.visualization import eda


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--only-ev", action="store_true")
    args = ap.parse_args()
    if args.only_ev:
        train_ev.run()
        ev_intervals.run()
        return
    build_dataset.run()        # 1. extract ZIPs, parse .mat -> data/processed/interim
    build_features.run()       # 2. features, quality flags, SoH/RUL targets
    eda.run()                  # 3. EDA figures
    train_soh.run()            # 4. SoH models
    train_rul.run(deep=not args.no_deep)  # 5. RUL models
    censored_rul.run()         # 5b. censoring-aware RUL comparison (optional model)
    anomaly.run()              # 6. Isolation Forest
    risk.run()                 # 7. risk classes (needs RUL predictions)
    range_estimation.run()     # 8. energy retention / illustrative range
    train_ev.run()             # 9. EV pack datasets: cleaning, features, models, final test
    ev_intervals.run()         #    and their 90 % prediction intervals


if __name__ == "__main__":
    main()
