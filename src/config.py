"""Project-wide paths and constants.

Every path is built with pathlib relative to the project root, so the code runs
unchanged on Windows, macOS and Linux.
"""
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_DIR = PROCESSED_DIR / "extracted"
INTERIM_DIR = PROCESSED_DIR / "interim"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42

# The two NASA archives are recognised by the top-level folder inside the ZIP,
# not by their file name (the downloads may still be called "*.crdownload").
DATASETS = {
    "battery_aging": {
        "zip_top_folder": "5. Battery Data Set",
        "title": "NASA PCoE Li-ion Battery Aging Data Set (B0005-B0056)",
    },
    "randomized_usage": {
        "zip_top_folder": "11. Randomized Battery Usage Data Set",
        "title": "NASA Randomized Battery Usage Data Set (RW1-RW28)",
    },
}

# --------------------------------------------------------------------------- #
# Target definitions (see README "Target definitions" for the reasoning)
# --------------------------------------------------------------------------- #
# End of life = SoH falling to 80 %. This is a project choice: the NASA READMEs use
# 70 % (1.4 Ah) for some groups and 80 % (1.6 Ah) for others.
EOL_SOH = 0.80

# Battery Aging: the README states a 2 Ah rated capacity, so SoH = C / 2.0 Ah.
# Capacity is integrated to a common 2.7 V cutoff - the "Capacity" field is
# documented as "for discharge till 2.7V" and 2.7 V is the highest cutoff used,
# so every complete discharge reaches it.
BA_RATED_CAPACITY_AH = 2.0
BA_CAPACITY_CUTOFF_V = 2.7

# Randomized Usage: no rated capacity is documented, so SoH is relative to the
# cell's first valid reference-discharge capacity (1 A to 3.2 V, measured in all cells).
RW_REFERENCE_CURRENT_A = 1.0
RW_REFERENCE_CUTOFF_V = 3.2

# Battery Aging experiment metadata transcribed from the dataset READMEs.
# profile: "cc" constant current | "square_wave" 0.05 Hz 4 A square wave |
#          "variable_cc" constant current that changes between cycles (1/2/4 A)
_BA_GROUPS = [
    # cells, cutoff voltages, discharge profile, nominal current (A), README ambient
    (("B0005", "B0006", "B0007", "B0018"), (2.7, 2.5, 2.2, 2.5), "cc", 2.0, "24"),
    (("B0025", "B0026", "B0027", "B0028"), (2.0, 2.2, 2.5, 2.7), "square_wave", 4.0, "24"),
    (("B0029", "B0030", "B0031", "B0032"), (2.0, 2.2, 2.5, 2.7), "cc", 4.0, "43"),
    (("B0033", "B0034"), (2.0, 2.2), "cc", 4.0, "24"),
    (("B0036",), (2.7,), "cc", 2.0, "24"),
    (("B0038", "B0039", "B0040"), (2.2, 2.5, 2.7), "variable_cc", np.nan, "24/44"),
    (("B0041", "B0042", "B0043", "B0044"), (2.0, 2.2, 2.5, 2.7), "variable_cc", np.nan, "4"),
    (("B0045", "B0046", "B0047", "B0048"), (2.0, 2.2, 2.5, 2.7), "cc", 1.0, "4"),
    (("B0049", "B0050", "B0051", "B0052"), (2.0, 2.2, 2.5, 2.7), "cc", 2.0, "4"),
    (("B0053", "B0054", "B0055", "B0056"), (2.0, 2.2, 2.5, 2.7), "cc", 2.0, "4"),
]
# Cells used for Battery Aging SoH / RUL models: constant-current discharge at
# 24 or 43 degC. Excluded, with reasons documented in README:
#   B0025-B0028 square-wave discharge (different load profile);
#   B0038-B0044 load current changes between cycles, so capacity is not comparable
#               from one cycle to the next;
#   B0041-B0056 at 4 degC: capacity is dominated by the cold test (many runs hit
#               the cutoff almost immediately, which the README calls unexplained).
BA_MODEL_CELLS = ["B0005", "B0006", "B0007", "B0018", "B0029", "B0030", "B0031", "B0032",
                  "B0033", "B0034", "B0036"]

# --------------------------------------------------------------------------- #
# EV pack datasets (synthetic, CSV). Recognised by their header, not their file or
# folder name. Both are pack/vehicle level and use SoH in % of nominal capacity, so
# they are not comparable with the NASA single-cell lab data and are modelled apart.
# --------------------------------------------------------------------------- #
EV_DIR = PROCESSED_DIR / "ev"
EV_DATASETS = {
    "ev_degradation": {
        "signature": {"battery_id", "observation_id", "state_of_health_pct", "remaining_useful_life_cycles"},
        "title": "EV Battery Health & Degradation (synthetic, 2,600 packs, 50,000 observations)",
        "unit_col": "battery_id",
    },
    "ev_telematics": {
        "signature": {"Vehicle_ID", "Trip_ID", "SOH_pct", "Range_Remaining_km"},
        "title": "EV Battery Health & Telematics (synthetic, 5,374 vehicles, 20,000 trips)",
        "unit_col": "vehicle_id",
    },
}
# Share of batteries / vehicles held out as the final test set. They are chosen once
# (seeded, stratified by chemistry) before any modelling choice and are only scored at the end.
EV_TEST_FRACTION = 0.20
EV_EOL_SOH_PCT = 80.0

BA_CELL_INFO = {
    cell: {"cutoff_v": v, "profile": prof, "nominal_current_a": cur, "readme_ambient_c": amb,
           "protocol_group": "+".join(cells)}
    for cells, volts, prof, cur, amb in _BA_GROUPS
    for cell, v in zip(cells, volts)
}
