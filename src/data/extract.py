"""Extract the two NASA battery archives found in data/raw/.

Both downloads are ZIPs that contain further ZIPs (one per experiment group).
This module:
  1. finds every ZIP in data/raw/ (whatever its extension) and identifies which
     NASA dataset it is from its top-level folder name;
  2. extracts each inner ZIP into data/processed/extracted/<dataset>/<group>/;
  3. writes a manifest.csv with the source, size and MD5 of every extracted file.

The original files in data/raw/ are only read, never modified.

Usage:
    python -m src.data.extract              # .mat files + READMEs only
    python -m src.data.extract --include-r  # also the R (.Rda) copies
    python -m src.data.extract --force      # re-extract even if already done
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from src.config import DATASETS, EXTRACTED_DIR, RAW_DIR

# The .Rda files duplicate the .mat contents, so they are skipped by default.
KEEP_SUFFIXES = {".mat", ".txt", ".m", ".html", ".rmd"}


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def identify_archive(path: Path) -> str | None:
    """Return the dataset key for a NASA archive, or None if it is not one."""
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path) as z:
        tops = {PurePosixPath(n).parts[0] for n in z.namelist() if n}
    for key, meta in DATASETS.items():
        if meta["zip_top_folder"] in tops:
            return key
    return None


def find_archives(raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for p in sorted(raw_dir.iterdir()):
        if not p.is_file():
            continue
        key = identify_archive(p)
        if key is None:
            continue
        if key in found:
            raise RuntimeError(f"Two archives for '{key}': {found[key].name} and {p.name}")
        found[key] = p
    return found


def _safe_target(base: Path, member: str) -> Path:
    """Resolve a ZIP member path under base, rejecting path traversal."""
    target = (base / member).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise ValueError(f"Unsafe path in archive: {member}")
    return target


def _group_name(inner_zip_name: str) -> str:
    # "5. Battery Data Set/3. BatteryAgingARC_25-44.zip" -> "BatteryAgingARC_25-44"
    stem = PurePosixPath(inner_zip_name).stem
    prefix, _, rest = stem.partition(". ")
    return rest if prefix.isdigit() and rest else stem


def extract_archive(outer: Path, dataset_key: str, out_root: Path, include_r: bool) -> list[dict]:
    keep = KEEP_SUFFIXES | ({".rda"} if include_r else set())
    dataset_dir = out_root / dataset_key
    records = []
    with zipfile.ZipFile(outer) as z_outer:
        inner_names = [n for n in z_outer.namelist() if n.lower().endswith(".zip")]
        for inner_name in inner_names:
            group = _group_name(inner_name)
            group_dir = dataset_dir / group
            print(f"[{dataset_key}] {group}")
            # Inner ZIPs need random access, so copy each one to a temp file first.
            with tempfile.TemporaryDirectory(dir=out_root) as tmp:
                tmp_zip = Path(tmp) / "inner.zip"
                with z_outer.open(inner_name) as src, open(tmp_zip, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                with zipfile.ZipFile(tmp_zip) as z_inner:
                    for info in z_inner.infolist():
                        if info.is_dir() or Path(info.filename).suffix.lower() not in keep:
                            continue
                        # Flatten the inner folder layout (e.g. "<name>/data/Matlab/RW9.mat").
                        target = _safe_target(group_dir, Path(info.filename).name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with z_inner.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst, 1 << 20)
                        records.append({
                            "dataset": dataset_key,
                            "group": group,
                            "file": target.name,
                            "relative_path": target.relative_to(out_root).as_posix(),
                            "source_archive": outer.name,
                            "source_member": f"{inner_name}::{info.filename}",
                            "size_bytes": target.stat().st_size,
                            "md5": md5sum(target),
                        })
    return records


def run(include_r: bool = False, force: bool = False) -> pd.DataFrame:
    manifest_path = EXTRACTED_DIR / "manifest.csv"
    if manifest_path.exists() and not force:
        print(f"Already extracted (see {manifest_path}). Use --force to redo.")
        return pd.read_csv(manifest_path)

    archives = find_archives()
    missing = set(DATASETS) - set(archives)
    if missing:
        raise FileNotFoundError(f"No archive found in {RAW_DIR} for: {sorted(missing)}")

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for key, path in archives.items():
        print(f"Archive {path.name} -> {key}")
        records += extract_archive(path, key, EXTRACTED_DIR, include_r)

    manifest = pd.DataFrame(records)
    manifest.to_csv(manifest_path, index=False)
    mats = manifest[manifest.file.str.endswith(".mat")]
    dup = mats[mats.duplicated(["dataset", "file"], keep=False)]
    print(f"\nExtracted {len(manifest)} files ({len(mats)} .mat). Manifest: {manifest_path}")
    if not dup.empty:
        same = dup.groupby(["dataset", "file"]).md5.nunique().eq(1)
        print("Cells present in more than one group:",
              ", ".join(f"{f} ({'identical' if ok else 'DIFFERENT'})" for (_, f), ok in same.items()))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-r", action="store_true", help="also extract the .Rda copies")
    ap.add_argument("--force", action="store_true", help="re-extract even if a manifest exists")
    args = ap.parse_args()
    run(include_r=args.include_r, force=args.force)


if __name__ == "__main__":
    main()
