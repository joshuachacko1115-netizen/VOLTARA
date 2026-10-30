"""Create a private deployment bundle after real inference smoke checks pass."""
import hashlib
import argparse
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.prepare_render import validate_runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models')
    args = parser.parse_args()
    models = args.models_dir.resolve()
    validate_runtime(models)
    from src.data.ev_loaders import find_ev_files
    files = [(p, 'models/' + p.name) for pattern in ('ev_*.joblib', 'ev_*.json') for p in models.glob(pattern)]
    files += [(p, p.relative_to(ROOT).as_posix()) for p in
              [ROOT / 'reports/results/ev_split_units.csv', *find_ev_files().values()]]
    output = ROOT / 'dist/voltara-runtime.zip'
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as bundle:
        for path, archive_name in files:
            bundle.write(path, archive_name)
    with output.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'Runtime bundle: {output}')
    print(f'SHA256: {sha}')


if __name__ == '__main__':
    main()
