"""Create a private deployment bundle after real inference smoke checks pass."""
import hashlib
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from scripts.prepare_render import validate_runtime

ROOT = Path(__file__).resolve().parents[1]


def main():
    validate_runtime()
    from src.data.ev_loaders import find_ev_files
    files = list((ROOT / 'models').glob('ev_*.joblib')) + list((ROOT / 'models').glob('ev_*.json'))
    files += [ROOT / 'reports/results/ev_split_units.csv', *find_ev_files().values()]
    output = ROOT / 'dist/voltara-runtime.zip'
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, 'w', ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with output.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    print(f'Runtime bundle: {output}')
    print(f'SHA256: {sha}')


if __name__ == '__main__':
    main()
