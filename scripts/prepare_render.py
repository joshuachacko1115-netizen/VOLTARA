"""Install a trusted, hash-verified runtime bundle and reject unusable models.

The ZIP uses project-relative paths under models/, data/raw/ and reports/results/.
Only provide bundles you trust: joblib models can execute code when loaded.
"""
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 2_000_000_000


def allowed(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        return False
    return ((len(path.parts) == 2 and path.parts[0] == 'models'
             and path.name.startswith('ev_') and path.suffix in {'.joblib', '.json'})
            or (len(path.parts) >= 3 and path.parts[:2] == ('data', 'raw') and path.suffix == '.csv')
            or name == 'reports/results/ev_split_units.csv')


def main():
    url = os.environ.get('VOLTARA_ASSET_BUNDLE_URL', '')
    digest = os.environ.get('VOLTARA_ASSET_BUNDLE_SHA256', '').lower()
    if not url or not digest:
        raise RuntimeError('Set VOLTARA_ASSET_BUNDLE_URL and VOLTARA_ASSET_BUNDLE_SHA256 to a verified runtime bundle.')
    if urlsplit(url).scheme != 'https' or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Runtime bundle requires an HTTPS URL and a 64-character SHA256 digest.')
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / 'runtime.zip'
        sha = hashlib.sha256()
        size = 0
        with urlopen(url, timeout=60) as response, archive.open('wb') as output:
            if urlsplit(response.url).scheme != 'https':
                raise ValueError('Runtime download redirected to an insecure URL.')
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError('Runtime bundle exceeds the download limit.')
                sha.update(chunk)
                output.write(chunk)
        if sha.hexdigest() != digest:
            raise ValueError('Runtime bundle SHA256 does not match; refusing to load it.')
        with ZipFile(archive) as bundle:
            files = [item for item in bundle.infolist() if not item.is_dir()]
            if sum(item.file_size for item in files) > MAX_BYTES or any(not allowed(item.filename) for item in files):
                raise ValueError('Runtime bundle contains unexpected paths or exceeds the extraction limit.')
            for item in files:
                target = ROOT / item.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open('wb') as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
    validate_runtime()


def validate_runtime(models_dir=None):
    # Validate the real API paths; fail the build rather than deploy broken inference.
    import sys
    sys.path.insert(0, str(ROOT))
    if models_dir is not None:
        import src.config as config
        config.MODELS_DIR = Path(models_dir).resolve()
    from webapp.server import meta, run_assessment, samples
    meta(None)
    run_assessment({'source': 'manual', 'capacity_kwh': 60, 'soc_pct': 80,
                    'age_years': 3, 'odometer_km': 45000, 'consumption_kwh_100km': 16,
                    'chemistry': 'NMC', 'climate': 'Unknown'})
    for dataset in ('ev_degradation', 'ev_telematics'):
        units = samples(dataset)
        if not units:
            raise RuntimeError(f'No held-out samples available for {dataset}.')
        run_assessment({'source': 'sample', 'dataset': dataset, 'unit': units[0]})
    print('Runtime bundle validated: manual assessment and both dataset sample paths.')


if __name__ == '__main__':
    main()
