"""Build a source distribution without datasets, models, or local secrets."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / name for name in ("README.md", "requirements.txt", "requirements-ci.txt")]
    for directory, suffixes in {
        "src": {".py"}, "dashboard": {".py"},
        "webapp": {".py", ".html", ".css", ".js"},
        "tests": {".py"}, "scripts": {".py"},
        ".github/workflows": {".yml"},
    }.items():
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    output = ROOT / "dist" / "voltara-source.zip"
    output.parent.mkdir(exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, Path("VOLTARA") / path.relative_to(ROOT))
    print(output)


if __name__ == "__main__":
    main()
