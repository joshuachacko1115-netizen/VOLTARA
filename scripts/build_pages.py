"""Build only public frontend assets for GitHub Pages; never publish the repo root."""
from pathlib import Path
import shutil
import json
import os
from urllib.parse import urlsplit
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "webapp" / "static"
    output = ROOT / "dist" / "pages"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("app.css", "app.js"):
        shutil.copyfile(source / name, output / name)
    html = (source / "index.html").read_text(encoding="utf-8")
    html = html.replace('href="/static/app.css"', 'href="./app.css"')
    html = html.replace('src="/static/app.js"', 'src="./app.js"')
    html = html.replace('href="/"', 'href="./"')
    backend = os.environ.get('VOLTARA_API_BASE_URL', '').rstrip('/')
    if backend:
        parsed = urlsplit(backend)
        if not re.fullmatch(r'https://[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::[0-9]{1,5})?', backend) or parsed.path or parsed.query or parsed.fragment:
            raise ValueError('VOLTARA_API_BASE_URL must be an HTTPS origin without path, query or credentials.')
        config = 'window.VOLTARA_API_BASE_URL = ' + json.dumps(backend).replace('<', '\\u003c') + ';'
    else:
        config = 'window.VOLTARA_PAGES_PREVIEW = true;'
    html = html.replace('<script src="./app.js"',
                        '<script>' + config + '</script>\n<script src="./app.js"')
    notice = '''<aside class="pages-note" aria-label="Hosting information">
      <h2>VOLTARA frontend preview</h2>
      <p>This GitHub Pages site displays the application interface. Live assessments,
      CSV uploads and model inference require the Python server and are disabled here.</p>
      <a href="https://github.com/joshuachacko1115-netizen/VOLTARA#github-actions-and-windows-setup">Run the full application locally</a>
      · <a href="https://github.com/joshuachacko1115-netizen/VOLTARA/actions">Builds and downloadable source</a>
    </aside>'''
    if not backend:
        html = html.replace('<main class="wrap">', '<main class="wrap">' + notice)
    # No API links should produce misleading 404 downloads on static hosting.
    if backend:
        html = html.replace('href="/api/', 'href="' + backend + '/api/')
    else:
        html = re.sub(r'<a href="/api/example.csv[^>]*>.*?</a>',
                      '<span>Available in the Python application</span>', html, flags=re.S)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / ".nojekyll").touch()
    assert '/static/' not in html and 'href="/api/' not in html
    assert {p.name for p in output.iterdir()} == {"index.html", "app.js", "app.css", ".nojekyll"}
    print(output)


if __name__ == "__main__":
    main()
