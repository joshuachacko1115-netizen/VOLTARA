"""Build only public frontend assets for GitHub Pages; never publish the repo root."""
from pathlib import Path
import shutil

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
    html = html.replace('<script src="./app.js"',
                        '<script>window.VOLTARA_PAGES_PREVIEW = true;</script>\n<script src="./app.js"')
    notice = '''<aside class="pages-note" aria-label="Hosting information">
      <h2>VOLTARA frontend preview</h2>
      <p>This GitHub Pages site displays the application interface. Live assessments,
      CSV uploads and model inference require the Python server and are disabled here.</p>
      <a href="https://github.com/joshuachacko1115-netizen/VOLTARA#github-actions-and-windows-setup">Run the full application locally</a>
      · <a href="https://github.com/joshuachacko1115-netizen/VOLTARA/actions">Builds and downloadable source</a>
    </aside>'''
    html = html.replace('<main class="wrap">', '<main class="wrap">' + notice)
    # No API links should produce misleading 404 downloads on static hosting.
    import re
    html = re.sub(r'<a href="/api/example.csv[^>]*>.*?</a>',
                  '<span>Available in the Python application</span>', html, flags=re.S)
    (output / "index.html").write_text(html, encoding="utf-8")
    (output / ".nojekyll").touch()
    assert '/static/' not in html and 'href="/api/' not in html
    assert {p.name for p in output.iterdir()} == {"index.html", "app.js", "app.css", ".nojekyll"}
    print(output)


if __name__ == "__main__":
    main()
