# Seungyeol Ryu portfolio

An academic homepage and a Typst CV built from the same records.

## Update the portfolio

1. Edit `content/profile.json`. The `research` list holds journal articles and presentations. Set `featured` to `true` to include a research entry on the homepage. Update `patents`, `projects`, and `teaching` to change the corresponding sections in both the homepage and CV. Use one record per course.
2. Run `python tools/build.py` using Python 3.10+ and Typst 0.15.1. If Typst is outside PATH, pass its path with `--typst`.
3. Review `docs/index.html` and `docs/cv/seungyeol-ryu-cv.pdf` before committing the updated content and generated files. For a local HTTP preview, run `python -m http.server 8000 --directory docs`.

`site/index.template.html` controls the page structure, `docs/assets/site.css` controls the styles, and `cv/cv.typ` controls the PDF layout.

For a project, `summary` is the short CV entry and `case_study` is the homepage description. Put the screenshot or figure in `docs/assets/projects/` and record its `file`, pixel `width` and `height`, and `alt` description under `visual`. Describe its source in `caption`.

## GitHub Pages

The publishing target is `wintenboy/wintenboy.github.io`. The site is ready to serve from `main` and `/docs` once the public repository and Pages publishing source are set up. Keep records meant for public release in this repository.
