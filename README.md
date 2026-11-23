# Seungyeol Ryu portfolio

An English and Korean portfolio and a Typst CV built from the same records. The site is published at [wintenboy.github.io](https://wintenboy.github.io/).

## Update the portfolio

1. Edit `content/profile.json` for the shared profile, research, projects, patents, and teaching records. Use one record per course. Set `featured` to `true` to include a research entry on the homepage.
2. Edit `content/details.json` for the bilingual long-form copy. Match each detail to its route slug. Update Korean labels and translations of shared records in `content/profile.ko.json` using the fields in `TRANSLATION_KEYS` in `tools/build.py`. Use [im-not-ai's Korean editing guide](https://github.com/epoko77-ai/im-not-ai/tree/c07df14d9f68da482ff5183591808a7d132d4a71/codex/skills/humanize-korean) when drafting or revising Korean copy. Preserve roles, dates, technical terms, and source claims.
3. Run `python tools/build.py --site-only` with Python 3.10+ to regenerate the site. Run `python tools/build.py` with Typst 0.15.1 to rebuild the PDF too. If Typst is outside PATH, pass its path with `--typst`.
4. Preview with `python -m http.server 8000 --directory docs`. Review the updated pages in both languages and click the language switch from one detail page before committing the source and generated files.

`site/index.template.html` defines the shared layout, `site/home.template.html` defines the homepage, `docs/assets/site.css` controls the styles, and `cv/cv.typ` controls the PDF layout.

For a project, `summary` is the short entry used in the project list and CV. `case_study` provides the detail page's metadata description. Put the screenshot or figure in `docs/assets/projects/` and record its `file`, pixel `width` and `height`, and `alt` description under `visual`. Describe its source in `caption`.

For a patent, use `name` for the homepage label and copy `title` from the patent document. The patent index, detail page, and English CV use the full title.

## GitHub Pages

GitHub Pages publishes from `/docs` on the `main` branch of `wintenboy/wintenboy.github.io`. Keep records meant for public release in this repository.
