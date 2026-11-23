"""Build the GitHub Pages site and CV from the same profile data."""
import argparse
import json
from html import escape
from pathlib import Path
import shutil
from string import Template
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
PDF = Path("cv/seungyeol-ryu-cv.pdf")


def anchor(label, href, css_class=""):
    css = f' class="{escape(css_class)}"' if css_class else ""
    return f'<a{css} href="{escape(href)}">{escape(label)}</a>'


def research_article(item):
    parts = [item["venue"], item["details"], item["role"]]
    description = " · ".join(escape(part) for part in parts if part)
    link = anchor(item["link_label"], item["href"], "text-link") if item["href"] else ""
    return f'''<article class="research-item">
        <div class="research-meta">
          <span class="research-kind">{escape(item["type"])}</span>
          <span class="research-date">{escape(item["year"])}</span>
        </div>
        <div class="research-content">
          <h3>{escape(item["title"])}</h3>
          <p class="credits">{description}</p>
          <p class="summary">{escape(item["summary"])}</p>
          {link}
        </div>
      </article>'''


def project_card(item):
    return f'''<article class="project">
          <p class="project-meta"><strong>{escape(item["organization"])}</strong><span>{escape(item["period"])}</span></p>
          <h3>{escape(item["name"])}</h3>
          <p>{escape(item["summary"])}</p>
        </article>'''


def timeline(items):
    rows = [
        f'<div><dt>{escape(item["name"])}</dt>'
        f'<dd>{escape(item["detail"])}<p class="timing">{escape(item["period"])}</p></dd></div>'
        for item in items
    ]
    return "\n".join(rows)


def render_site(profile):
    source = (ROOT / "site/index.template.html").read_text(encoding="utf-8")
    social_links = " ".join(anchor(item["label"], item["href"], "quiet") for item in profile["links"])
    return Template(source).substitute(
        name=escape(profile["name"]),
        korean_name=escape(profile["korean_name"]),
        description=escape(f'{profile["name"]}, {profile["role"]} at {profile["university"]}.'),
        role=escape(profile["role"]),
        university=escape(profile["university"]),
        bio=escape(profile["bio"]),
        lab=escape(profile["lab"]),
        social_links=social_links,
        topics="".join(f"<li>{escape(topic)}</li>" for topic in profile["topics"]),
        research="\n".join(research_article(item) for item in profile["research"] if item["featured"]),
        projects="\n".join(project_card(item) for item in profile["projects"]),
        education=timeline(profile["education"]),
        experience=timeline(profile["experience"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typst", help="Path to the Typst CLI, if it isn't on PATH.")
    args = parser.parse_args()
    compiler = args.typst or shutil.which("typst")
    if compiler is None:
        parser.exit(1, "Typst CLI not found. Install it or pass --typst.\n")
    profile = json.loads((ROOT / "content/profile.json").read_text(encoding="utf-8"))
    page = render_site(profile)
    with TemporaryDirectory(dir=ROOT) as build_dir:
        output = Path(build_dir) / "cv.pdf"
        subprocess.run([compiler, "compile", "--root", str(ROOT), str(ROOT / "cv/cv.typ"), str(output)], check=True)
        (SITE / PDF).parent.mkdir(parents=True, exist_ok=True)
        output.replace(SITE / PDF)
    (SITE / "index.html").write_text(page, encoding="utf-8", newline="\n")
    print(f"Built {SITE / 'index.html'} and {SITE / PDF}")


if __name__ == "__main__":
    main()
