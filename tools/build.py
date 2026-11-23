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


def anchor(label, href):
    return f'<a href="{escape(href)}">{escape(label)}</a>'


def record(title, date, metadata, summary="", source_link=""):
    paragraphs = [f'<p class="metadata">{escape(metadata)}</p>']
    if summary:
        paragraphs.append(f'<p class="summary">{escape(summary)}</p>')
    if source_link:
        paragraphs.append(f'<p class="source-link">{source_link}</p>')
    details = "\n            ".join(paragraphs)
    return f'''<li class="record">
          <h3>{escape(title)}</h3>
          <p class="record-date">{escape(date)}</p>
          <div class="record-details">
            {details}
          </div>
        </li>'''


def research_record(item):
    metadata = ". ".join((item["venue"], item["details"], item["role"])) + "."
    link = anchor(item["link_label"], item["href"]) if item["href"] else ""
    return record(item["title"], item["year"], metadata, item["summary"], link)


def patent_record(item):
    link = anchor(item["number"], item["href"])
    return record(item["name"], item["period"], item["status"], source_link=link)


def project_record(item):
    metadata = f'{item["role"]}, {item["organization"]}'
    return record(item["name"], item["period"], metadata, item["summary"])


def course_record(item):
    title = f'{item["name"]} ({item["code"]})'
    metadata = f'{item["role"]}, {item["institution"]}'
    return record(title, item["period"], metadata)


def timeline(items):
    return "\n".join(record(item["name"], item["period"], item["detail"]) for item in items)


def render_site(profile):
    source = (ROOT / "site/index.template.html").read_text(encoding="utf-8")
    social_links = "\n".join(anchor(item["label"], item["href"]) for item in profile["links"])
    current_roles = [
        f'{item["detail"]} at {item["name"]}'
        for item in profile["professional_experience"]
        if item["period"] == "Present"
    ]
    current_roles.append(f'{profile["role"]} at {profile["university"]}')
    return Template(source).substitute(
        name=escape(profile["name"]),
        korean_name=escape(profile["korean_name"]),
        description=escape(f'{profile["name"]}. ' + ". ".join(current_roles) + "."),
        current_roles="\n".join(f"<li>{escape(role)}</li>" for role in current_roles),
        bio=escape(profile["bio"]),
        social_links=social_links,
        research="\n".join(research_record(item) for item in profile["research"] if item["featured"]),
        patents="\n".join(patent_record(item) for item in profile["patents"]),
        projects="\n".join(project_record(item) for item in profile["projects"]),
        teaching="\n".join(course_record(item) for item in profile["teaching"]),
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
