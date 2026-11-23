"""Build the GitHub Pages site and CV from the same profile data."""
import argparse
from hashlib import sha256
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
ENGLISH_LABELS = {
    "language_label": "Language",
    "sections_label": "Sections",
    "skip_label": "Skip to content",
    "projects_label": "Projects",
    "research_label": "Research",
    "patents_label": "Patents",
    "teaching_label": "Teaching",
    "education_label": "Education",
    "experience_label": "Research experience",
    "cv_label": "CV",
    "portrait_label": "Portrait of Seungyeol Ryu",
    "full_size_label": "View full size",
    "top_label": "Back to top",
    "motion_label": "Motion",
}
TRANSLATION_KEYS = {
    "professional_experience": "name",
    "research": "href",
    "patents": "number",
    "projects": "id",
    "teaching": "code",
    "education": "name",
    "experience": "name",
}


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


def project_figure(visual, caption, alt, prefix, full_size):
    file = Path("assets/projects") / Path(visual["file"]).name
    if not (SITE / file).is_file():
        raise FileNotFoundError(SITE / file)
    href = escape(prefix + file.as_posix())
    width, height = int(visual["width"]), int(visual["height"])
    preview = f' data-preview="{escape(visual["preview"])}"' if visual.get("preview") else ""
    return f'''
          <figure class="project-figure">
            <a class="project-visual" href="{href}"{preview}>
              <img src="{href}" alt="{escape(alt)}" width="{width}" height="{height}" loading="lazy" decoding="async">
              <span class="project-visual-action">{escape(full_size)}</span>
            </a>
            <figcaption>{escape(caption)}</figcaption>
          </figure>'''


def project_record(item, prefix, full_size):
    metadata = f'{item["role"]}, {item["organization"]}'
    description = item.get("case_study", item["summary"])
    source = item.get("source")
    source_link = ""
    if source:
        label = item.get("source_label", source["label"])
        source_link = f'\n            <p class="source-link">{anchor(label, source["href"])}</p>'
    figure = ""
    if visual := item.get("visual"):
        alt = item.get("visual_alt", visual["alt"])
        if filename := item.get("visual_file"):
            visual = {**visual, "file": filename}
        figure = project_figure(visual, item["caption"], alt, prefix, full_size)
    return f'''<li class="project">
          <div class="project-copy">
            <p class="metadata">{escape(metadata)}</p>
            <h3>{escape(item["name"])}</h3>
            <p class="project-date">{escape(item["period"])}</p>
            <p class="summary">{escape(description)}</p>{source_link}
          </div>{figure}
        </li>'''


def course_record(item):
    title = f'{item["name"]} ({item["code"]})'
    metadata = f'{item["role"]}, {item["institution"]}'
    return record(title, item["period"], metadata)


def timeline(items):
    return "\n".join(record(item["name"], item["period"], item["detail"]) for item in items)


def current_role(item, korean):
    name = anchor(item["name"], item["href"]) if item.get("href") else escape(item["name"])
    detail = escape(item["detail"])
    return f"<li>{name} {detail}</li>" if korean else f"<li>{detail} at {name}</li>"


def translate_profile(profile, translation):
    localized = {**profile, **translation["profile"]}
    for section, key in TRANSLATION_KEYS.items():
        overrides = translation.get(section, {})
        localized[section] = [
            {**item, **overrides.get(item.get(key), {})} for item in profile[section]
        ]
    return localized


def asset_version():
    digest = sha256()
    for file in ("site.css", "projects.css", "site.js"):
        digest.update((SITE / "assets" / file).read_bytes())
    return digest.hexdigest()[:12]


def render_site(profile, language="en", translation=None):
    source = (ROOT / "site/index.template.html").read_text(encoding="utf-8")
    korean = language == "ko"
    prefix = "../" if korean else ""
    labels = {**ENGLISH_LABELS, **translation["labels"]} if korean else ENGLISH_LABELS
    if korean:
        profile = translate_profile(profile, translation)
    social_links = "\n".join(anchor(item["label"], item["href"]) for item in profile["links"])
    current_roles = [
        current_role(item, korean)
        for item in profile["professional_experience"]
        if item["period"] == "Present"
    ]
    academic_role = (f'{profile["university"]} {profile["role"]}' if korean
                     else f'{profile["role"]} at {profile["university"]}')
    current_roles.append(f'<li>{escape(academic_role)}</li>')
    name = profile["korean_name"] if korean else profile["name"]
    animated_name = ("" if korean else " ").join(
        f'<span class="name-word" style="--word-index: {index}"><span>{escape(word)}</span></span>'
        for index, word in enumerate(list(name) if korean else name.split())
    )
    return Template(source).substitute(
        language=language,
        asset_prefix=prefix,
        asset_version=asset_version(),
        english_path=prefix + "index.html",
        korean_path=prefix + "ko/index.html",
        english_current='' if korean else ' aria-current="page"',
        korean_current=' aria-current="page"' if korean else '',
        name=escape(name),
        animated_name=animated_name,
        korean_name=escape(profile["korean_name"]),
        description=escape(f'{name}. {profile["bio"]}'),
        current_roles="\n".join(current_roles),
        bio=escape(profile["bio"]),
        social_links=social_links,
        research="\n".join(research_record(item) for item in profile["research"] if item["featured"]),
        patents="\n".join(patent_record(item) for item in profile["patents"]),
        projects="\n".join(project_record(item, prefix, labels["full_size_label"]) for item in profile["projects"]),
        teaching="\n".join(course_record(item) for item in profile["teaching"]),
        education=timeline(profile["education"]),
        experience=timeline(profile["experience"]),
        **{key: escape(value) for key, value in labels.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typst", help="Path to the Typst CLI, if it isn't on PATH.")
    parser.add_argument("--site-only", action="store_true", help="Build the homepage without recompiling the CV.")
    args = parser.parse_args()
    profile = json.loads((ROOT / "content/profile.json").read_text(encoding="utf-8"))
    korean = json.loads((ROOT / "content/profile.ko.json").read_text(encoding="utf-8"))
    pages = {
        SITE / "index.html": render_site(profile),
        SITE / "ko/index.html": render_site(profile, "ko", korean),
    }
    if not args.site_only:
        compiler = args.typst or shutil.which("typst")
        if compiler is None:
            parser.exit(1, "Typst CLI not found. Install it or pass --typst.\n")
        with TemporaryDirectory(dir=ROOT) as build_dir:
            output = Path(build_dir) / "cv.pdf"
            subprocess.run([compiler, "compile", "--root", str(ROOT), str(ROOT / "cv/cv.typ"), str(output)], check=True)
            (SITE / PDF).parent.mkdir(parents=True, exist_ok=True)
            output.replace(SITE / PDF)
        print(f"Built {SITE / PDF}")
    for path, page in pages.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(page, encoding="utf-8", newline="\n")
        print(f"Built {path}")


if __name__ == "__main__":
    main()
