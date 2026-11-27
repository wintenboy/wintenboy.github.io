"""Build the GitHub Pages site and CV from the same profile data."""
import argparse
from hashlib import sha256
import json
from html import escape
from pathlib import Path
from posixpath import relpath
import shutil
from string import Template
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
PDF = Path("cv/seungyeol-ryu-cv.pdf")
SITE_URL = "https://wintenboy.github.io/"
SECTIONS = ("education", "research", "projects", "patents", "teaching")
ENGLISH_LABELS = {
    "language_label": "Language",
    "sections_label": "Sections",
    "skip_label": "Skip to content",
    "projects_label": "Projects",
    "research_label": "Research",
    "patents_label": "Patents",
    "teaching_label": "Teaching",
    "education_label": "Education",
    "profile_label": "Education and research experience",
    "experience_label": "Research experience",
    "cv_label": "CV",
    "portrait_label": "Portrait of Seungyeol Ryu",
    "full_size_label": "View full size",
    "top_label": "Back to top",
    "patent_record_label": "Patent record",
    "co_inventor_label": "Co-inventor",
    "view_all": {
        "education": "View education and experience",
        "research": "View all research",
        "projects": "View all projects",
        "patents": "View all patents",
        "teaching": "View all courses",
    },
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


def record(title, date, metadata, summary="", source_link="", detail_link=""):
    paragraphs = [f'<p class="metadata">{escape(metadata)}</p>']
    if summary:
        paragraphs.append(f'<p class="summary">{escape(summary)}</p>')
    if source_link:
        paragraphs.append(f'<p class="source-link">{source_link}</p>')
    details = "\n            ".join(paragraphs)
    heading = anchor(title, detail_link) if detail_link else escape(title)
    level = "h2" if detail_link else "h3"
    return f'''<li class="record">
          <{level}>{heading}</{level}>
          <p class="record-date">{escape(date)}</p>
          <div class="record-details">
            {details}
          </div>
        </li>'''


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


def figure_for(item, prefix, full_size):
    if visual := item.get("visual"):
        alt = item.get("visual_alt", visual["alt"])
        if filename := item.get("visual_file"):
            visual = {**visual, "file": filename}
        return project_figure(visual, item["caption"], alt, prefix, full_size)
    return ""


def project_record(item, href, prefix, full_size):
    metadata = f'{item["role"]}, {item["organization"]}'
    figure = figure_for(item, prefix, full_size)
    return f'''<li class="project">
          <div class="project-copy">
            <p class="metadata">{escape(metadata)}</p>
            <h2><a href="{escape(href)}">{escape(item["name"])}</a></h2>
            <p class="project-date">{escape(item["period"])}</p>
            <p class="summary">{escape(item["summary"])}</p>
          </div>{figure}
        </li>'''


def entry_info(section, item, labels):
    if section == "research":
        meta = " · ".join((item["venue"], item["details"], item["role"]))
        reference = anchor(item["link_label"], item["href"]) if item["href"] else ""
        return item["title"], item["year"], meta, item["summary"], reference
    if section == "patents":
        meta = " · ".join((item["status"], item["number"], labels["co_inventor_label"]))
        reference = anchor(item.get("link_label", labels["patent_record_label"]), item["href"])
        return item["title"], item["period"], meta, "", reference
    if section == "teaching":
        title = f'{item["name"]} ({item["code"]})'
        return title, item["period"], f'{item["role"]}, {item["institution"]}', "", ""
    source = item.get("source")
    link = anchor(item.get("source_label", source["label"]), source["href"]) if source else ""
    return item["name"], item["period"], f'{item["role"]}, {item["organization"]}', item["case_study"], link


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


def page_path(language, route=""):
    root = Path("ko") if language == "ko" else Path()
    return root / route / "index.html"


def page_link(page, language, route=""):
    return relpath(page_path(language, route).as_posix(), page.parent.as_posix())


def page_url(language, route):
    parts = (["ko"] if language == "ko" else []) + ([route] if route else [])
    return SITE_URL + ("/".join(parts) + "/" if parts else "")


def render_shell(template, profile, labels, language, version, route, title, description, body):
    page = page_path(language, route)
    section = route.split("/")[0]
    navigation = []
    for name in SECTIONS:
        current = ""
        if section == name:
            current = ' aria-current="page"' if route == name else ' aria-current="location"'
        href = escape(page_link(page, language, name))
        navigation.append(f'<a href="{href}"{current}>{escape(labels[name + "_label"])}</a>')
    return template.substitute(
        language=language,
        asset_prefix="../" * len(page.parent.parts),
        asset_version=version,
        english_path=escape(page_link(page, "en", route)),
        korean_path=escape(page_link(page, "ko", route)),
        english_url=page_url("en", route),
        korean_url=page_url("ko", route),
        english_current=' aria-current="page"' if language == "en" else "",
        korean_current=' aria-current="page"' if language == "ko" else "",
        page_title=escape(title),
        korean_name=escape(profile["korean_name"]),
        home_href=escape("#top" if not route else page_link(page, language)),
        navigation="\n        ".join(navigation),
        description=escape(description),
        body=body,
        **{key: escape(labels[key]) for key in ("skip_label", "language_label", "sections_label", "top_label")},
    )


def section_items(profile, section):
    if section == "research":
        return [item for item in profile[section] if item["featured"]]
    return profile[section]


def render_index(profile, labels, language):
    page = page_path(language)
    blocks = []
    for section in SECTIONS:
        section_href = page_link(page, language, section)
        title = labels["profile_label"] if section == "education" else labels[section + "_label"]
        items = section_items(profile, section)
        previews = []
        for item in items[:3]:
            if section == "education":
                heading = f'<strong>{escape(item["name"])}</strong>'
                meta = item["detail"]
            else:
                name, date, meta, _, _ = entry_info(section, item, labels)
                if section == "research":
                    meta = f'{item["venue"]} · {item["role"]}'
                elif section == "patents":
                    name = item["name"]
                elif section == "teaching":
                    meta = date
                heading = anchor(name, page_link(page, language, f'{section}/{item["id"]}'))
            previews.append(f'<li>{heading}<p>{escape(meta)}</p></li>')
        anchor_id = "profile" if section == "education" else section
        list_id = ' id="experience"' if section == "education" else ""
        view_all = (
            f'<p class="index-more">{anchor(labels["view_all"][section], section_href)}</p>'
            if len(items) > len(previews) else ""
        )
        blocks.append(f'''<section id="{anchor_id}" class="content-section" aria-labelledby="{anchor_id}-title">
      <h2 id="{anchor_id}-title"><a href="{section_href}">{escape(title)}</a></h2>
      <div><ul class="index-list"{list_id}>{"".join(previews)}</ul>{view_all}</div>
    </section>''')
    return "\n    ".join(blocks)


def render_home(template, profile, labels, language):
    korean = language == "ko"
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
    return template.substitute(
        asset_prefix="../" if korean else "",
        name=escape(name),
        animated_name=animated_name,
        current_roles="\n".join(current_roles),
        bio=escape(profile["bio"]),
        social_links=social_links,
        cv_label=escape(labels["cv_label"]),
        portrait_label=escape(labels["portrait_label"]),
        index=render_index(profile, labels, language),
    )


def article_sections(sections):
    parts = []
    for heading, *paragraphs in sections:
        content = "".join(f"<p>{escape(p)}</p>" for p in paragraphs)
        parts.append(f'<section><h2>{escape(heading)}</h2>{content}</section>')
    return "\n".join(parts)


def route_heading(title, meta="", reference="", back=""):
    meta_html = f'<p class="metadata">{escape(meta)}</p>' if meta else ""
    ref_html = f'<p class="source-link">{reference}</p>' if reference else ""
    return f'<header class="route-heading">{back}<h1>{escape(title)}</h1>{meta_html}{ref_html}</header>'


def render_section(profile, labels, language, section, details):
    page = page_path(language, section)
    title = labels["profile_label"] if section == "education" else labels[section + "_label"]
    heading = route_heading(title)
    if section == "education":
        content = f'''<div class="article-body">
        <section><h2>{escape(labels["education_label"])}</h2><ul class="records">{timeline(profile["education"])}</ul></section>
        <section id="experience"><h2>{escape(labels["experience_label"])}</h2><ul class="records">{timeline(profile["experience"])}</ul></section>
        {article_sections(details["education"])}
      </div>'''
    else:
        rows = []
        for item in section_items(profile, section):
            href = page_link(page, language, f'{section}/{item["id"]}')
            if section == "projects":
                rows.append(project_record(item, href, "../" * len(page.parent.parts), labels["full_size_label"]))
            else:
                name, date, meta, summary, reference = entry_info(section, item, labels)
                rows.append(record(name, date, meta, summary, reference, href))
        list_class = "project-records" if section == "projects" else "records"
        content = f'<ul class="{list_class}">{"".join(rows)}</ul>'
    return title, f'<div class="section-page">{heading}{content}</div>'


def render_detail(item, labels, language, section, details):
    route = f'{section}/{item["id"]}'
    page = page_path(language, route)
    title, date, meta, lead, reference = entry_info(section, item, labels)
    back_url = escape(page_link(page, language, section))
    back = f'<a class="back-link" href="{back_url}">← {escape(labels[section + "_label"])}</a>'
    heading = route_heading(title, f"{date} · {meta}", reference=reference, back=back)
    figure = figure_for(item, "../" * len(page.parent.parts), labels["full_size_label"])
    body = f'<article class="detail-page">{heading}{figure}<div class="article-body">{article_sections(details[item["id"]])}</div></article>'
    return route, title, lead, body


def render_site(profile, translation, copy):
    shell = Template((ROOT / "site/index.template.html").read_text(encoding="utf-8"))
    home = Template((ROOT / "site/home.template.html").read_text(encoding="utf-8"))
    version = asset_version()
    pages = {}
    for language in ("en", "ko"):
        local = translate_profile(profile, translation) if language == "ko" else profile
        labels = {**ENGLISH_LABELS, **translation["labels"]} if language == "ko" else ENGLISH_LABELS
        details = copy[language]
        name = local["korean_name"] if language == "ko" else local["name"]
        routes = [("", name, local["bio"], render_home(home, local, labels, language))]
        for section in SECTIONS:
            title, body = render_section(local, labels, language, section, details)
            routes.append((section, title, f"{name} · {title}", body))
            if section != "education":
                routes.extend(render_detail(item, labels, language, section, details)
                              for item in section_items(local, section))
        for route, title, description, body in routes:
            pages[SITE / page_path(language, route)] = render_shell(
                shell, local, labels, language, version, route, f"{title} | {name}" if route else name,
                description or f"{name} · {title}", body,
            )
    return pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typst", help="Path to the Typst CLI, if it isn't on PATH.")
    parser.add_argument("--site-only", action="store_true", help="Build the site without recompiling the CV.")
    args = parser.parse_args()
    profile = json.loads((ROOT / "content/profile.json").read_text(encoding="utf-8"))
    korean = json.loads((ROOT / "content/profile.ko.json").read_text(encoding="utf-8"))
    details = json.loads((ROOT / "content/details.json").read_text(encoding="utf-8"))
    pages = render_site(profile, korean, details)
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
