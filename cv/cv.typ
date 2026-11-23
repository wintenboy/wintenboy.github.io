#let d = json("../content/profile.json")
#let ink = rgb("#142942")
#let blue = rgb("#215479")
#let muted = rgb("#425671")
#let rule-color = rgb("#a9bad7")
#set document(title: d.name + " | Curriculum Vitae", author: d.name)
#set page(paper: "a4", margin: (x: 17mm, top: 14mm, bottom: 15mm), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt, fill: ink)
#set par(leading: .62em, justify: false)
#show link: set text(fill: blue)
#let section(title) = block(above: 14pt, below: 7pt, width: 100%, sticky: true, stack(
  spacing: 4pt,
  text(size: 13pt, weight: "bold", fill: blue, title),
  line(length: 100%, stroke: .55pt + rule-color),
))
#let entry(item) = block(above: 7pt, below: 7pt, breakable: false, stack(
  spacing: 3pt,
  grid(columns: (1fr, auto), column-gutter: 10pt,
    text(weight: "bold", item.name), text(size: 9.5pt, fill: muted, item.period)),
  text(item.detail),
))
#let research-entry(item) = block(above: 9pt, below: 9pt, breakable: false)[
  #text(weight: "bold", item.title)
  #linebreak()
  #item.credit
  #linebreak()
  #emph(item.venue), #item.year. #item.details. #item.role.
  #if item.href != "" { [ #link(item.href)[#item.link_label]] }
]
#let patent(item) = block(above: 8pt, below: 8pt, breakable: false)[
  #text(weight: "bold", item.title)
  #linebreak()
  #item.inventors
  #linebreak()
  #item.status, #link(item.href)[#item.number]. #item.period.
]
#let course(item) = entry((
  name: item.name + " (" + item.code + ")",
  detail: item.role + ", " + item.institution,
  period: item.period,
))
#let project(item) = block(above: 9pt, below: 9pt, breakable: false, stack(
  spacing: 3pt,
  grid(columns: (1fr, auto), column-gutter: 8pt,
    text(weight: "bold", item.name), text(size: 9.5pt, fill: muted, item.period)),
  [#text(fill: muted)[#item.organization · #item.role] #linebreak() #item.summary],
))
#let records(items, render) = items.map(render).join(v(8pt))

#stack(
  spacing: 8pt,
  text(size: 36pt, weight: "bold", d.name),
  [#text(size: 12pt)[#d.role at #d.university] #linebreak() #text(fill: muted)[#d.lab]],
  d.links.map(item => link(item.href)[#item.label]).join([ · ]),
)

#section("Professional experience")
#records(d.professional_experience, entry)
#section("Education")
#records(d.education, entry)
#section("Research experience")
#records(d.experience, entry)
#section("Research interests")
#d.topics.join(" / ")
#section("Publications and presentations")
#records(d.research, research-entry)
#section("Patents")
#records(d.patents, patent)

#pagebreak()

#section("Projects")
#records(d.projects, project)
#section("Teaching")
#records(d.teaching, course)
#section("Mentoring")
#records(d.mentoring, entry)
#section("Honors")
#records(d.recognition, entry)
