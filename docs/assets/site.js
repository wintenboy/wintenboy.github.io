(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const progressBar = document.querySelector(".reading-progress");
  const intro = document.querySelector(".intro");
  const name = intro?.querySelector("h1");
  let frame = 0;

  const languageLink = document.querySelector(".language-switch a:not([aria-current])");
  const otherLanguageUrl = new URL(languageLink.href);
  function keepSection() {
    otherLanguageUrl.hash = window.location.hash;
    languageLink.href = otherLanguageUrl.href;
  }

  keepSection();
  window.addEventListener("hashchange", keepSection);

  function draw() {
    frame = 0;
    const at = Math.max(0, window.scrollY);
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    const progress = distance > 0 ? Math.min(at / distance, 1) : 0;
    progressBar.style.transform = "scaleX(" + progress + ")";
    if (name) {
      const drift = reduce.matches ? 0 : Math.min(at / Math.max(intro.offsetHeight * .65, 1), 1);
      name.style.setProperty("--name-drift", (drift * 32) + "px");
    }
  }

  function scheduleDraw() {
    if (!frame) frame = window.requestAnimationFrame(draw);
  }

  window.addEventListener("scroll", scheduleDraw, { passive: true });
  window.addEventListener("resize", scheduleDraw);
  window.addEventListener("load", scheduleDraw, { once: true });
  reduce.addEventListener("change", scheduleDraw);
  scheduleDraw();

  if (!("IntersectionObserver" in window)) return;
  let nameLeftView = false;
  if (name) {
    const nameObserver = new IntersectionObserver(([entry]) => {
      if (entry.intersectionRatio === 0) nameLeftView = true;
      if (entry.intersectionRatio < .7 || !nameLeftView) return;
      nameLeftView = false;
      intro.classList.add("reset-name");
      // Resolve the reset before starting the same CSS animation again.
      void name.offsetWidth;
      intro.classList.remove("reset-name");
    }, { threshold: [0, .7] });
    nameObserver.observe(name);
  }

  const titleTokens = new WeakMap();

  function prepareTitle(title, byLetter) {
    const label = title.textContent.trim();
    const visual = document.createElement("span");
    visual.setAttribute("aria-hidden", "true");
    const tokens = [];

    for (const word of label.split(/\s+/)) {
      if (tokens.length) visual.append(" ");
      const group = document.createElement("span");
      group.className = "type-word";
      for (const text of byLetter ? Array.from(word) : [word]) {
        const token = document.createElement("span");
        token.className = "type-token";
        token.textContent = text;
        group.append(token);
        tokens.push(token);
      }
      visual.append(group);
    }

    title.setAttribute("aria-label", label);
    title.replaceChildren(visual);
    titleTokens.set(title, tokens);
    return tokens;
  }

  function animateTitle(title) {
    const byLetter = title.dataset.motionTitle === "letters";
    const tokens = titleTokens.get(title) ?? prepareTitle(title, byLetter);
    const gentle = reduce.matches;
    const entry = gentle ? "translateY(.17em)"
      : byLetter ? "translateY(1.05em) rotate(7deg)" : "translateY(.88em)";

    tokens.forEach((token, index) => {
      token.getAnimations().forEach((animation) => animation.cancel());
      token.animate(
        [{ transform: entry, opacity: gentle ? .4 : .25 }, { transform: "none", opacity: 1 }],
        {
          delay: index * (byLetter ? 34 : 58),
          duration: 470,
          easing: "cubic-bezier(.22, 1, .36, 1)",
          fill: "backwards"
        }
      );
    });
  }

  if (!Element.prototype.animate) return;
  const observer = new IntersectionObserver((entries) => {
    for (const { target, isIntersecting, intersectionRatio } of entries) {
      if (isIntersecting && intersectionRatio >= .18) animateTitle(target);
    }
  }, { threshold: .18, rootMargin: "0px 0px -12% 0px" });
  document.querySelectorAll("[data-motion-title]").forEach((heading) => observer.observe(heading));
})();
