(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const progressBar = document.querySelector(".reading-progress");
  const intro = document.querySelector(".intro");
  const name = intro.querySelector("h1");
  const motionButton = document.querySelector(".motion-toggle");
  const motionKey = "portfolio-motion";
  let motionPreference;
  let motionEnabled = false;
  let frame = 0;

  try {
    motionPreference = localStorage.getItem(motionKey);
  } catch {
    // The switch still works for this page when storage is unavailable.
  }

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
    if (!motionEnabled) {
      progressBar.style.removeProperty("transform");
      name.style.removeProperty("--name-drift");
      return;
    }

    const at = Math.max(0, window.scrollY);
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    const progress = distance > 0 ? Math.min(at / distance, 1) : 0;
    const drift = Math.min(at / Math.max(intro.offsetHeight * .65, 1), 1);
    progressBar.style.transform = "scaleX(" + progress + ")";
    name.style.setProperty("--name-drift", (drift * 32) + "px");
  }

  function scheduleDraw() {
    if (!frame) frame = window.requestAnimationFrame(draw);
  }

  window.addEventListener("scroll", scheduleDraw, { passive: true });
  window.addEventListener("resize", scheduleDraw);
  window.addEventListener("load", scheduleDraw, { once: true });

  const canObserve = "IntersectionObserver" in window;
  let nameLeftView = false;
  if (canObserve) {
    const nameObserver = new IntersectionObserver(([entry]) => {
      if (entry.intersectionRatio === 0) nameLeftView = true;
      if (entry.intersectionRatio < .7 || !nameLeftView) return;
      nameLeftView = false;
      if (!motionEnabled) return;

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
    const byLetter = title.tagName === "H2";
    const tokens = titleTokens.get(title) ?? prepareTitle(title, byLetter);
    const entry = byLetter ? "translateY(1.05em) rotate(7deg)" : "translateY(.88em)";

    tokens.forEach((token, index) => {
      token.getAnimations().forEach((animation) => animation.cancel());
      token.animate(
        [{ transform: entry, opacity: .25 }, { transform: "none", opacity: 1 }],
        {
          delay: index * (byLetter ? 34 : 58),
          duration: 470,
          easing: "cubic-bezier(.22, 1, .36, 1)",
          fill: "backwards"
        }
      );
    });
  }

  const headings = [...document.querySelectorAll(".content-section > h2, .project-copy > h3")];
  const observer = canObserve && Element.prototype.animate
    ? new IntersectionObserver((entries) => {
      if (!motionEnabled) return;
      for (const { target, isIntersecting, intersectionRatio } of entries) {
        if (isIntersecting && intersectionRatio >= .18) animateTitle(target);
      }
    }, { threshold: .18, rootMargin: "0px 0px -12% 0px" })
    : null;

  function updateMotion() {
    motionEnabled = motionPreference === "on" || (motionPreference !== "off" && !reduce.matches);
    document.documentElement.classList.toggle("motion-enabled", motionEnabled);
    motionButton.setAttribute("aria-pressed", String(motionEnabled));
    scheduleDraw();

    if (!observer) return;
    if (motionEnabled) {
      headings.forEach((heading) => observer.observe(heading));
    } else {
      observer.disconnect();
      headings.forEach((heading) => {
        heading.getAnimations({ subtree: true }).forEach((animation) => animation.cancel());
      });
    }
  }

  motionButton.addEventListener("click", () => {
    motionPreference = motionEnabled ? "off" : "on";
    try {
      localStorage.setItem(motionKey, motionPreference);
    } catch {
      // A browser can block storage on local files or in private mode.
    }
    updateMotion();
  });
  reduce.addEventListener("change", updateMotion);
  motionButton.hidden = false;
  updateMotion();
})();
