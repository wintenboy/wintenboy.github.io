(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const progressBar = document.querySelector(".reading-progress");
  const intro = document.querySelector(".intro");
  const name = intro.querySelector("h1");
  const replay = intro.querySelector(".motion-replay");
  let frame = 0;

  replay.hidden = false;
  replay.addEventListener("click", () => {
    intro.classList.add("reset-name");
    // Resolve the reset before asking the browser to start the same CSS animation.
    void name.offsetWidth;
    intro.classList.remove("reset-name");
    intro.classList.add("replay-name");
  });

  function draw() {
    frame = 0;
    if (reduce.matches) {
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
  reduce.addEventListener("change", () => {
    if (reduce.matches) intro.classList.remove("replay-name");
    scheduleDraw();
  });
  scheduleDraw();

  if (!("IntersectionObserver" in window) || !Element.prototype.animate) return;

  const moments = [...document.querySelectorAll(".project-visual, .content-section > h2")];
  const observer = new IntersectionObserver((entries) => {
    if (reduce.matches) return;
    for (const { target, isIntersecting } of entries) {
      if (!isIntersecting) continue;
      observer.unobserve(target);

      const heading = target.tagName === "H2";
      const keyframes = heading
        ? [
            { clipPath: "inset(-.2em 100% -.2em -.06em)", transform: "translateY(.24em)" },
            { clipPath: "inset(-.2em -.06em -.2em -.06em)", transform: "none" }
          ]
        : [{ transform: "translateY(14px)", opacity: .72 }, { transform: "none", opacity: 1 }];
      target.animate(keyframes, { duration: heading ? 580 : 440, easing: "cubic-bezier(.22, 1, .36, 1)" });
    }
  }, { threshold: .18 });

  if (!reduce.matches) moments.forEach((element) => observer.observe(element));

  reduce.addEventListener("change", ({ matches }) => {
    if (!matches) return;
    observer.disconnect();
    moments.forEach((element) => element.getAnimations().forEach((animation) => animation.cancel()));
  });
})();
