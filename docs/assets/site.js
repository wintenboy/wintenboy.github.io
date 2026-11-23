(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const progressBar = document.querySelector(".reading-progress");
  const portrait = document.querySelector(".portrait");
  const intro = document.querySelector(".intro");
  let frame = 0;

  function draw() {
    frame = 0;
    if (reduce.matches) {
      progressBar.style.removeProperty("transform");
      portrait.style.removeProperty("--photo-offset");
      portrait.style.removeProperty("--photo-angle");
      return;
    }

    const at = Math.max(0, window.scrollY);
    const distance = document.documentElement.scrollHeight - window.innerHeight;
    const progress = distance > 0 ? Math.min(at / distance, 1) : 0;
    const drift = Math.min(at / Math.max(intro.offsetHeight * .65, 1), 1);
    progressBar.style.transform = "scaleX(" + progress + ")";
    portrait.style.setProperty("--photo-offset", (drift * 40) + "px");
    portrait.style.setProperty("--photo-angle", (-4 + drift * 6) + "deg");
  }

  function scheduleDraw() {
    if (!frame) frame = window.requestAnimationFrame(draw);
  }

  window.addEventListener("scroll", scheduleDraw, { passive: true });
  window.addEventListener("resize", scheduleDraw);
  window.addEventListener("load", scheduleDraw, { once: true });
  reduce.addEventListener("change", scheduleDraw);
  scheduleDraw();

  if (!("IntersectionObserver" in window) || !Element.prototype.animate) return;

  const moments = [...document.querySelectorAll(".project-visual, .content-section > h2")];
  const observer = new IntersectionObserver((entries) => {
    if (reduce.matches) return;
    for (const { target, isIntersecting } of entries) {
      if (!isIntersecting) continue;
      observer.unobserve(target);

      target.animate(
        [{ transform: "translateY(14px)", opacity: .72 }, { transform: "none", opacity: 1 }],
        { duration: 440, easing: "cubic-bezier(.22, 1, .36, 1)" }
      );
    }
  }, { threshold: .18 });

  if (!reduce.matches) moments.forEach((element) => observer.observe(element));

  reduce.addEventListener("change", ({ matches }) => {
    if (!matches) return;
    observer.disconnect();
    moments.forEach((element) => element.getAnimations().forEach((animation) => animation.cancel()));
  });
})();
