(() => {
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const progressBar = document.querySelector(".reading-progress");
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
  }

  function scheduleDraw() {
    if (!frame) frame = window.requestAnimationFrame(draw);
  }

  window.addEventListener("scroll", scheduleDraw, { passive: true });
  window.addEventListener("resize", scheduleDraw);
  window.addEventListener("load", scheduleDraw, { once: true });
  scheduleDraw();

  if (!("IntersectionObserver" in window)) return;

  const observer = new IntersectionObserver((entries) => {
    for (const { target, isIntersecting, intersectionRatio } of entries) {
      if (!isIntersecting || intersectionRatio < .4) continue;
      observer.unobserve(target);
      if (!reduce.matches && target.getBoundingClientRect().top >= 0) {
        target.classList.add("section-title-entered");
      }
    }
  }, { threshold: .4, rootMargin: "0px 0px -8% 0px" });
  document.querySelectorAll(".home-index h2").forEach((heading) => observer.observe(heading));
})();
