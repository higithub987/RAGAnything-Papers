(function () {
  const THEME_KEY = "ragTheme";
  const toggleBtn = document.getElementById("theme-toggle");
  if (!toggleBtn) return;

  function updateGlyph(theme) {
    toggleBtn.textContent = theme === "dark" ? "☀" : "🌙";
  }

  updateGlyph(document.documentElement.dataset.theme || "light");

  toggleBtn.addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem(THEME_KEY, next);
    updateGlyph(next);
  });
})();
