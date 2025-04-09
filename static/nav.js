document.addEventListener("DOMContentLoaded", () => {
  const topbar = document.createElement("div");
  topbar.className = "topbar";
  topbar.innerHTML = `
    <div class="branding">Millennium</div>
    <div class="nav">
      <a href="/dashboard.html" class="nav-link">Portfolio</a>
      <a href="/index.html" class="nav-link">Stocks</a>
      <a href="/options.html" class="nav-link">Options</a>
    </div>
  `;
  document.body.prepend(topbar);
});
  