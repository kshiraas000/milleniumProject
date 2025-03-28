document.addEventListener("DOMContentLoaded", () => {
  const topbar = document.createElement("div");
  topbar.className = "topbar";
  topbar.innerHTML = `
    <div class="branding">Millennium</div>
    <div class="nav">
      <a href="/" class="nav-link">Stocks</a>
      <a href="/options" class="nav-link">Options</a>
    </div>
  `;
  document.body.prepend(topbar);
});
  