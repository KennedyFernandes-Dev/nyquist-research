const app = document.querySelector("#app");
const status = document.querySelector("#status");

try {
  const [response, { mount }] = await Promise.all([
    fetch(new URL("./app.json", import.meta.url)),
    import("https://cdn.jsdelivr.net/npm/@stlite/browser@1.12.3/build/stlite.js"),
  ]);
  if (!response.ok) {
    throw new Error(`Arquivos do simulador indisponíveis (${response.status}).`);
  }
  const files = await response.json();

  // The root launcher puts the mounted package on Python's import path.
  files["app.py"] = "from application.streamlit.app import main\n\nmain()\n";
  mount({
    entrypoint: "app.py",
    files,
    requirements: ["numpy", "plotly==6.9.0"],
    streamlitConfig: {
      "browser.gatherUsageStats": false,
      "client.toolbarMode": "minimal",
    },
  }, app);
} catch (error) {
  status.textContent = "O simulador não carregou. Confira a conexão e atualize a página.";
  console.error(error);
}
