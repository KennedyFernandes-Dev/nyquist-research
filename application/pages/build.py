"""Prepare a static browser build of the Streamlit app for GitHub Pages."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = Path("application/streamlit/app.py")
PRESETS = Path("experiments/configurations/systems.json")
PACKAGE = Path("src/nyquist_research")
THEME = Path(".streamlit/config.toml")


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "main.js"):
        copyfile(Path(__file__).with_name(name), output / name)

    files = {
        path.as_posix(): (ROOT / path).read_text(encoding="utf-8")
        for path in (ENTRYPOINT, PRESETS, THEME)
    }
    for module in sorted((ROOT / PACKAGE).glob("*.py")):
        files[f"nyquist_research/{module.name}"] = module.read_text(encoding="utf-8")

    (output / "app.json").write_text(json.dumps(files, ensure_ascii=False), encoding="utf-8")
    (output / ".nojekyll").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
