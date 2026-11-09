"""Zips Upscaled Textures (obsidian.json, LICENSE and res/) into build/ as <id>-<version>.zip, the name a release
uses, replacing earlier zips there."""
import json
import zipfile
from pathlib import Path

mod = Path(__file__).resolve().parents[1]
manifest = json.loads((mod / "obsidian.json").read_text(encoding="utf-8"))
build = mod / "build"
build.mkdir(exist_ok=True)
for old in build.glob("*.zip"):
    old.unlink()
out = build / f"{manifest['id']}-{manifest['version']}.zip"
files = [mod / "obsidian.json", mod / "LICENSE"] + sorted((mod / "res").glob("*.texture"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for f in files:
        z.write(f, f"{manifest['id']}/{f.relative_to(mod).as_posix()}")
print(f"{out}: {len(files)} files, {out.stat().st_size / 1e6:.0f} MB")
