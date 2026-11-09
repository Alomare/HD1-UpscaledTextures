# Rules for this repository

**Every release has a changelog.** Before a release, add a section for the version to `CHANGELOG.md`, headed `## <version>`, listing the new, changed or removed features. Each change is one list item, starting `- New:`, `- Changed:` or `- Removed:`, and describes the change briefly and directly. That section becomes the release notes.

**A release is made by hand.** `res/` and `obsidian.json` are generated from the game and aren't in the repository, so no workflow can pack them. Set the version in `tools/textures.py` and in `obsidian.json`, run `python tools/zip_textures.py`, push, then `gh release create v<version> build/alomare.upscaledtextures-<version>.zip --title "Upscaled Textures <version>" --notes-file <the changelog section>`. Obsidian picks the zip by the mod id in the name.
