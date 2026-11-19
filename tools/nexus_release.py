"""The Nexus Mods half of a release (.github/workflows/nexus.yml). The same file in every mod with a Nexus page;
only the three constants below differ.

    python tools/nexus_release.py files              # the title and description of the file on Nexus
    python tools/nexus_release.py lines 1.2.0        # the changelog entries of a version, as Nexus gets them
    python tools/nexus_release.py changelog 1.2.0    # adds them to the Nexus page (needs NEXUS_API_KEY)

files: the upload action always sends a title (the file name, when none is given), so the workflow passes back
what the page has now and the title and description set by hand on Nexus stay. Read from the public v2 API, no
key needed. The file is the live (MAIN) one whose name starts with FILE_NAME; anything else than exactly one
stops the run before an upload. Prints lines for $GITHUB_OUTPUT.

changelog: the "## <version>" section of CHANGELOG.md, one Nexus changelog entry per change line, so the page
lists the changes one per line. The list marks, bold, code marks and link targets are dropped. Nexus only appends
entries, so lines the page already has for that version are skipped, and a second run adds nothing.
(The wording rules are checked before this, by the shared release workflow.)
"""

import json
import os
import re
import sys
import urllib.request
from pathlib import Path

MOD_ID = 48               # nexusmods.com/helldivers/mods/<MOD_ID>
FILE_NAME = "Upscaled Textures"         # the start of the file's name on the page, when the page holds other live files
ENTRY_PREFIX = "Upscaled Textures: "      # put before each changelog entry, when the page carries more than this mod

GAME_ID = 3552            # helldivers
GAME = "helldivers"
UNIQUE_ID = GAME_ID << 32 | MOD_ID   # the Unique Mod ID on the Files tab, Advanced
CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"
AGENT = "Obsidian-mod-release"


def request(url, data=None, key=None):
    headers = {"Content-Type": "application/json", "User-Agent": AGENT}
    if key:
        headers["apikey"] = key
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def files():
    q = {"query": f"{{ modFiles(modId: {MOD_ID}, gameId: {GAME_ID}) {{ name description category }} }}"}
    data = request("https://api.nexusmods.com/v2/graphql", q)
    if data.get("errors"):
        sys.exit(f"Nexus refused the file list: {data['errors']}")
    live = [f for f in data["data"]["modFiles"] if f["category"] == "MAIN" and f["name"].startswith(FILE_NAME)]
    if len(live) != 1:
        sys.exit(f"Expected one live Nexus file{' named ' + FILE_NAME if FILE_NAME else ''}, found {len(live)}: "
                 f"{[f['name'] for f in live]}. Nothing uploaded.")
    f = live[0]
    desc = (f["description"] or "").replace("\r", "")
    return f"name={f['name']}\ndescription<<DESC_END\n{desc}\nDESC_END"


def lines(version):
    """The change lines of the CHANGELOG.md section headed "## <version>", as plain text."""
    out, on = [], False
    for line in CHANGELOG.read_text(encoding="utf-8").replace("\r", "").split("\n"):
        if line.startswith("## "):
            on = line.strip() == f"## {version}"
            continue
        s = line.strip()
        if on and s:
            s = re.sub(r"^[-*]\s+", "", s)
            s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
            s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
            out.append(ENTRY_PREFIX + s.replace("`", ""))
    return out


def changelog(version):
    key = os.environ.get("NEXUS_API_KEY")
    if not key:
        sys.exit("NEXUS_API_KEY is not set.")
    entries = lines(version)
    if not entries:
        sys.exit(f"CHANGELOG.md has no section for {version}.")
    have = request(f"https://api.nexusmods.com/v1/games/{GAME}/mods/{MOD_ID}/changelogs.json", key=key)
    have = set(have.get(version, [])) if isinstance(have, dict) else set()
    for s in entries:
        if s in have:
            print(f"already on Nexus: {s}")
            continue
        request(f"https://api.nexusmods.com/v3/mods/{UNIQUE_ID}/changelogs", {"version": version, "changelog": s}, key)
        print(f"added: {s}")


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "files":
        print(files())
    elif len(sys.argv) == 3 and sys.argv[1] == "lines":
        print("\n".join(lines(sys.argv[2])))
    elif len(sys.argv) == 3 and sys.argv[1] == "changelog":
        changelog(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
