# Upscaled Textures

The textures of HELLDIVERS (2015) at twice the size, upscaled by an AI model (Real-ESRGAN) and compressed back to the game's own formats, for the [Obsidian](https://github.com/Alomare/ObsidianModManager) mod manager. Download `alomare.upscaledtextures-<version>.zip` from the Releases page and import the zip into Obsidian. The mod works with or without [Enhanced Visuals](https://github.com/Alomare/HD1-EnhancedVisuals).

The upscaler adds detail that the original art didn't have, so some surfaces look sharper and a few look rough.

Before 1.0.0 this mod was called Enhanced Textures. Obsidian treats the new name as a different mod, so remove Enhanced Textures before importing this one.

The mod has two on/off options, one per kind of texture:

| Option | Default | What it covers |
|---|---|---|
| World textures | On | The colour textures of levels, characters and objects |
| Normal maps | Off | The surface detail of levels, characters and objects. Sharper, but sloppier than the originals |

Interface textures (uncompressed ones and odd sizes: icons, menus, loading screens) were upscaled in the 0.2.0 test and dropped: the gain wasn't worth it.

## Folder

| Path | What it is |
|---|---|
| `obsidian.json`, `res/` | The mod, written by `tools/textures.py pack`. Generated and about 600 MB, so they ship only in the release zip |
| `build/` | The zip to import into Obsidian, written by `tools/zip_textures.py` |
| `tools/textures.py` | The pipeline: select, extract, upscale, pack, check, status |
| `tools/texture_survey.py` | Every texture of the game: size, format, copies. Writes `internal/research/textures.csv` |
| `tools/stingray.py` | Bundle reading, a copy of the one in Enhanced Visuals |
| `tools/textures_exclude.txt` | Textures left out, by hand or because they failed the check |
| `internal/run_textures_all.ps1` | The whole run, unattended: every step, then the zip. Safe to run again: each step skips what is done |
| `tools/zip_textures.py` | Zips the mod into `build/` |
| `internal/work/` | Extracted, padded and upscaled images (about 8 GB) |

Needs Windows, Python with numpy and Pillow, and `realesrgan-ncnn-vulkan` (expected at `X:\Tools\realesrgan`). The game is read from `HD1_CONTENTS`, or the default Steam folder.

## What the game has

- 13,072 texture entries, but only 2,045 distinct textures: the same file sits in several level bundles (median 2 copies, up to 60). Distinct data is 0.25 GB, all entries 1.93 GB. Nothing is streamed: every texture is whole inside the bundles, so a replacement is a plain resource.
- Small for a modern screen, as a top-down game's are: most are 512 px or less on the longest side, 64 are 1024, three are 2048 or 4096.
- Formats: DXT5 (906), DXT1 (669), ATI2/BC5 normal maps (249), uncompressed 32-bit (217), DXT3 (3). The resource is a plain DDS with every mip, then the 4-byte category tail.
- Every copy has to be replaced, or the old texture shows wherever a copy was left out.

## How the pipeline works

- Colour channels that move together go to the upscaler as one RGB image. Packed masks (gloss, metal, occlusion) go one channel at a time, so one map can't bleed into another. Alpha goes alone.
- Normal maps (ATI2) upscale their two channels apart; the shader rebuilds the third.
- Tiling textures are padded by wrapping the opposite edge around, so the seams still meet.
- The upscaler works at 4x; Lanczos brings the result down to 2x.
- Mips are rebuilt (or the top level alone, where the original has none) and compressed by Pillow. The header is the original's with only the size and mip count changed.
- Every resize works channel by channel: Pillow resizes RGBA with colour weighted by alpha, which darkens colour under low alpha.
- Cube maps and volume textures are skipped: Pillow reads only their first face or slice.
- `check` compares each packed texture to the original (format, mip count, PSNR) and, with `--exclude`, adds the failures to the exclude list.

Proven in game on 2026-11-05 with a probe build that striped each replaced texture magenta: the desert ground drew the stripes from the double-size texture, with no crash.

## Last full run (2026-11-07)

0.2.0, with interface textures: 2,027 selected, 2,007 packed, 359 MB zipped (921 MB unpacked); the upscale took 2 hours 18 minutes. Left out: 13 cube maps and volumes, 6 that failed the check, and one 4096×4096 texture whose 4x intermediate (16384×16384) is over Pillow's image size limit. 0.3.0 repacked the same work without the interface textures.
