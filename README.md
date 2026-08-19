# SmartCV-Tokon

![SmartCV-Tokon](assets/demo.mp4)

SmartCV-Tokon reads **MARVEL Tōkon: Fighting Souls** match state from the screen. No game mods. Polls one frame every 0.5s and prefers single-pixel / small-region color checks over template matching or OCR.

OCR runs **once per game**, on the two HUD leader names, then latches. Everything else is pixel probes.

Shared engine lives in `core/` ([smartcv-core](https://github.com/skpeter/smartcv-core)). Game logic is `routines.py`.

## Requirements

- [OBS (optional if streaming)](https://obsproject.com/download)
- Game UI must be **English**. Do not use UI mods.
- Capture assumed **16:9**, coordinates authored at **1920×1080**.

## Step 1: Installation

### CPU / release

Download `release.zip` from [Releases](https://github.com/skpeter/SmartCV-Tokon/releases). Skip to step 2.

### GPU / source

- Download **source.zip** from the [latest release](https://github.com/skpeter/SmartCV-Tokon/releases/latest/download/source.zip). Do not use GitHub's auto-generated "Source code" zip — it is missing `core`.
- Install [Python 3.12](https://www.python.org/downloads/).
- `pip install -r core/requirements.txt`
- Install PyTorch from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (Stable, your OS, Pip, Python, your CUDA).

## Step 2: Setup

Copy `core/config.ini.example` to `config.ini` in the repo root.

### Game capture

Set `executable_title` to a substring of the window title, default:

```
executable_title = MARVEL Tōkon: Fighting Souls
```

If the window is not found, check the title in Task Manager and put a unique substring here.

### OBS capture

Set `capture_mode = obs`, fill `[obs]` (`source_title`, host, port, password, width, height). Enable OBS Websocket.

## Step 3: Usage

- Source: `smartcv.bat` / `smartcv.sh` / `smartcv.command` (git clone: same files under `core/`).
- Release: `smartcv.exe`.

WebSocket JSON on port **6565** (configurable). UDP discovery on **6500**. Client integration: [S.M.A.R.T.](https://skpeter.github.io/smart-user-guide). Schema: `example-json.json`.

## Payload

| Field | Meaning |
| --- | --- |
| `state` | `character_select` (stub), `loading`, `in_game`, `game_end` |
| `round` | `p1.rounds + p2.rounds + 1` (never parsed from "ROUND N" / "FINAL ROUND") |
| `players[n].character` | Current leader. OCR once, then latched |
| `players[n].team` | 4 slots, always null until character-select footage exists |
| `players[n].rounds` | Yellow star dots, first to 3 |
| `players[n].games` | Counted internally. HUD has no set score; results `SCORE` goes past 9 |

Versus screen = new **set** (games + rounds reset). Rematch skips versus → `BRACE YOURSELF` = new **game** (rounds reset, games kept).

## Pixel probes (1080p)

Retune these after a UI patch. All `x,y` are base 1920×1080; runtime multiplies by `scale_x` / `scale_y`.

| Detector | Probe | Color | Dev |
| --- | --- | --- | --- |
| Versus | `(70, 930)` | `(255, 105, 0)` | 0.08 |
| Versus | `(1750, 930)` | `(0, 165, 255)` | 0.08 |
| Results top | `(100, 50)`, `(1820, 50)` | `(82, 187, 254)` | 0.05 |
| Results bottom | `(100, 950)`, `(1820, 950)` | `(233, 223, 212)` | 0.05 |
| Round start P1 HP box | `x 240–262, y 80–96` mean | `(252, 201, 0)` | 0.10 |
| Round start P2 HP box | `x 1670–1692, y 80–96` mean | `(85, 254, 255)` | 0.12 |
| K.O. glyph | `(1099, 467)`, `(720, 481)`, `(811, 481)`, `(1111, 508)`, `(988, 520)` | `(236, 222, 212)` | 0.055 |
| K.O. negative guard | `(200, 540)`, `(1740, 540)` must **not** be cream | `(233, 223, 212)` | 0.06 |
| HUD dots P1 (inner→outer) | `(824, 47)`, `(785, 48)`, `(747, 48)` | chroma `R-B>50, G-B>25, R>100` on 15×15 | — |
| HUD dots P2 | `(1097, 48)`, `(1134, 48)`, `(1173, 48)` | same | — |
| Results dots P1 | `(679, 882)`, `(641, 882)`, `(603, 882)` | same | — |
| Results dots P2 | `(1241, 882)`, `(1279, 882)`, `(1317, 882)` | same | — |
| BRACE YOURSELF | `(457, 407)`, `(854, 404)`, `(1002, 400)`, `(1246, 394)`, `(1490, 392)` | `(68, 70, 60)` | 0.08 |
| Leader OCR (once) | P1 `(105, 18, 390, 44)`, P2 `(1420, 18, 415, 44)` | EasyOCR + roster fuzzy match | — |
| Character select P1 | `(52, 82)`, `(50, 100)` | `(253, 114, 0)` | 0.08 |
| Character select P2 | `(1830, 91)`, `(1850, 110)` | `(0, 180, 252)` | 0.08 |
| Character select paper | `(40, 40)`, `(1880, 40)` | `(251, 242, 229)` | 0.05 |

Do not move versus probes up: pause menu draws orange `P1` at top-left.

## VOD replay (dev)

```
python dev/validate_vod.py path/to.mp4 --start 50 --end 380 --step 0.5
```

Walks the file at the same 0.5s interval SmartCV uses. `--ocr` enables the one-shot name read (slow).

## Check out also

- [SmartCV-SF6](https://github.com/skpeter/SmartCV-SF6)
- [SmartCV-SSBU](https://github.com/skpeter/SmartCV-SSBU)

## Contact

[Discord](https://discord.gg/zecMKvF8b5)
