import configparser
import time
import tokon
import numpy as np
import core.core as core
from core.matching import findBestMatch

client_name = "smartcv-tokon"
config = configparser.ConfigParser()
config.read("config.ini")
previous_states = [None]

# Wall-clock source. VOD harness replaces this with video timestamp.
_now = time.time
ocr_enabled = True
CHAR_OCR_MAX_TRIES = 5
leader_ocr_attempts = 0

payload = {
    "state": None,
    "round": 0,
    "players": [
        {
            "name": None,
            "character": None,
            "team": [None] * 4,
            "rounds": 0,
        },
        {
            "name": None,
            "character": None,
            "team": [None] * 4,
            "rounds": 0,
        },
    ],
}

# First-to-3. Versus screen = new set. Rematch skips versus (set score lives in S.M.A.R.T.).
_expect_round_start = False
_round_start_lock_until = 0.0
_ko_lock_until = 0.0
_score_pending = False
_game_awarded = False

# Pause-menu orange P1 sits top-left. Versus probes stay at y=930 so they
# do not collide. Do not move them upward.
# Interior of the P1/P2 blocks, not the label edge. Edge at x=1820
# reads (19,175,252) which blows a 0.07 channel budget.
VS_P1 = (70, 930, (255, 105, 0))
VS_P2 = (1750, 930, (0, 165, 255))
VS_DEV = 0.08

# Outer HP tips. Bars deplete inward, so these are only colored at 100%.
# Box mean: bar sheen animates single pixels.
HP_P1 = (240, 80, 262, 96, (252, 201, 0), 0.10)
HP_P2 = (1670, 80, 1692, 96, (85, 254, 255), 0.12)

# K.O. glyph is fixed across stages. Cream loading screens also hit the
# five glyph points; reject if the side guards are cream too.
KO_POINTS = ((1099, 467), (720, 481), (811, 481), (1111, 508), (988, 520))
KO_GUARDS = ((200, 540), (1740, 540))
KO_COLOR = (236, 222, 212)
KO_DEV = 0.055
KO_GUARD_COLOR = (233, 223, 212)
KO_GUARD_DEV = 0.06

# 3 dots/side, 38px spacing, fill inner-to-outer. Over live sky, so chroma
# on a 15x15 box mean instead of exact color. HUD hidden during supers/K.O.
# reads 0-0; never decrease a stored count.
HUD_P1_DOTS = ((824, 47), (785, 48), (747, 48))
HUD_P2_DOTS = ((1097, 48), (1134, 48), (1173, 48))

# BRACE YOURSELF olive text. Rematch has no versus screen.
BRACE_POINTS = ((457, 407), (854, 404), (1002, 400), (1246, 394), (1490, 392))
BRACE_COLOR = (68, 70, 60)
BRACE_DEV = 0.08

# Leader nameplates. P2 is right-aligned; rect anchored at the right edge.
P1_NAME_RECT = (105, 18, 390, 44)
P2_NAME_RECT = (1420, 18, 415, 44)

# Character select: P1/P2 letter interiors + paper-beige top corners.
# Versus uses the same orange/blue at y=930; pause has P1 orange only.
# Beige (251,242,229) is tighter than results cream (233,223,212).
CSS_P1 = ((52, 82), (50, 100))
CSS_P2 = ((1830, 91), (1850, 110))
CSS_BG = ((40, 40), (1880, 40))
CSS_ORANGE = (253, 114, 0)
CSS_BLUE = (0, 180, 252)
CSS_BEIGE = (251, 242, 229)
CSS_ORANGE_DEV = 0.08
CSS_BLUE_DEV = 0.08
CSS_BEIGE_DEV = 0.05


def _debug():
    return config.getboolean("settings", "debug_mode", fallback=False)


def _set_state(payload, state):
    payload["state"] = state
    if previous_states[-1] != state:
        previous_states.append(state)


def _as_rgb(img):
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3]
    return arr


def _px(img, x, y, scale_x, scale_y):
    arr = _as_rgb(img)
    sx = min(max(int(x * scale_x), 0), arr.shape[1] - 1)
    sy = min(max(int(y * scale_y), 0), arr.shape[0] - 1)
    return tuple(int(v) for v in arr[sy, sx])


def _region_mean(img, x0, y0, x1, y1, scale_x, scale_y):
    arr = _as_rgb(img)
    xa, ya = int(x0 * scale_x), int(y0 * scale_y)
    xb, yb = int(x1 * scale_x), int(y1 * scale_y)
    xa, xb = sorted((max(xa, 0), min(xb, arr.shape[1])))
    ya, yb = sorted((max(ya, 0), min(yb, arr.shape[0])))
    if xb <= xa or yb <= ya:
        return (0, 0, 0)
    box = arr[ya:yb, xa:xb].reshape(-1, 3)
    return tuple(float(v) for v in box.mean(0))


def _is_star(img, x, y, scale_x, scale_y, half=7):
    arr = _as_rgb(img)
    cx, cy = int(x * scale_x), int(y * scale_y)
    hx, hy = max(int(half * scale_x), 1), max(int(half * scale_y), 1)
    xa, xb = max(cx - hx, 0), min(cx + hx + 1, arr.shape[1])
    ya, yb = max(cy - hy, 0), min(cy + hy + 1, arr.shape[0])
    if xb <= xa or yb <= ya:
        return False
    mean = arr[ya:yb, xa:xb].reshape(-1, 3).mean(0)
    # Gold ring + black star hole averages R~130 in-match. Chroma (not R
    # floor) separates coins from white empty slots (those have dR < 0).
    return (mean[0] - mean[2]) > 70 and (mean[1] - mean[2]) > 40 and mean[0] > 110


def _count_stars(img, points, scale_x, scale_y):
    n = 0
    for x, y in points:
        if not _is_star(img, x, y, scale_x, scale_y):
            break
        n += 1
    return n


def _reset_set(payload):
    global _game_awarded, _expect_round_start, _score_pending, leader_ocr_attempts
    payload["round"] = 0
    for player in payload["players"]:
        player["rounds"] = 0
        player["character"] = None
        player["team"] = [None] * 4
    _game_awarded = False
    _expect_round_start = True
    _score_pending = False
    leader_ocr_attempts = 0


def _reset_game(payload):
    global _game_awarded, _expect_round_start, _score_pending, leader_ocr_attempts
    payload["round"] = 0
    for player in payload["players"]:
        player["rounds"] = 0
        player["character"] = None
        player["team"] = [None] * 4
    _game_awarded = False
    _expect_round_start = True
    _score_pending = False
    leader_ocr_attempts = 0


def detect_character_select_screen(payload, img, scale_x, scale_y):
    if not all(
        core.is_within_deviation(_px(img, x, y, scale_x, scale_y), CSS_ORANGE, CSS_ORANGE_DEV)
        for x, y in CSS_P1
    ):
        return
    if not all(
        core.is_within_deviation(_px(img, x, y, scale_x, scale_y), CSS_BLUE, CSS_BLUE_DEV)
        for x, y in CSS_P2
    ):
        return
    if not all(
        core.is_within_deviation(_px(img, x, y, scale_x, scale_y), CSS_BEIGE, CSS_BEIGE_DEV)
        for x, y in CSS_BG
    ):
        return
    if _debug():
        print("Character select pixels matched")
    if payload["state"] == "character_select":
        return
    core.print_with_time("- Character select screen detected")
    if payload["state"] in (None, "in_game", "game_end"):
        _reset_set(payload)
    _set_state(payload, "character_select")


def detect_versus_screen(payload, img, scale_x, scale_y):
    p1 = _px(img, VS_P1[0], VS_P1[1], scale_x, scale_y)
    p2 = _px(img, VS_P2[0], VS_P2[1], scale_x, scale_y)
    if _debug():
        print("Versus screen pixels:", p1, p2)
    if not (
        core.is_within_deviation(p1, VS_P1[2], VS_DEV)
        and core.is_within_deviation(p2, VS_P2[2], VS_DEV)
    ):
        return
    if payload["state"] != "loading":
        core.print_with_time("- Versus screen detected (new set)")
        _reset_set(payload)
        _set_state(payload, "loading")


def detect_match_starting(payload, img, scale_x, scale_y):
    global _expect_round_start
    hits = 0
    for x, y in BRACE_POINTS:
        pixel = _px(img, x, y, scale_x, scale_y)
        if core.is_within_deviation(pixel, BRACE_COLOR, BRACE_DEV):
            hits += 1
    if _debug():
        print("BRACE olive hits:", hits)
    if hits < len(BRACE_POINTS):
        return
    if payload["state"] in (None, "game_end", "character_select"):
        core.print_with_time("- BRACE YOURSELF (match starting)")
        if payload["state"] == "game_end":
            _reset_game(payload)
        else:
            _expect_round_start = True
        _set_state(payload, "loading")


def detect_leaders(payload, img, scale_x, scale_y):
    global leader_ocr_attempts
    if not ocr_enabled:
        return
    if payload["players"][0]["character"] and payload["players"][1]["character"]:
        return
    if leader_ocr_attempts >= CHAR_OCR_MAX_TRIES:
        return
    leader_ocr_attempts += 1
    texts = []
    for rect in (P1_NAME_RECT, P2_NAME_RECT):
        x, y, w, h = (
            int(rect[0] * scale_x),
            int(rect[1] * scale_y),
            int(rect[2] * scale_x),
            int(rect[3] * scale_y),
        )
        result = core.read_text(img, (x, y, w, h), contrast=2)
        texts.append(" ".join(result) if result else "")
    for i, raw in enumerate(texts):
        if payload["players"][i]["character"] or not raw:
            continue
        match, _ = findBestMatch(raw, tokon.characters)
        if match:
            payload["players"][i]["character"] = match
            core.print_with_time(
                f"{payload['players'][i]['name'] or f'Player {i + 1}'} as:", match
            )


def detect_round_start(payload, img, scale_x, scale_y):
    global _round_start_lock_until, _expect_round_start, _ko_lock_until
    if _now() < _round_start_lock_until:
        return
    if payload["state"] == "in_game" and not _expect_round_start:
        return
    if payload["state"] == "in_game" and (
        payload["players"][0]["rounds"] >= 3 or payload["players"][1]["rounds"] >= 3
    ):
        return
    p1 = _region_mean(img, HP_P1[0], HP_P1[1], HP_P1[2], HP_P1[3], scale_x, scale_y)
    p2 = _region_mean(img, HP_P2[0], HP_P2[1], HP_P2[2], HP_P2[3], scale_x, scale_y)
    if _debug():
        print("HP box means:", p1, p2)
    if not (
        core.is_within_deviation(p1, HP_P1[4], HP_P1[5])
        and core.is_within_deviation(p2, HP_P2[4], HP_P2[5])
    ):
        return
    _round_start_lock_until = _now() + 10
    _ko_lock_until = 0.0
    _expect_round_start = False
    payload["round"] = (
        payload["players"][0]["rounds"] + payload["players"][1]["rounds"] + 1
    )
    detect_leaders(payload, img, scale_x, scale_y)
    core.print_with_time(f"Round {payload['round']} starting")
    _set_state(payload, "in_game")


def detect_rounds(payload, img, scale_x, scale_y):
    global _score_pending
    if payload["state"] != "in_game":
        return
    p1 = _count_stars(img, HUD_P1_DOTS, scale_x, scale_y)
    p2 = _count_stars(img, HUD_P2_DOTS, scale_x, scale_y)
    if _debug():
        print("HUD dots:", p1, p2)
    if not _score_pending:
        return
    s1 = payload["players"][0]["rounds"]
    s2 = payload["players"][1]["rounds"]
    if p1 == 0 and p2 == 0:
        # HUD hidden (K.O. / super cinematic). Keep waiting.
        return
    up1 = p1 > s1
    up2 = p2 > s2
    if up1 and up2:
        # Both sides jumped — sky / wipe artifact, not a single round win.
        return
    if not (up1 or up2):
        # Incomplete refill after wipe (e.g. 2-0 while stored 2-2).
        return
    payload["players"][0]["rounds"] = min(3, max(s1, p1))
    payload["players"][1]["rounds"] = min(3, max(s2, p2))
    _score_pending = False
    _maybe_award_game(payload)


def detect_ko(payload, img, scale_x, scale_y):
    global _ko_lock_until, _expect_round_start, _score_pending
    if _now() < _ko_lock_until:
        return
    if payload["players"][0]["rounds"] >= 3 or payload["players"][1]["rounds"] >= 3:
        return
    if _expect_round_start:
        return
    hits = all(
        core.is_within_deviation(_px(img, x, y, scale_x, scale_y), KO_COLOR, KO_DEV)
        for x, y in KO_POINTS
    )
    if not hits:
        return
    if any(
        core.is_within_deviation(
            _px(img, x, y, scale_x, scale_y), KO_GUARD_COLOR, KO_GUARD_DEV
        )
        for x, y in KO_GUARDS
    ):
        return
    _ko_lock_until = _now() + 3
    _expect_round_start = True
    _score_pending = True
    core.print_with_time("K.O.")


def _maybe_award_game(payload):
    global _game_awarded, _expect_round_start
    if _game_awarded:
        return
    r1 = payload["players"][0]["rounds"]
    r2 = payload["players"][1]["rounds"]
    if r1 < 3 and r2 < 3:
        return
    if r1 == r2:
        return
    winner = 0 if r1 > r2 else 1
    name = payload["players"][winner]["character"] or f"Player {winner + 1}"
    core.print_with_time(f"{name} wins game ({r1}-{r2})")
    _game_awarded = True
    _expect_round_start = True
    _set_state(payload, "game_end")


states_to_functions = {
    None: [
        detect_character_select_screen,
        detect_versus_screen,
        detect_match_starting,
    ],
    "character_select": [detect_versus_screen, detect_round_start, detect_match_starting],
    "loading": [detect_round_start],
    "in_game": [
        detect_character_select_screen,
        detect_versus_screen,
        detect_round_start,
        detect_leaders,
        detect_rounds,
        detect_ko,
    ],
    "game_end": [
        detect_character_select_screen,
        detect_versus_screen,
        detect_match_starting,
        detect_round_start,
    ],
}
