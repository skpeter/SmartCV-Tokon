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

payload = {
    "state": None,
    "round": 0,
    "players": [
        {
            "name": None,
            "character": None,
            "team": [None] * 4,
            "games": 0,
            "rounds": 0,
        },
        {
            "name": None,
            "character": None,
            "team": [None] * 4,
            "games": 0,
            "rounds": 0,
        },
    ],
}

# First-to-3. Versus screen = new set. Rematch skips versus and keeps games.
_expect_round_start = False
_round_start_lock_until = 0.0
_ko_lock_until = 0.0
_results_latched = False
_game_awarded = False

# Pause-menu orange P1 sits top-left. Versus probes stay at y=930 so they
# do not collide. Do not move them upward.
# Interior of the P1/P2 blocks, not the label edge. Edge at x=1820
# reads (19,175,252) which blows a 0.07 channel budget.
VS_P1 = (70, 930, (255, 105, 0))
VS_P2 = (1750, 930, (0, 165, 255))
VS_DEV = 0.08

# Light-blue top strip + cream bottom band. Both required: cream-only
# loading transitions otherwise match the bottom probes.
RES_TOP = ((100, 50), (1820, 50))
RES_BOTTOM = ((100, 950), (1820, 950))
RES_TOP_COLOR = (82, 187, 254)
RES_BOTTOM_COLOR = (233, 223, 212)
RES_DEV = 0.05

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
RES_P1_DOTS = ((679, 882), (641, 882), (603, 882))
RES_P2_DOTS = ((1241, 882), (1279, 882), (1317, 882))

# BRACE YOURSELF olive text. Rematch has no versus screen.
BRACE_POINTS = ((457, 407), (854, 404), (1002, 400), (1246, 394), (1490, 392))
BRACE_COLOR = (68, 70, 60)
BRACE_DEV = 0.08

# Leader nameplates. P2 is right-aligned; rect anchored at the right edge.
P1_NAME_RECT = (105, 18, 390, 44)
P2_NAME_RECT = (1420, 18, 415, 44)

# Character select: no footage yet. Fill these when CSS is captured.
# CSS_POINT_A = (x, y)
# CSS_COLOR_A = (r, g, b)
# CSS_POINT_B = (x, y)
# CSS_COLOR_B = (r, g, b)
# CSS_DEV = 0.15


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
    return (mean[0] - mean[2]) > 50 and (mean[1] - mean[2]) > 25 and mean[0] > 100


def _count_stars(img, points, scale_x, scale_y):
    return sum(1 for x, y in points if _is_star(img, x, y, scale_x, scale_y))


def _reset_set(payload):
    global _results_latched, _game_awarded, _expect_round_start
    payload["round"] = 0
    for player in payload["players"]:
        player["games"] = 0
        player["rounds"] = 0
        player["character"] = None
        player["team"] = [None] * 4
    _results_latched = False
    _game_awarded = False
    _expect_round_start = True


def _reset_game(payload):
    global _results_latched, _game_awarded, _expect_round_start
    payload["round"] = 0
    for player in payload["players"]:
        player["rounds"] = 0
        player["character"] = None
        player["team"] = [None] * 4
    _results_latched = False
    _game_awarded = False
    _expect_round_start = True


def detect_character_select_screen(payload, img, scale_x, scale_y):
    # No CSS footage yet. Slot kept so it can be filled without reshaping
    # the state machine. Probe constants are commented at module top.
    return


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
    if not ocr_enabled:
        return
    if payload["players"][0]["character"] and payload["players"][1]["character"]:
        return
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
    detect_rounds(payload, img, scale_x, scale_y)
    core.print_with_time(f"Round {payload['round']} starting")
    _set_state(payload, "in_game")


def detect_rounds(payload, img, scale_x, scale_y):
    if payload["state"] != "in_game":
        return
    p1 = _count_stars(img, HUD_P1_DOTS, scale_x, scale_y)
    p2 = _count_stars(img, HUD_P2_DOTS, scale_x, scale_y)
    if p1 == 0 and p2 == 0 and (
        payload["players"][0]["rounds"] or payload["players"][1]["rounds"]
    ):
        # HUD hidden (K.O. / super cinematic). Keep last count.
        return
    if p1 < payload["players"][0]["rounds"] and p2 < payload["players"][1]["rounds"]:
        return
    if p1 >= payload["players"][0]["rounds"]:
        payload["players"][0]["rounds"] = p1
    if p2 >= payload["players"][1]["rounds"]:
        payload["players"][1]["rounds"] = p2
    if _debug():
        print("HUD dots:", p1, p2)


def detect_ko(payload, img, scale_x, scale_y):
    global _ko_lock_until, _expect_round_start
    if _now() < _ko_lock_until:
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
    core.print_with_time("K.O.")


def detect_results(payload, img, scale_x, scale_y):
    global _results_latched, _game_awarded, _expect_round_start
    top_ok = all(
        core.is_within_deviation(
            _px(img, x, y, scale_x, scale_y), RES_TOP_COLOR, RES_DEV
        )
        for x, y in RES_TOP
    )
    bot_ok = all(
        core.is_within_deviation(
            _px(img, x, y, scale_x, scale_y), RES_BOTTOM_COLOR, RES_DEV
        )
        for x, y in RES_BOTTOM
    )
    if not (top_ok and bot_ok):
        return
    if not _results_latched:
        r1 = _count_stars(img, RES_P1_DOTS, scale_x, scale_y)
        r2 = _count_stars(img, RES_P2_DOTS, scale_x, scale_y)
        if _debug():
            print("Results dots:", r1, r2)
        # Latch the first reading. ORDER SELECT later slides over the dots.
        if r1 or r2:
            payload["players"][0]["rounds"] = max(payload["players"][0]["rounds"], r1)
            payload["players"][1]["rounds"] = max(payload["players"][1]["rounds"], r2)
        _results_latched = True
    if not _game_awarded:
        r1 = payload["players"][0]["rounds"]
        r2 = payload["players"][1]["rounds"]
        winner = None
        if r1 >= 3 or r2 >= 3:
            winner = 0 if r1 >= r2 else 1
        elif r1 != r2:
            winner = 0 if r1 > r2 else 1
        if winner is not None:
            payload["players"][winner]["games"] += 1
            name = payload["players"][winner]["character"] or f"Player {winner + 1}"
            core.print_with_time(
                f"{name} wins game "
                f"({payload['players'][0]['rounds']}-"
                f"{payload['players'][1]['rounds']})  "
                f"set {payload['players'][0]['games']}-"
                f"{payload['players'][1]['games']}"
            )
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
        detect_round_start,
        detect_rounds,
        detect_ko,
        detect_results,
    ],
    "game_end": [
        detect_results,
        detect_character_select_screen,
        detect_versus_screen,
        detect_match_starting,
        detect_round_start,
    ],
}
