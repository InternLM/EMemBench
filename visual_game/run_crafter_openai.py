#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, sys, io, base64, datetime
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gym
import crafter
from openai import OpenAI
# Note: we no longer import dump_crafter_map.run; see map export logic in main().

# ---------- actions ----------
ACTIONS = [
    "NOOP",              # 0
    "MOVE_LEFT",         # 1
    "MOVE_RIGHT",        # 2
    "MOVE_UP",           # 3
    "MOVE_DOWN",         # 4
    "DO",                # 5
    "SLEEP",             # 6
    "PLACE_STONE",       # 7
    "PLACE_TABLE",       # 8
    "PLACE_FURNACE",     # 9
    "PLACE_PLANT",       # 10
    "MAKE_WOOD_PICKAXE", # 11
    "MAKE_STONE_PICKAXE",# 12
    "MAKE_IRON_PICKAXE", # 13
    "MAKE_WOOD_SWORD",   # 14
    "MAKE_STONE_SWORD",  # 15
    "MAKE_IRON_SWORD",   # 16
]

BASE_SYSTEM = (
    "You are a vision game agent for the Crafter environment. "
    'You will see a single 64x64 RGB frame and a HUD(text) line with the last step\'s numeric stats. '
    'Choose EXACTLY ONE action and respond with STRICT JSON only:\n'
    '{"action_id": <0-16>, "action_name": "<string>", "reason": "<brief diagnosis & plan (no raw numbers)>"}\n'
    "Valid actions: " + ", ".join(f"{i}:{a}" for i, a in enumerate(ACTIONS)) + ". "
)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_eky")  # <-- set env var
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1")
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ---------- compatibility helpers ----------
def safe_reset(env, seed=None):
    # Always try the new API first
    try:
        out = env.reset(seed=seed)
        return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})
    except Exception:
        # Fall back to unwrapped if wrappers don’t pass through the seed
        try:
            base = getattr(env, "unwrapped", env)
            out = base.reset(seed=seed)
            return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})
        except Exception:
            # As a last resort, seed via legacy API, then reset with no seed
            try:
                if hasattr(env, "seed"):
                    env.seed(seed)
                elif hasattr(getattr(env, "unwrapped", env), "seed"):
                    env.unwrapped.seed(seed)
            except Exception:
                pass
            out = env.reset()
            return out if (isinstance(out, tuple) and len(out) == 2) else (out, {})

# ---------- helpers ----------
def parse_action_id(txt: str) -> int:
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1 or e == -1:
        return 0
    try:
        obj = json.loads(txt[s:e+1])
        aid = int(obj.get("action_id", 0))
        return aid if 0 <= aid < len(ACTIONS) else 0
    except Exception:
        return 0

def json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, (np.bool_,)):     return bool(obj)
    if isinstance(obj, (np.ndarray,)):   return obj.tolist()
    if isinstance(obj, dict):            return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):   return [json_safe(v) for v in obj]
    return str(obj)

def prune_info(info: Dict[str, Any], mode: str = "basic") -> Dict[str, Any]:
    """
    Keep compact info for logging but preserve full inventory/achievements.
    """
    if mode == "none":
        return {}
    if mode == "full":
        return json_safe(info)

    kept, pruned = {}, {}
    allow_keys = {
        "score", "x", "y", "vx", "vy", "health", "ammo", "coins", "time",
        "inventory", "achievements", "discount", "semantic", "player_pos", "reward",
    }

    def small_list(x: Any) -> bool:
        return isinstance(x, (list, tuple)) and len(x) <= 32 and all(
            isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)) for v in x
        )

    for k, v in (info or {}).items():
        try:
            kl = k.lower()
            if kl in {"rgb", "rgb_image", "pixels", "frame", "frames"}:
                if isinstance(v, np.ndarray):
                    pruned[k] = {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
                elif isinstance(v, (list, tuple)):
                    pruned[k] = {"type": "list", "len": len(v)}
                else:
                    pruned[k] = {"type": type(v).__name__}
                continue

            if k in {"inventory", "achievements"} and isinstance(v, dict):
                kept[k] = json_safe(v)
                continue

            if k in allow_keys:
                if isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)):
                    kept[k] = json_safe(v); continue
                if small_list(v):
                    kept[k] = json_safe(v); continue
                if isinstance(v, np.ndarray):
                    kept[k] = v.tolist() if v.size <= 256 else {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
                    continue

            if isinstance(v, (int, float, bool, str, np.integer, np.floating, np.bool_)):
                kept[k] = json_safe(v)
            elif small_list(v):
                kept[k] = json_safe(v)
            elif isinstance(v, np.ndarray):
                kept[k] = v.tolist() if v.size <= 256 else {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
            else:
                pruned[k] = {"type": type(v).__name__}
        except Exception as e:
            pruned[k] = {"type": type(v).__name__, "error": str(e)}

    if pruned:
        kept["_info_pruned"] = pruned
    return kept

def dict_delta(now: Dict[str, Any], prev: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(now.keys()) | set(prev.keys())
    out = {}
    for k in keys:
        a, b = now.get(k, 0), prev.get(k, 0)
        try:
            if a != b:
                out[k] = (a - b)
        except Exception:
            pass
    return out

# ---------- world-map helpers ----------

def unwrap_all(e):
    seen = set()
    cur = e
    while True:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        nxt = None
        for name in ("unwrapped", "env", "_env", "inner_env", "venv", "envs"):
            if hasattr(cur, name):
                cand = getattr(cur, name)
                if isinstance(cand, (list, tuple)) and cand:
                    cand = cand[0]
                nxt = cand
                break
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def iter_objects(root, max_depth=5):
    """
    BFS 遍历环境里的对象，寻找 2D int/bool array。
    这是你原来代码的思路。
    """
    from collections import deque
    import inspect
    q = deque([(root, 0, "<root>")])
    seen = set()
    while q:
        obj, d, path = q.popleft()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj, d, path
        if d >= max_depth:
            continue
        if isinstance(obj, (np.ndarray, bytes, str, int, float, bool)):
            continue
        if inspect.ismodule(obj) or inspect.isfunction(obj) or inspect.ismethod(obj):
            continue
        if isinstance(obj, dict):
            for k, v in list(obj.items())[:128]:
                q.append((v, d + 1, f"{path}[{repr(k)}]"))
            continue
        try:
            attrs = [a for a in dir(obj) if not a.startswith("__")]
        except Exception:
            attrs = []
        for a in attrs:
            if a in ("np_random", "random", "rng", "action_space", "observation_space"):
                continue
            try:
                v = getattr(obj, a)
            except Exception:
                continue
            q.append((v, d + 1, f"{path}.{a}"))


def is_int_grid(x):
    """跟你原来一样：只认 2D、int/bool 的 array."""
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 2
        and x.size > 0
        and (np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_))
    )


PREFERRED_TILE_KEYS = ("map", "tiles", "grid", "layout", "terrain", "worldmap")


def locate_world_grid(base):
    """
    找 2D int grid，优先路径名里带 map/tiles/grid 的，和你原来的评分方式一样。
    """
    candidates = []
    preferred = []
    for obj, d, path in iter_objects(base, max_depth=5):
        if is_int_grid(obj):
            candidates.append((path, obj.shape, obj.dtype, obj))
            for key in PREFERRED_TILE_KEYS:
                if f".{key}" in path or path.endswith(key) or f"[{repr(key)}]" in path:
                    preferred.append((path, obj.shape, obj.dtype, obj))
                    break

    def score(entry):
        path, shape, dtype, arr = entry
        h, w = shape
        # 既考虑尺寸，又稍微偏向接近正方形
        return (min(h, w), -(abs(h - w)))

    lst = preferred if preferred else candidates
    if not lst:
        raise RuntimeError("Failed to locate a 2D integer grid for the world map.")
    path, shape, dtype, arr = sorted(lst, key=score, reverse=True)[0]
    return np.array(arr), path


def dump_ascii_map_from_seed(seed: int, map_txt: str, base_dir: str) -> None:
    try:
        tmp_env = crafter.Env(seed=seed, reward=True)
        try:
            init = tmp_env.reset()
            if isinstance(init, tuple) and len(init) == 2:
                pass
        except Exception:
            try:
                tmp_env.reset()
            except Exception:
                pass

        base_tmp = unwrap_all(tmp_env)
        tiles_raw, _ = locate_world_grid(base_tmp)
        tiles_export = tiles_raw.T

        # 从 env 里拿 id->name 映射
        id2name: Dict[int, str] = {}
        for attr in ("id2name", "_id2name", "tile_id_to_name", "id_to_name"):
            if hasattr(base_tmp, attr):
                maybe = getattr(base_tmp, attr)
                if isinstance(maybe, dict) and maybe:
                    tmp_map: Dict[int, str] = {}
                    for k, v in maybe.items():
                        try:
                            k_int = int(k)
                        except Exception:
                            continue
                        tmp_map[k_int] = str(v)
                    if tmp_map:
                        id2name = tmp_map
                        break

        # 建立 id -> glyph
        unique_ids = set(int(v) for v in tiles_export.flatten())
        id2glyph: Dict[int, str] = {}
        for tid in unique_ids:
            name = id2name.get(tid)
            if name:
                glyph = name[0].upper()
            else:
                glyph = str(tid)
            id2glyph[tid] = glyph

        ascii_lines: List[str] = []
        for row in tiles_export:
            # 这一行就是你原来的行为：用 (t - 1) 做索引
            ascii_lines.append("".join(id2glyph.get(int(t) - 1, "0") for t in row))

        ascii_map = "\n".join(ascii_lines)

        # 附带一个 legend，方便人看
        legend_lines = ["", "Legend (tile_id -> name):"]
        for tid in sorted(unique_ids):
            legend_lines.append(f"  {tid} -> {id2name.get(tid, f'id{tid}')}")
        ascii_map_full = ascii_map + "\n" + "\n".join(legend_lines)

        os.makedirs(base_dir, exist_ok=True)
        with open(map_txt, "w", encoding="utf-8") as f:
            f.write(ascii_map_full + "\n")
    except Exception as e:
        print(f"[WARN] failed to dump ASCII map: {e}", file=sys.stderr)



def load_game_instructions(path: str) -> Dict[str, str]:
    """
    Supports either a single string or an array of strings. Arrays will be joined with newlines.
    """
    with open(path, "r", encoding="utf-8") as f:
        games = json.load(f).get("games", {})
    if not games:
        raise SystemExit(f"[ERROR] No 'games' key found in instruction file: {path}")
    out = {}
    for k, v in games.items():
        if isinstance(v, list):
            out[k.lower()] = "\n".join(str(x) for x in v)
        else:
            out[k.lower()] = str(v)
    return out

# ---------- stitching png utilities ----------
from PIL import ImageFont, ImageDraw

def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()

def list_pngs_sorted(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    files.sort()
    return [os.path.join(folder, f) for f in files]

def measure_caption_h(draw: ImageDraw.ImageDraw, text: str,
                      font: ImageFont.ImageFont, pad_y: int) -> Tuple[int, int, int, int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    text_h = b - t
    try:
        ascent, descent = font.getmetrics()
        text_h = max(text_h, ascent + descent)
    except Exception:
        pass
    cap_h = text_h + 2 * pad_y
    return l, t, r, b, cap_h

def stitch_frames(frames_dir: str, out_png: str,
                  font_size: int = 16, pad: int = 12,
                  cap_pad_y: int = 6, cap_pad_x: int = 8,
                  sep: int = 0, scale_width: int = 120) -> None:
    paths = list_pngs_sorted(frames_dir)
    if not paths:
        print(f"[stitch] No PNG found in: {frames_dir}")
        return
    font = load_font(font_size)
    images = [Image.open(p).convert("RGB") for p in paths]

    if scale_width and scale_width > 0:
        scaled = []
        for im in images:
            w, h = im.size
            if w != scale_width:
                nh = int(round(h * (scale_width / float(w))))
                im = im.resize((scale_width, nh), Image.BILINEAR)
            scaled.append(im)
        images = scaled

    max_w = max(im.width for im in images)
    _tmp = Image.new("RGB", (10, 10), "white")
    _draw = ImageDraw.Draw(_tmp)

    cap_info = []
    for p in paths:
        name = os.path.basename(p)
        l, t, r, b, cap_h = measure_caption_h(_draw, name, font, cap_pad_y)
        cap_info.append((name, (l, t, r, b), cap_h))

    total_h = pad
    for im, (_, _, cap_h) in zip(images, cap_info):
        total_h += im.height + cap_h + (sep if sep > 0 else 0) + pad
    canvas_w = max_w + 2 * pad
    canvas = Image.new("RGB", (canvas_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    y = pad
    for im, (name, bbox, cap_h) in zip(images, cap_info):
        x = pad + (max_w - im.width) // 2
        canvas.paste(im, (x, y))
        y += im.height

        draw.rectangle([pad, y, pad + max_w, y + cap_h], fill="white")
        draw.text((pad + cap_pad_x, y + cap_pad_y - bbox[1]), name, fill="black", font=font)
        y += cap_h

        if sep > 0:
            draw.rectangle([pad, y, pad + max_w, y + sep], fill=(230, 230, 230))
            y += sep

        y += pad

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    canvas.save(out_png)
    print("[stitch] saved:", out_png)

# ---------- dataclass ----------
@dataclass
class StepLog:
    episode: int
    step: int
    env: str
    seed: int
    model: str
    action_id: int
    action_name: str
    reward: float
    done: bool
    frame: str
    info: Dict[str, Any]
    reason: str
    llm_raw: str
    ep_return: float
    achievements_unlocked: Dict[str, int]
    inventory_delta: Dict[str, Any]
    health_delta: float
    health_events: int

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, default="crafter")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)  
    ap.add_argument("--out", type=str, default="")      
    ap.add_argument("--instr-file", type=str, default="instructions/crafter_instructions.json")
    ap.add_argument("--log-info", choices=["none", "basic", "full"], default="basic")
    ap.add_argument("--log-llm-raw", action="store_true", default=False)
    ap.add_argument("--save-frames", action="store_true", default=True)
    ap.add_argument("--history-turns", type=int, default=10, help="how many past user+assistant turns to include")
    args = ap.parse_args()

    env_name = "crafter"
    model_tag = re.sub(r"[^a-z0-9]+", "", OPENAI_MODEL.lower()) or "model"

    # --- folder layout ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_folder = f"{model_tag}_{args.steps}steps_{args.history_turns}rounds_{timestamp}"
    base_dir = os.path.join("log", f"seed{args.seed}", run_folder)
    frames_dir = os.path.join(base_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "logs.jsonl")
    stitched_path = os.path.join(base_dir, "stitched_frames.png")
    # ---- dump full ASCII map for this fixed seed  ----
    map_txt = os.path.join(base_dir, f"map_seed{args.seed}.txt")
    try:
        # Construct a temporary environment solely to export the map.  
        _tmp_env = crafter.Env(seed=args.seed, reward=True)
        try:
            _init = _tmp_env.reset()
            # swallow (obs, info) if returned
            if isinstance(_init, tuple) and len(_init) == 2:
                pass
        except Exception:
            try:
                _tmp_env.reset()
            except Exception:
                pass
        # locate the raw tile grid from the env
        base_tmp = unwrap_all(_tmp_env)
        tiles_raw, _ = locate_world_grid(base_tmp)
        # transpose so that row/col match the render orientation
        tiles_export = tiles_raw.T
        # Attempt to find an id->name mapping on the underlying env
        id2name = None
        for attr in ("id2name", "_id2name", "tile_id_to_name", "id_to_name"):
            if id2name is None and hasattr(base_tmp, attr):
                try:
                    maybe = getattr(base_tmp, attr)
                    if isinstance(maybe, dict) and maybe:
                        tmp_map = {}
                        for k, v in maybe.items():
                            try:
                                k_int = int(k)
                            except Exception:
                                continue
                            tmp_map[k_int] = str(v)
                        if tmp_map:
                            id2name = tmp_map
                            break
                except Exception:
                    pass
        if id2name is None:
            id2name = {}
        # build glyph and canon mappings
        id2glyph = {}
        id2canon = {}
        id2raw = {}
        unique_ids = set(int(v) for v in tiles_export.flatten())
        for tid in unique_ids:
            name = id2name.get(tid)
            if name:
                glyph = name[0].upper()
                canon = name
                raw = name
            else:
                glyph = str(tid)
                canon = str(tid)
                raw = str(tid)
            id2glyph[tid] = glyph
            id2canon[tid] = canon
            id2raw[tid] = raw
        # build ascii map lines
        ascii_lines = []
        for row in tiles_export:
            ascii_lines.append("".join(id2glyph.get(int(t)-1, "0") for t in row))
        ascii_map = "\n".join(ascii_lines)
        # write map text
        os.makedirs(base_dir, exist_ok=True)
        with open(map_txt, "w", encoding="utf-8") as f:
            f.write(ascii_map + "\n")
    except Exception as e:
        # if anything fails, at least record a warning
        print(f"[WARN] failed to export map: {e}")


    # ---- load instructions ----
    if not os.path.isfile(args.instr_file):
        raise SystemExit(f"[ERROR] Instruction file not found: {args.instr_file}")
    games = load_game_instructions(args.instr_file)
    game_instr = games.get(env_name, "").strip()
    if not game_instr:
        raise SystemExit(f"[ERROR] No instruction text found for env='{env_name}' in {args.instr_file}")
    system_prompt = BASE_SYSTEM + "\nGame rules: " + game_instr
    print(system_prompt)

    # ---- load icons ----
    icons_path = "instructions/crafter_icons.png"
    if not os.path.isfile(icons_path):
        raise SystemExit(f"[ERROR] Icon legend image not found: {icons_path}")
    with Image.open(icons_path) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        icons_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        icons_data_url = f"data:image/png;base64,{icons_b64}"
    if not icons_data_url:
        raise SystemExit(f"[ERROR] Failed to load icon legend image: {icons_path}")

    # legend message (injected ONCE per request)
    legend_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "ICON LEGEND: reference image for object recognition (Player, Plant, Cow, Zombie, Skeleton, Arrow, Water, Sand, Grass, Tree, Path, Stone, Coal, Iron, Diamond, Lava, Table, Furnace). Use this as ground truth for appearance."},
            {"type": "image_url", "image_url": {"url": icons_data_url}}
        ],
    }

    
    # Build the environment.  Prefer using crafter.Env directly so that the
    # world generation depends solely on the provided seed.
    try:
        env = crafter.Env(seed=args.seed, reward=True)
        _init = env.reset()
        # some versions return (obs, info)
        if isinstance(_init, tuple) and len(_init) == 2:
            obs, info = _init
        else:
            obs, info = _init, {}
    except Exception:
        env = gym.make("CrafterReward-v1", apply_api_compatibility=True)
        env.action_space.seed(args.seed)
        obs, info = safe_reset(env, seed=args.seed)
        if isinstance(obs, tuple):
            obs = obs[0]
    done, t = False, 0
    recent_actions: List[int] = []

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    ep_return = 0.0
    prev_inventory: Dict[str, Any] = {}
    prev_achievements: Dict[str, int] = {}
    last_inv: Dict[str, Any] = None

    def save_frame_from_array(arr: np.ndarray, index: int) -> str:
        fn = f"{env_name}_t{index:06d}.png"
        full = os.path.join(frames_dir, fn)
        try:
            Image.fromarray(arr).save(full)
            return fn
        except Exception as e:
            print(f"[WARN] save frame failed: {e}", file=sys.stderr)
            return ""

    def fmt_bool(x: bool) -> str:
        return "true" if x else "false"

    try:
        while (not done) and t < args.steps:
            frame_rel = ""
            # Save only the initial pre-action frame as t000000.png for reference
            if args.save_frames and obs is not None and t == 0:
                _ = save_frame_from_array(obs, 0)

            frame_np = obs if obs is not None else env.render()
            img = Image.fromarray(frame_np)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data_url = f"data:image/png;base64,{b64}"

            if isinstance(last_inv, dict):
                hp   = last_inv.get("health")
                food = last_inv.get("food")
                water= last_inv.get("drink")
                rest = last_inv.get("energy")
                wood = last_inv.get("wood"); stone = last_inv.get("stone"); coal = last_inv.get("coal")
                iron = last_inv.get("iron"); diamond = last_inv.get("diamond")
                wpk  = last_inv.get("wood_pickaxe"); spk = last_inv.get("stone_pickaxe"); ipk = last_inv.get("iron_pickaxe")
                ws   = last_inv.get("wood_sword");   ss  = last_inv.get("stone_sword");   isw = last_inv.get("iron_sword")

                needs_water_low = (water is not None and water <= 3)
                needs_food_low  = (food  is not None and food  <= 3)
                needs_rest_low  = (rest  is not None and rest  <= 3)

                hud_text = (
                    f"HUD(text): HP={hp}, Food={food}, Water={water}, Rest={rest}. "
                    f"Inv: wood={wood}, stone={stone}, coal={coal}, iron={iron}, diamond={diamond}. "
                    f"Tools: WPick={wpk}, SPick={spk}, IPick={ipk}, WSword={ws}, SSword={ss}, ISword={isw}. "
                    f"Needs: water_low={fmt_bool(needs_water_low)}, food_low={fmt_bool(needs_food_low)}, rest_low={fmt_bool(needs_rest_low)}."
                )
            else:
                hud_text = "HUD(text): unavailable (first step)."

            recent = ", ".join(ACTIONS[a] for a in recent_actions[-3:]) or "None"

            display_step = t + 1
            user_text = (
                f"{hud_text}\n"
                f"Env: {env_name}. Step {display_step}. Recent actions: {recent}. "
                "Use HUD(text) as ground truth; DO NOT restate raw numbers in your reason. "
                "An ICON LEGEND image is provided once in this request for recognition; match tiles/objects to it. "
                "Provide a brief diagnosis (e.g., thirsty/hungry/sleepy/safe) and the plan."
            )

            # step message: text + CURRENT FRAME
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
            messages.append(user_msg)

            k = max(0, int(args.history_turns))
            recent_slice = messages[max(1, len(messages)-2*k):]  # exclude system at index 0
            window = [messages[0], legend_msg] + recent_slice

            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=window,
                temperature=0,
                max_tokens=128,
            )
            out_text = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": out_text})

            a_id = parse_action_id(out_text)
            a_name = ACTIONS[a_id]
            recent_actions.append(a_id)

            # env step
            step_out = env.step(a_id)
            if len(step_out) == 4:
                obs_next, reward, done_flag, info = step_out
                terminated, truncated = done_flag, False
            else:
                obs_next, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
            reward = float(reward)

            inv_now = info.get("inventory", {}) if isinstance(info.get("inventory", {}), dict) else {}
            ach_now = info.get("achievements", {}) if isinstance(info.get("achievements", {}), dict) else {}

            ach_unlocked = {}
            for kname, v in ach_now.items():
                pv = prev_achievements.get(kname, 0)
                if isinstance(v, (int, np.integer)) and v > pv:
                    ach_unlocked[kname] = int(v - pv)

            inv_delta = dict_delta(inv_now, prev_inventory)
            achievement_points = float(sum(ach_unlocked.values()))
            health_delta = float(round(reward - achievement_points, 6))
            health_events = int(round(health_delta / 0.1)) if abs(health_delta) > 1e-8 else 0

            ep_return += reward
            prev_inventory = dict(inv_now)
            prev_achievements = {k: int(v) for k, v in ach_now.items() if isinstance(v, (int, np.integer))}

            reason = ""
            try:
                s, e = out_text.find("{"), out_text.rfind("}")
                if s != -1 and e != -1:
                    reason = json.loads(out_text[s:e+1]).get("reason", "")
            except Exception:
                pass

            info_compact = prune_info(info, mode=args.log_info)

            # Save post-action frame so that step 1 -> crafter_t000001.png, etc.
            if args.save_frames and obs_next is not None:
                frame_rel = save_frame_from_array(obs_next, t + 1)
            else:
                frame_rel = ""


            # write log line
            rec = StepLog(
                episode=0, step=display_step, env=env_name, seed=args.seed,
                model=f"openai:{OPENAI_MODEL}",
                action_id=a_id, action_name=a_name, reward=reward,
                done=done, frame=frame_rel, info=info_compact,
                reason=reason, llm_raw=(out_text if args.log_llm_raw else ""),
                ep_return=round(ep_return, 4),
                achievements_unlocked=ach_unlocked,
                inventory_delta=inv_delta,
                health_delta=round(health_delta, 4),
                health_events=health_events,
            )
            with open(out_path, "a", encoding="utf-8") as writer:
                writer.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

            # post-action frame already saved above for every step
            last_inv = inv_now if inv_now else None
            obs = obs_next
            t += 1
    finally:
        env.close()

    try:
        stitch_frames(frames_dir, stitched_path,
                      font_size=16, pad=12, cap_pad_y=6, cap_pad_x=8,
                      sep=0, scale_width=120)
    except Exception as e:
        print(f"[stitch] failed: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
