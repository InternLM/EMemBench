#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import time
import numpy as np
import os
import time

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

_OPENAI_CLIENT = None


###############################################################################
# Global constants
###############################################################################
CHAR_TO_TERRAIN: Dict[str, str] = {
    '0': 'water',
    '1': 'grass',
    '2': 'stone',
    '3': 'path',
    '4': 'sand',
    '5': 'tree',
    '6': 'coal',
    '7': 'iron',
    '8': 'diamond',
    '9': 'lava',
}
# In Crafter, some objects such as crafting table and furnace may
# occupy a tile that is not represented by a single digit.  We assign
# them new numeric identifiers (10 for table and 11 for furnace) and
# provide names for them so that terrain lookups remain consistent.
CHAR_TO_TERRAIN.update({
    '10': 'table',
    '11': 'furnace',
})

# Rebuild the inverse mapping to include the special objects.  This
# dictionary maps integer tile ids to their terrain names.
TERRAIN_ID_TO_NAME: Dict[int, str] = {int(k): v for k, v in CHAR_TO_TERRAIN.items()}

# Provide a reverse mapping from terrain name to terrain id. 
TERRAIN_NAME_TO_ID: Dict[str, int] = {}
for ch, name in CHAR_TO_TERRAIN.items():
    try:
        TERRAIN_NAME_TO_ID[name] = int(ch)
    except Exception:
        # Strings like '10' map cleanly via int()
        TERRAIN_NAME_TO_ID[name] = int(ch)

# Define how achievements map to terrain changes.  
ACHIEVEMENT_TILE_CHANGES: Dict[str, Tuple[str, str]] = {
    # Harvesting achievements: remove the resource and leave a walkable tile
    'collect_wood': ('tree', 'grass'),
    'collect_stone': ('stone', 'path'),
    'collect_coal': ('coal', 'path'),
    'collect_iron': ('iron', 'path'),
    'collect_diamond': ('diamond', 'path'),
    # Placement achievements: overwrite the target tile with a placed object
    'place_furnace': ('grass', 'furnace'),
    'place_table': ('grass', 'table'),
    'place_stone': ('path', 'stone'),
}

def compute_dynamic_maps(
    steps: List[Dict[str, Any]],
    initial_grid: np.ndarray,
    pos_by_step: List[Tuple[int, int]],
    last_move_dir_by_step: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Compute a sequence of dynamic maps, one per step, reflecting changes
    caused by achievements and placement actions.  The returned list has
    length ``len(steps)`` and element ``t`` is the map after processing
    all effects of step ``t``.
    """
    H, W = initial_grid.shape
    # Copy the starting grid so modifications don't affect the original
    grid = initial_grid.copy()
    dynamic_maps: List[np.ndarray] = []
    # Keep track of previous achievement counters
    prev_achievements: Dict[str, int] = {}
    # Track the last movement action (e.g. 'MOVE_RIGHT')
    last_move_dir: Optional[str] = None
    for idx, step in enumerate(steps):
        # Track last movement direction (for placement)
        act = step.get('action_name') or step.get('action') or ''
        if act in ["MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"]:
            last_move_dir = act
        # Extract achievements dictionary
        info = step.get("info", {})
        ach = info.get("achievements", {}) if isinstance(info, dict) else {}
        if not isinstance(ach, dict):
            ach = {}
        # Apply any newly triggered tile changes
        for ach_name, (old_name, new_name) in ACHIEVEMENT_TILE_CHANGES.items():
            prev_val = prev_achievements.get(ach_name, 0)
            # Convert current value to int (treat missing as no change)
            try:
                cur_val = int(ach.get(ach_name, prev_val))
            except Exception:
                cur_val = prev_val
            if cur_val > prev_val:
                # Determine facing direction: use provided list or track local state
                if last_move_dir_by_step is not None and idx < len(last_move_dir_by_step):
                    facing_action = last_move_dir_by_step[idx] or last_move_dir
                else:
                    facing_action = last_move_dir
                # Compute target tile coordinates in front of the player
                row, col = pos_by_step[idx]
                dx, dy = MOVE_TO_DELTA.get(facing_action, (0, 0))
                target_r, target_c = row + dy, col + dx
                if 0 <= target_r < H and 0 <= target_c < W:
                    # Determine the new id for the new terrain
                    new_id = TERRAIN_NAME_TO_ID.get(new_name)
                    if new_id is None:
                        # Assign next unused id
                        new_id = max(TERRAIN_NAME_TO_ID.values(), default=9) + 1
                        TERRAIN_NAME_TO_ID[new_name] = new_id
                        TERRAIN_ID_TO_NAME[new_id] = new_name
                    grid[target_r, target_c] = new_id
        # Update previous achievements record
        prev_achievements = {k: int(v) for k, v in ach.items()
                             if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit())}
        # Append a snapshot of the current grid
        dynamic_maps.append(grid.copy())
    return dynamic_maps

def compute_dynamic_step_data(
    steps: List[Dict[str, Any]],
    static_grid: np.ndarray,
    char_to_terrain: Dict[str, str],
    view_h: int,
    view_w: int
) -> Tuple[
    List[str],
    List[str],
    List[Dict[str, int]],
    List[Dict[str, int]],
    List[Tuple[int, int]],
    List[str],
    List[Dict[str, int]],
    List[Dict[str, int]],
    List[np.ndarray],
]:
    """
    A dynamic version of the original `compute_step_data`, but
    incorporate map changes due to achievements.  The returned
    terrain and visibility data reflect the map after each step.
    """
    # Use the original function to get basic sequences
    (actions_by_step, reasons_by_step, stats_by_step, inventory_by_step,
     pos_by_step, _terrain_unused, _vis_unused, _vis_each_unused, _adj_unused) = compute_step_data(
         steps, static_grid, char_to_terrain, view_h, view_w)
    # Build last_move_dir_by_step list to track facing direction
    last_move_dir_by_step: List[str] = []
    last_move_dir: Optional[str] = None
    for step in steps:
        act = step.get('action_name') or step.get('action') or ''
        if act in MOVE_TO_DELTA:
            last_move_dir = act
        last_move_dir_by_step.append(last_move_dir or '')

    # Compute dynamic maps based on achievements (and placements)
    dynamic_maps = compute_dynamic_maps(steps, static_grid, pos_by_step, last_move_dir_by_step)

    # Now recompute terrain and visibility using the dynamic map sequence.
    H, W = static_grid.shape
    half_h = view_h // 2
    half_w = view_w // 2
    terrain_by_step: List[str] = []
    visible_counts_by_step: List[Dict[str, int]] = []
    visible_each_counts_by_step: List[Dict[str, int]] = []
    for t, step in enumerate(steps):
        grid_t = dynamic_maps[t]
        r0, c0 = pos_by_step[t]
        # Terrain underfoot from dynamic map
        if 0 <= r0 < H and 0 <= c0 < W:
            val = int(grid_t[r0, c0])
            terr = TERRAIN_ID_TO_NAME.get(val, char_to_terrain.get(str(val % 10), 'unknown'))
        else:
            terr = "unknown"
        terrain_by_step.append(terr)
        # Visible counts
        vis_counts: Dict[str, int] = defaultdict(int)
        for dr in range(-half_h, half_h + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(-half_w, half_w + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                val2 = int(grid_t[rr, cc])
                name = TERRAIN_ID_TO_NAME.get(val2, char_to_terrain.get(str(val2 % 10), 'unknown'))
                vis_counts[name] += 1
        vis_each = {name: count for name, count in vis_counts.items() if count > 0}
        visible_counts_by_step.append(dict(vis_counts))
        visible_each_counts_by_step.append(dict(vis_each))
    return (actions_by_step, reasons_by_step, stats_by_step, inventory_by_step,
            pos_by_step, terrain_by_step, visible_counts_by_step,
            visible_each_counts_by_step, dynamic_maps)

def compute_dynamic_discovered_maps(
    pos_by_step: List[Tuple[int, int]],
    dynamic_maps: List[np.ndarray],
    view_h: int,
    view_w: int
) -> List[np.ndarray]:
    """
    Compute partial observable maps based on dynamic map sequence.  At
    each step, tiles visible within the view window are revealed from the
    corresponding dynamic map.
    """
    if not dynamic_maps:
        return []
    H, W = dynamic_maps[0].shape
    discovered = np.full((H, W), fill_value=-1, dtype=np.int32)
    half_h = view_h // 2
    half_w = view_w // 2
    maps: List[np.ndarray] = []
    for t, grid_t in enumerate(dynamic_maps):
        r0, c0 = pos_by_step[t]
        for dr in range(-half_h, half_h + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(-half_w, half_w + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                discovered[rr, cc] = grid_t[rr, cc]
        maps.append(discovered.copy())
    return maps
TERRAIN_ID_TO_NAME: Dict[int, str] = {int(k): v for k, v in CHAR_TO_TERRAIN.items()}
KEYWORDS_SET: set[str] = {
    "water", "tree", "craft", "table", "sleep", "fight", "bridge",
    "mine", "drink", "eat", "lake", "sand", "tree", "cave", "lava",
    "coal", "iron", "diamond", "wood", "stone"
}
MOVE_TO_DELTA: Dict[str, Tuple[int, int]] = {
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1)
}

###############################################################################
# Map and log loading
###############################################################################

def load_map(map_path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    lines: List[str] = []
    with open(map_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                lines.append(line)
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    grid = np.zeros((height, width), dtype=np.int32)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch.isdigit():
                grid[r, c] = int(ch)
            else:
                # non-digit fallback: treat as 0 (diamond); this is unlikely
                grid[r, c] = 0
    return grid, CHAR_TO_TERRAIN


def load_steps(log_path: str) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if log_path.endswith(".jsonl"):
        with open(log_path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    print(f"[WARN] bad json at line {ln}: {e}")
                    continue
                steps.append(obj)
    else:
        with open(log_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            steps.extend(data)
        else:
            print(f"[WARN] log file {log_path} is not a list")
    return steps


###############################################################################
# Basic per-step data extraction
###############################################################################

def compute_step_data(
    steps: List[Dict[str, Any]],
    grid: np.ndarray,
    char_to_terrain: Dict[str, str],
    view_h: int,
    view_w: int
) -> Tuple[
    List[str],               # actions_by_step
    List[str],               # reasons_by_step
    List[Dict[str, int]],    # stats_by_step
    List[Dict[str, int]],    # inventory_by_step
    List[Tuple[int, int]],   # pos_by_step
    List[str],               # terrain_by_step
    List[Dict[str, int]],    # visible_counts_by_step
    List[Dict[str, int]],    # visible_each_counts_by_step
    List[bool]               # adjacency flags (unused but reserved)
]:
    """
    Compute basic per-step data for the episode.
    """
    actions_by_step: List[str] = []
    reasons_by_step: List[str] = []
    stats_by_step: List[Dict[str, int]] = []
    inventory_by_step: List[Dict[str, int]] = []
    pos_by_step: List[Tuple[int, int]] = []
    terrain_by_step: List[str] = []
    visible_counts_by_step: List[Dict[str, int]] = []
    visible_each_counts_by_step: List[Dict[str, int]] = []
    adjacency_flags: List[bool] = []

    # For local view
    half_h = view_h // 2
    half_w = view_w // 2

    # Map grid size
    H, W = grid.shape

    for step in steps:
        # Action: prefer 'action_name' field used in Crafter logs; fall back to 'action' or NOOP.
        act = step.get("action_name") or step.get("action") or "NOOP"
        actions_by_step.append(act)
        reasons_by_step.append(step.get("reason", ""))

        s: Dict[str, int] = {}
        # Attempt to read from info.inventory first
        info = step.get("info", {})
        inv_info = info.get("inventory", {}) if isinstance(info, dict) else {}
        # Use drink→water, energy→rest mapping
        s['health'] = int(inv_info.get('health', inv_info.get('hp', inv_info.get('health', 0))) or 0) if str(inv_info.get('health', inv_info.get('hp', inv_info.get('health', 0)))).isdigit() else 0
        s['food'] = int(inv_info.get('food', inv_info.get('hunger', inv_info.get('food', 0))) or 0) if str(inv_info.get('food', inv_info.get('hunger', inv_info.get('food', 0)))).isdigit() else 0
        s['water'] = int(inv_info.get('drink', inv_info.get('water', 0)) or 0) if str(inv_info.get('drink', inv_info.get('water', 0))).isdigit() else 0
        s['rest'] = int(inv_info.get('energy', inv_info.get('rest', 0)) or 0) if str(inv_info.get('energy', inv_info.get('rest', 0))).isdigit() else 0
        # If top-level 'stats' exists, use it to override missing values
        st = step.get('stats', {})
        for key, map_key in [('health', 'health'), ('food', 'food'), ('water', 'water'), ('rest', 'rest')]:
            if key not in s or s.get(key, 0) == 0:
                try:
                    s[key] = int(st.get(map_key, s.get(key, 0)))
                except Exception:
                    pass
        stats_by_step.append(s)

        # Inventory: parse from info.inventory.  Exclude vital stats keys
        inv: Dict[str, int] = {}
        # Start with entire info.inventory
        for k, v in inv_info.items():
            # Skip stats keys
            if k in ['health', 'food', 'drink', 'energy']:
                continue
            try:
                inv[k] = int(v)
            except Exception:
                inv[k] = 0
        # Some logs may include top-level 'inventory' as resource counts; merge them
        for k, v in step.get('inventory', {}).items():
            try:
                iv = int(v)
            except Exception:
                iv = 0
            inv[k] = max(inv.get(k, 0), iv)
        inventory_by_step.append(inv)

        # Position: prefer info['player_pos'] or fallback to provided step fields
        pos = None
        if isinstance(info, dict) and 'player_pos' in info:
            pos = info.get('player_pos')
        elif 'player_pos' in step:
            pos = step.get('player_pos')
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                last_pos = (int(pos[1]), int(pos[0]))
            except Exception:
                last_pos = (0, 0)
        else:
            # if missing, default to (0,0)
            last_pos = (0, 0)
        pos_by_step.append(last_pos)

        # Terrain underfoot
        r0, c0 = last_pos
        if 0 <= r0 < H and 0 <= c0 < W:
            ch = str(int(grid[r0, c0] % 10))
            terr = char_to_terrain.get(ch, "unknown")
        else:
            terr = "unknown"
        terrain_by_step.append(terr)

        # Visible counts within view window
        vis_counts: Dict[str, int] = defaultdict(int)
        vis_each: Dict[str, int] = defaultdict(int)
        for dr in range(-half_h, half_h + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(-half_w, half_w + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                ch = str(int(grid[rr, cc] % 10))
                terr2 = char_to_terrain.get(ch, "unknown")
                vis_counts[terr2] += 1
                vis_each[terr2] += 1
        visible_counts_by_step.append(dict(vis_counts))
        visible_each_counts_by_step.append(dict(vis_each))
        adjacency_flags.append(False)

    return (actions_by_step, reasons_by_step, stats_by_step, inventory_by_step,
            pos_by_step, terrain_by_step, visible_counts_by_step,
            visible_each_counts_by_step, adjacency_flags)


###############################################################################
# Discovered map construction
###############################################################################

def compute_discovered_maps(
    pos_by_step: List[Tuple[int, int]],
    grid: np.ndarray,
    view_h: int,
    view_w: int
) -> List[np.ndarray]:
    """
    Build a sequence of partial‐observable maps. At step t, any tile
    within the view window around the player's position is revealed.
    """
    H, W = grid.shape
    discovered = np.full((H, W), fill_value=-1, dtype=np.int32)
    maps: List[np.ndarray] = []
    half_h = view_h // 2
    half_w = view_w // 2
    for (r0, c0) in pos_by_step:
        for dr in range(-half_h, half_h + 1):
            rr = r0 + dr
            if rr < 0 or rr >= H:
                continue
            for dc in range(-half_w, half_w + 1):
                cc = c0 + dc
                if cc < 0 or cc >= W:
                    continue
                discovered[rr, cc] = grid[rr, cc]
        maps.append(discovered.copy())
    return maps


###############################################################################
# Event detection
###############################################################################

def detect_attack_steps(steps: List[Dict[str, Any]], stats_by_step: List[Dict[str, int]]) -> List[int]:
    """
    Detect steps where the player was attacked (pure health drop).
    """
    attacked_steps: List[int] = []
    for i in range(1, len(stats_by_step)):
        prev = stats_by_step[i - 1]
        cur = stats_by_step[i]
        if cur.get("health", 0) < prev.get("health", 0):
            attacked_steps.append(i)
    return attacked_steps


def compute_event_indices(
    steps: List[Dict[str, Any]],
    inventory_by_step: List[Dict[str, int]],
    stats_by_step: List[Dict[str, int]]
) -> Tuple[List[int], List[int], List[int]]:
    """
    Compute indices for water drinking, food eating, and sleeping events.
    """
    drink_steps: List[int] = []
    eat_steps: List[int] = []
    sleep_steps: List[int] = []

    for i in range(1, len(steps)):
        prev_stats = stats_by_step[i - 1]
        cur_stats = stats_by_step[i]
        # Drinking: water increases
        if cur_stats.get("water", 0) > prev_stats.get("water", 0):
            drink_steps.append(i)
        # Eating: food increases
        if cur_stats.get("food", 0) > prev_stats.get("food", 0):
            eat_steps.append(i)
        # Sleep: rest increases significantly
        if cur_stats.get("rest", 0) > prev_stats.get("rest", 0) + 1:
            sleep_steps.append(i)

    return drink_steps, eat_steps, sleep_steps


###############################################################################
# Achievement and crafting events
###############################################################################

def compute_achievement_times(steps: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Compute the earliest step index where each achievement was obtained.
    """
    achievement_times: Dict[str, int] = {}
    for i, step in enumerate(steps):
        info = step.get("info", {})
        ach = info.get("achievements", {}) if isinstance(info, dict) else {}
        if not isinstance(ach, dict):
            continue
        for name, val in ach.items():
            try:
                v = int(val)
            except Exception:
                continue
            if v > 0 and name not in achievement_times:
                achievement_times[name] = i
    return achievement_times


def compute_craft_positions(
    steps: List[Dict[str, Any]],
    pos_by_step: List[Tuple[int, int]]
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Compute positions where crafting stations might have been placed.
    """
    craft_positions: Dict[str, List[Tuple[int, int]]] = {
        "table": [],
        "furnace": []
    }
    for i, step in enumerate(steps):
        action = step.get("action_name") or step.get("action") or ""
        if action in ["PLACE_TABLE", "PLACE_FURNACE"]:
            r0, c0 = pos_by_step[i]
            if action == "PLACE_TABLE":
                craft_positions["table"].append((r0, c0))
            else:
                craft_positions["furnace"].append((r0, c0))
    return craft_positions


###############################################################################
# Consumption rate estimation
###############################################################################

def compute_consumption_rate(
    values: List[int],
    replenishment_steps: List[int],
    all_steps: List[int]
) -> int:
    """
    Estimate a consumption rate for a resource by averaging the intervals
    between declines.
    """
    if not values or len(values) < 2:
        return 0
    # Collect all drops (excluding replenishments)
    drops: List[int] = []
    for i in range(1, len(values)):
        if i in replenishment_steps:
            continue
        if values[i] < values[i - 1]:
            drops.append(i)
    if len(drops) < 2:
        return 0
    intervals = [drops[i] - drops[i - 1] for i in range(1, len(drops))]
    if not intervals:
        return 0
    avg = sum(intervals) / len(intervals)
    # Round to nearest integer
    return max(1, int(round(avg)))


###############################################################################
# Death reason
###############################################################################

def compute_death_reason(steps: List[Dict[str, Any]], stats_by_step: List[Dict[str, int]]) -> str:
    """
    Heuristic computation of death reason based on the final stats.
    """
    if not steps or not stats_by_step:
        return "not answerable"
    last_stats = stats_by_step[-1]
    health = last_stats.get("health", 0)
    food = last_stats.get("food", 0)
    water = last_stats.get("water", 0)
    rest = last_stats.get("rest", 0)
    # If health is zero, check resource stats to infer cause
    if health <= 0:
        if food <= 0 and water <= 0:
            return "starvation and dehydration"
        if food <= 0:
            return "starvation"
        if water <= 0:
            return "dehydration"
        if rest <= 0:
            return "exhaustion"
        return "damage from enemies or environment"
    return "not answerable"


###############################################################################
# Paraphrasing helper (optional; used by generator)
###############################################################################

def _get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is not None:
        return _OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    _OPENAI_CLIENT = OpenAI(api_key=api_key, base_url=base_url)
    return _OPENAI_CLIENT

def paraphrase_text(text: str, rng) -> str:
    client = _get_openai_client()
    system_prompt = (
        "You are a precise paraphraser for some evaluation questions. Your goal is to improve sentence style variation."
        "Strict rules:\n"
        "1) Do NOT change meaning.\n"
        "2) Do NOT change any numbers or step indices.\n"
        "3) Preserve domain terms (actions/resources/terrains) as-is.\n"
        "4) English only. Output a single sentence without extra commentary."
    )
    user_prompt = f"Paraphrase the following sentence without changing its meaning or numbers:\n<<<{text}>>>"

    try:
        # Use a dedicated environment variable for paraphrasing model; default to gpt-4o.
        # This decouples the paraphrase model from OPENAI_MODEL used elsewhere in the
        # pipeline.  If PARAPHRASE_MODEL is unset, we fall back to "gpt-4o".
        model_name = os.environ.get("PARAPHRASE_MODEL", None)
        if model_name is None:
            model_name = "gpt-4o"
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=128,
        )
        out = resp.choices[0].message.content or ""
        return out.strip()
    except Exception as e:
        print(f"[WARN] paraphrase failed: {e}")
        return text


__all__ = [
    "CHAR_TO_TERRAIN",
    "TERRAIN_ID_TO_NAME",
    "TERRAIN_NAME_TO_ID",
    "KEYWORDS_SET",
    "MOVE_TO_DELTA",
    "load_map",
    "load_steps",
    "compute_step_data",
    "compute_discovered_maps",
    "detect_attack_steps",
    "compute_event_indices",
    "compute_achievement_times",
    "compute_craft_positions",
    "compute_consumption_rate",
    "compute_death_reason",
    "paraphrase_text",
    "ACHIEVEMENT_TILE_CHANGES",
    "compute_dynamic_maps",
    "compute_dynamic_step_data",
    "compute_dynamic_discovered_maps",
]
