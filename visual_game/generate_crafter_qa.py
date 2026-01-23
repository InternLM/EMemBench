#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re

# A global constant controlling how many questions to generate per template.
# For each template (e.g. "A_action", "B_missing_xyz", etc.), we will sample
# up to this many questions in the final output.  The default is 2.  This
# allows callers to uniformly scale the number of generated questions for
# all templates without editing individual code paths.
TEMPLATE_QUESTIONS_PER_TEMPLATE = 2
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crafter_qa_utils import (
    CHAR_TO_TERRAIN,
    KEYWORDS_SET,
    MOVE_TO_DELTA,
    TERRAIN_ID_TO_NAME,
    load_map,
    load_steps,
    compute_step_data,
    detect_attack_steps,
    compute_event_indices,
    compute_achievement_times,
    compute_craft_positions,
    compute_consumption_rate,
    compute_discovered_maps,
    compute_death_reason,
    paraphrase_text,
    compute_dynamic_step_data,
    compute_dynamic_discovered_maps,
)


USER_FRIENDLY_NAME = {
    "water": "lake",
    "grass": "grass",
    "stone": "stone",
    "path": "path",
    "sand": "sand",
    "tree": "tree",
    "lava": "lava",
    "coal": "coal",
    "iron": "iron",
    "diamond": "diamond",
    "table": "table",
    "furnace": "furnace",
}

# Default starting position on the semantic map.  Crafter episodes
# begin at (32, 32) in row‑column coordinates before any action is
# taken.  When computing displacements across the entire episode or
# counting moves starting from the first logged step, we compare
# against this position.
DEFAULT_START_POS = (32, 32)  # (row, col)


###############################################################################
# Helper functions
###############################################################################

import re

def contains_keyword(text: str, kw: str) -> bool:
    return re.search(rf"\b{re.escape(kw.lower())}\b", text.lower()) is not None


def add_item(items: List[Dict[str, Any]], qtype: str, subtype: str,
             template: str, question: str, paraphrase: str,
             difficulty: int, lo: int, hi: int, gt: Any) -> None:

    item: Dict[str, Any] = {
        "type": subtype,
        "template": template,
        "question": question,
        "paraphrase": paraphrase,
        "difficulty": int(difficulty),
        "range": [int(lo), int(hi)],
        "gt": gt,
    }
    # Override the type for unanswerable questions
    if isinstance(gt, str) and gt == "not answerable":
        item["type"] = "adversarial"
    items.append(item)


def pick_range(total_steps: int, difficulty: int, rng: random.Random) -> Tuple[int, int]:
    """
    Select an inclusive subrange of steps.

    When ``difficulty`` is negative or ``None``, the entire episode (0..total_steps-1) is used.
    For positive difficulty, instead of sampling a random sliding window, we constrain
    the range to the prefix of the episode of length ``difficulty``.  This ensures that
    range‑based questions for difficulty > 0 only consider the first n steps of the
    episode, rather than arbitrary subranges.  If the episode is shorter than the
    requested difficulty, the range is clamped to the full episode length.
    """
    # Full episode when difficulty is unspecified or negative
    if difficulty is None or difficulty < 0:
        return 0, max(0, total_steps - 1)
    # Positive difficulty: restrict to prefix of that length
    if difficulty > 0:
        hi = min(difficulty - 1, total_steps - 1)
        return 0, hi
    # In the degenerate case of difficulty == 0, fall back to full episode
    return 0, max(0, total_steps - 1)


def safe_clamp(lo: int, hi: int, total_steps: int) -> Tuple[int, int]:
    """Clamp a range to valid step indices (0 ≤ lo ≤ hi < total_steps)."""
    lo2 = max(0, min(lo, total_steps - 1))
    hi2 = max(lo2, min(hi, total_steps - 1))
    return lo2, hi2


def sample_from_list(rng: random.Random, seq: List[Any], k: int) -> List[Any]:
    """Sample up to k unique elements from seq using rng.choice.
    If seq has fewer than k elements, all elements are returned in a random order.
    """
    if not seq:
        return []
    if len(seq) <= k:
        return rng.sample(seq, len(seq))
    return rng.sample(seq, k)


def normalize_direction(dx: int, dy: int) -> str:
    """Convert a displacement into a human‑readable direction string."""
    parts: List[str] = []
    # dx affects columns: positive -> right, negative -> left
    if dx != 0:
        parts.append(f"{abs(dx)} step{'s' if abs(dx) != 1 else ''} {'right' if dx > 0 else 'left'}")
    # dy affects rows: positive -> down, negative -> up
    if dy != 0:
        parts.append(f"{abs(dy)} step{'s' if abs(dy) != 1 else ''} {'down' if dy > 0 else 'up'}")
    return " and ".join(parts) if parts else "0"


def bfs_shortest_path(
    discovered_map: np.ndarray,
    start: Tuple[int, int],
    target_ids: List[int]
) -> Optional[List[str]]:
    """Compute the shortest path to the nearest target tile on the discovered map."""
    import numpy as np  # local import to avoid global dependency
    H, W = discovered_map.shape
    sr, sc = start
    if not (0 <= sr < H and 0 <= sc < W):
        return None
    # Guard against unknown start or start being itself a target
    if discovered_map[sr, sc] in target_ids:
        return []
    # Define movement deltas
    moves = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
    walkable = set(range(10)) - {0, 6}  # avoid water (0) and lava (6)
    visited = set([(sr, sc)])
    queue = deque([((sr, sc), [])])
    while queue:
        (r, c), path = queue.popleft()
        for name, dr, dc in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if (nr, nc) in visited:
                continue
            tile_id = discovered_map[nr, nc]
            # Unknown or unwalkable
            if tile_id < 0 or tile_id not in walkable:
                continue
            if tile_id in target_ids:
                return path + [name]
            visited.add((nr, nc))
            queue.append(((nr, nc), path + [name]))
    return None


###############################################################################
# Question generation
###############################################################################

def generate_questions(
    steps: List[Dict[str, Any]],
    actions_by_step: List[str],
    reasons_by_step: List[str],
    stats_by_step: List[Dict[str, int]],
    inventory_by_step: List[Dict[str, int]],
    pos_by_step: List[Tuple[int, int]],
    terrain_by_step: List[str],
    visible_counts_by_step: List[Dict[str, int]],
    visible_each_counts_by_step: List[Dict[str, int]],
    discovered_maps: List[Any],
    attacked_steps: List[int],
    drink_steps: List[int],
    eat_steps: List[int],
    sleep_steps: List[int],
    achievement_times: Dict[str, int],
    craft_positions: Dict[str, List[Tuple[int, int]]],
    grid: Any,
    difficulty: int,
    view_h: int,
    view_w: int,
    consumption_rates: Dict[str, float],
    death_reason: Optional[str],
    rng: random.Random,
    dynamic_maps: Optional[List[np.ndarray]] = None
) -> List[Dict[str, Any]]:
    """Generate QA items across all categories for a single episode."""
    items: List[Dict[str, Any]] = []
    total_steps = len(steps)
    if total_steps == 0:
        return items

    visible_tree_coords_by_step: List[set[Tuple[int, int]]] = []
    half_h = view_h // 2
    half_w = view_w // 2
    # Determine if a dynamic map sequence was supplied
    if dynamic_maps is not None and isinstance(dynamic_maps, list) and len(dynamic_maps) == total_steps:
        # Use the last dynamic map as the working grid for BFS
        dyn_grid = dynamic_maps[-1].copy()
        H_dyn, W_dyn = dyn_grid.shape
        # Compute visible tree coordinates directly from each dynamic map
        for t in range(total_steps):
            grid_t = dynamic_maps[t]
            r0, c0 = pos_by_step[t]
            coords: set[Tuple[int, int]] = set()
            for dr in range(-half_h, half_h + 1):
                rr = r0 + dr
                if rr < 0 or rr >= H_dyn:
                    continue
                for dc in range(-half_w, half_w + 1):
                    cc = c0 + dc
                    if cc < 0 or cc >= W_dyn:
                        continue
                    if int(grid_t[rr, cc]) == 5:
                        coords.add((rr, cc))
            visible_tree_coords_by_step.append(coords)
    else:
        dyn_grid = grid.copy()
        TABLE_ID = 10
        FURNACE_ID = 11
        resource_to_tile = {
            "wood": 5,
            "stone": 2,
            "coal": 7,
            "iron": 8,
            "diamond": 9,
        }
        H_dyn, W_dyn = dyn_grid.shape
        for t in range(total_steps):
            act = actions_by_step[t]
            r0, c0 = pos_by_step[t]
            # Place stations on current tile
            if act == "PLACE_TABLE":
                if 0 <= r0 < H_dyn and 0 <= c0 < W_dyn:
                    dyn_grid[r0, c0] = TABLE_ID
            elif act == "PLACE_FURNACE":
                if 0 <= r0 < H_dyn and 0 <= c0 < W_dyn:
                    dyn_grid[r0, c0] = FURNACE_ID
            # Apply harvesting updates based on inventory delta
            if t > 0:
                for res, tid in resource_to_tile.items():
                    prev = inventory_by_step[t - 1].get(res, 0)
                    cur = inventory_by_step[t].get(res, 0)
                    if cur > prev:
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            rr = r0 + dr
                            cc = c0 + dc
                            if 0 <= rr < H_dyn and 0 <= cc < W_dyn:
                                if dyn_grid[rr, cc] == tid:
                                    dyn_grid[rr, cc] = 1 if tid == 5 else 3
                                    break
            # Record visible trees from the mutated dyn_grid
            coords: set[Tuple[int, int]] = set()
            r_center, c_center = r0, c0
            for dr in range(-half_h, half_h + 1):
                rr = r_center + dr
                if rr < 0 or rr >= H_dyn:
                    continue
                for dc in range(-half_w, half_w + 1):
                    cc = c_center + dc
                    if cc < 0 or cc >= W_dyn:
                        continue
                    if dyn_grid[rr, cc] == 5:
                        coords.add((rr, cc))
            visible_tree_coords_by_step.append(coords)

    # Precompute steps at which vital resources decreased by 1. 
    decrease_steps: Dict[str, List[int]] = {"water": [], "food": [], "rest": []}
    for name in ["water", "food", "rest"]:
        for t in range(1, total_steps):
            try:
                prev_val = stats_by_step[t - 1].get(name, 0)
                cur_val = stats_by_step[t].get(name, 0)
            except Exception:
                prev_val = 0
                cur_val = 0
            if cur_val < prev_val:
                decrease_steps[name].append(t)

    # Determine used vocabularies
    # Resources with at least one positive inventory entry
    resource_usage: Dict[str, List[int]] = defaultdict(list)
    for t, inv in enumerate(inventory_by_step):
        for k, v in inv.items():
            try:
                iv = int(v)
            except Exception:
                iv = 0
            if iv > 0:
                resource_usage[k].append(t)
    used_resources = list(resource_usage.keys())

    # Resources that never exceed zero (potential adversarial targets)
    unused_resources = []
    for k in set(k for inv in inventory_by_step for k in inv.keys()):
        if k not in resource_usage:
            unused_resources.append(k)

    # Actions executed
    action_usage: Dict[str, List[int]] = defaultdict(list)
    for t, act in enumerate(actions_by_step):
        action_usage[act].append(t)
    used_actions = list(action_usage.keys())

    # ---------------------------------------------------------------------------
    # Additional statistics for temporal reasoning (new event and state handling)
    #
    # We build a dictionary of "anchor" events capturing the first time certain
    # conditions occur within the episode.  These anchors are used to form
    # temporal questions such as ordering (before/after), intervals between
    # events, counts of events between anchors, and state queries relative to
    # anchor steps.  We also precompute lists of steps for various boolean
    # conditions (e.g. low food) to support event counting.
    anchor_events: Dict[str, int] = {}
    # First time the player placed a table
    if "PLACE_TABLE" in action_usage:
        anchor_events["FirstPlaceTable"] = action_usage["PLACE_TABLE"][0]
    # First time the player placed a furnace
    if "PLACE_FURNACE" in action_usage:
        anchor_events["FirstPlaceFurnace"] = action_usage["PLACE_FURNACE"][0]
    # First time the player slept (rest replenished)
    if sleep_steps:
        anchor_events["FirstSleep"] = sleep_steps[0]
    # First time the player drank from a lake
    if drink_steps:
        anchor_events["FirstDrink"] = drink_steps[0]
    # First time the player ate any food
    if eat_steps:
        anchor_events["FirstEat"] = eat_steps[0]
    # First time the player took damage from an enemy
    if attacked_steps:
        anchor_events["FirstAttack"] = attacked_steps[0]
    # First time food dropped below 3
    for t in range(total_steps):
        if stats_by_step[t].get("food", 0) < 3:
            anchor_events["FirstFoodBelow3"] = t
            break
    # First time water dropped below 3
    for t in range(total_steps):
        if stats_by_step[t].get("water", 0) < 3:
            anchor_events["FirstWaterBelow3"] = t
            break
    # First time rest dropped below 3
    for t in range(total_steps):
        if stats_by_step[t].get("rest", 0) < 3:
            anchor_events["FirstRestBelow3"] = t
            break
    # First time health became 2 or lower
    for t in range(total_steps):
        if stats_by_step[t].get("health", 0) <= 2:
            anchor_events["FirstHealthCritical"] = t
            break
    # First time the player stepped onto specific terrain
    for t in range(total_steps):
        if terrain_by_step[t] == "water":
            anchor_events["FirstReachWaterTile"] = t
            break
    for t in range(total_steps):
        if terrain_by_step[t] == "sand":
            anchor_events["FirstReachSandTile"] = t
            break
    for t in range(total_steps):
        if terrain_by_step[t] == "path":
            anchor_events["FirstReachPathTile"] = t
            break
    # The death step is always considered the last step of the episode
    if total_steps > 0:
        anchor_events["Death"] = total_steps - 1

    # Human‑readable descriptions for each anchor event
    event_phrases: Dict[str, str] = {
        "FirstPlaceTable": "the step when you first placed a table",
        "FirstPlaceFurnace": "the step when you first placed a furnace",
        "FirstSleep": "the step when you first went to sleep",
        "FirstDrink": "the step when you first drank from a lake",
        "FirstEat": "the step when you first ate any food",
        "FirstAttack": "the step when you first took damage from an enemy",
        "FirstFoodBelow3": "the step when your food first dropped below 3",
        "FirstWaterBelow3": "the step when your water first dropped below 3",
        "FirstRestBelow3": "the step when your rest first dropped below 3",
        "FirstHealthCritical": "the step when your health first became 2 or lower",
        "FirstReachWaterTile": "the step when you first stepped onto a water tile",
        "FirstReachSandTile": "the step when you first stepped onto a sand tile",
        "FirstReachPathTile": "the step when you first stepped onto a path tile",
        "Death": "the step when you died at the end of the episode",
    }

    # Lists of steps where certain conditions hold, used for counting events between anchors
    low_food_steps = [t for t in range(total_steps) if stats_by_step[t].get("food", 0) < 3]
    low_water_steps = [t for t in range(total_steps) if stats_by_step[t].get("water", 0) < 3]
    low_rest_steps = [t for t in range(total_steps) if stats_by_step[t].get("rest", 0) < 3]
    critical_health_steps = [t for t in range(total_steps) if stats_by_step[t].get("health", 0) <= 2]

    count_events: Dict[str, List[int]] = {
        "Drink": drink_steps,
        "Eat": eat_steps,
        "Sleep": sleep_steps,
        "Attack": attacked_steps,
        "LowFood": low_food_steps,
        "LowWater": low_water_steps,
        "LowRest": low_rest_steps,
        "CriticalHealth": critical_health_steps,
    }
    count_event_phrases: Dict[str, str] = {
        "Drink": "you drank from a lake",
        "Eat": "you ate any food",
        "Sleep": "you went to sleep",
        "Attack": "you were attacked",
        "LowFood": "your food was below 3",
        "LowWater": "your water was below 3",
        "LowRest": "your rest was below 3",
        "CriticalHealth": "your health was 2 or lower",
    }
    # State names to question phrases for state queries
    state_phrases: Dict[str, str] = {
        "health": "your health value",
        "food": "your food value",
        "water": "your water value",
        "rest": "your rest value",
        "terrain": "the terrain directly under you",
    }

    # Keywords used in reasoning
    keyword_usage: Dict[str, List[int]] = defaultdict(list)
    for t, reason in enumerate(reasons_by_step):
        for kw in KEYWORDS_SET:
            if contains_keyword(reason, kw):
                keyword_usage[kw].append(t)
    used_keywords = list(keyword_usage.keys())
    unused_keywords = [kw for kw in KEYWORDS_SET if kw not in keyword_usage]

    # Terrain visited (player standing on)
    terrain_usage: Dict[str, List[int]] = defaultdict(list)
    for t, ter in enumerate(terrain_by_step):
        terrain_usage[ter].append(t)
    used_terrains = list(terrain_usage.keys())
    # Terrains seen (in visible windows)
    seen_terrains = set()
    for t in range(total_steps):
        for terr in visible_each_counts_by_step[t].keys():
            seen_terrains.add(terr)
    seen_terrains = list(seen_terrains)
    unseen_terrains = [TER for TER in TERRAIN_ID_TO_NAME.values() if TER not in seen_terrains]

    # Precompute most common terrain.  
    most_common_terrain: Optional[str] = None
    if terrain_usage:
        # count occurrences
        freq_pairs = [(terr, len(idx_list)) for terr, idx_list in terrain_usage.items()]
        # sort by descending count then terrain name
        freq_pairs.sort(key=lambda x: (-x[1], x[0]))
        most_common_terrain = freq_pairs[0][0]

    # Define a helper to translate internal terrain name to user name
    def user_name(terrain: str) -> str:
        return USER_FRIENDLY_NAME.get(terrain, terrain)

    # Helper: choose a random subrange and embed in prefix
    def choose_range_and_prefix() -> Tuple[int, int, str]:
        """Select a subrange and generate the prefix string."""
        lo, hi = pick_range(total_steps, difficulty, rng)
        lo, hi = safe_clamp(lo, hi, total_steps)
        if difficulty is None or difficulty < 0:
            prefix = ""
        else:
            # Convert zero‑based indices to 1‑based step numbers for the prompt
            prefix = f"From step {lo + 1} to {hi + 1}, "
        return lo, hi, prefix

    # Category A: Single‑Hop Memory Retrieving
    # ---------------------------------------------------------------
    # A.1: Step attributes: action, reasoning, stats, terrain, inventory
    # Generate a few samples per sub‑type
    single_step_trials = min(6, total_steps)
    for _ in range(single_step_trials):
        lo, hi, prefix = choose_range_and_prefix()
        # pick a base step within range
        t = rng.randint(lo, hi)
        # Action: direct lookup
        act = actions_by_step[t]
        # Convert zero‑based index to 1‑based step number for the prompt
        step_num = t + 1
        q = f"What is the action at step {step_num}?"
        p = paraphrase_text(q, rng)
        add_item(items, "A", "Single‑Hop", "A_action", q, p, hi - lo + 1 if difficulty >= 0 else 1, lo, hi, act)
        # Reason
        reason = reasons_by_step[t]
        q2 = f"What was the reasoning at step {step_num}?"
        p2 = paraphrase_text(q2, rng)
        add_item(items, "A", "Single‑Hop", "A_reason", q2, p2, hi - lo + 1 if difficulty >= 0 else 1, lo, hi, reason)
        # Stat: choose a stat and prefer a step where it's non‑zero if possible
        stat = rng.choice(["health", "food", "water", "rest"])
        cand_ts = [tt for tt in range(lo, hi + 1) if stats_by_step[tt].get(stat, 0) > 0]
        t_stat = rng.choice(cand_ts) if cand_ts else t
        val = stats_by_step[t_stat].get(stat, 0)
        # Convert to 1‑based step number
        step_stat_num = t_stat + 1
        q3 = f"What was your {stat} value at step {step_stat_num}?"
        p3 = paraphrase_text(q3, rng)
        add_item(items, "A", "Single‑Hop", "A_stat", q3, p3, hi - lo + 1 if difficulty >= 0 else 1, lo, hi, val)
        # Terrain underfoot: choose a step away from the most common terrain if possible
        t_terr = t
        if most_common_terrain and (hi - lo + 1) > 1:
            # Candidate steps where terrain != most_common_terrain
            cand_ts2 = [tt for tt in range(lo, hi + 1) if terrain_by_step[tt] != most_common_terrain]
            if cand_ts2:
                t_terr = rng.choice(cand_ts2)
        terr = terrain_by_step[t_terr]
        terr_name = user_name(terr)
        step_terr_num = t_terr + 1
        q4 = f"What terrain was under you at step {step_terr_num}?"
        p4 = paraphrase_text(q4, rng)
        add_item(items, "A", "Single‑Hop", "A_terrain", q4, p4, hi - lo + 1 if difficulty >= 0 else 1, lo, hi, terr_name)
        # Inventory query for a resource: pick a resource and prefer a step where count>0
        if used_resources:
            res = rng.choice(used_resources)
            cand_ts3 = [tt for tt in range(lo, hi + 1) if inventory_by_step[tt].get(res, 0) > 0]
            t_res = rng.choice(cand_ts3) if cand_ts3 else t
            val2 = inventory_by_step[t_res].get(res, 0)
            step_res_num = t_res + 1
            q5 = f"How many {res} did you have at step {step_res_num}?"
            p5 = paraphrase_text(q5, rng)
            add_item(items, "A", "Single‑Hop", "A_inventory", q5, p5, hi - lo + 1 if difficulty >= 0 else 1, lo, hi, val2)

    # A.2: First/kth/last occurrence queries for action/keyword/terrain
    def generate_occurrence_queries(kind: str, label: str, occs: List[int]) -> None:
        # Determine phrase type (first/kth/last)
        choice = rng.choice(["first", "kth", "last"])
        if choice == "first":
            occ_idx = 0
            phrase = "first"
        elif choice == "last":
            occ_idx = -1
            phrase = "last"
        else:
            # kth: choose k between 2 and 3
            k = rng.randint(2, 3)
            occ_idx = k - 1
            if k==2:
                phrase = "second"
            else:
                phrase = "third"
        lo, hi, prefix = choose_range_and_prefix()
        # Filter occurrences within the chosen range
        occs_in_range = [t for t in occs if lo <= t <= hi]
        if len(occs_in_range) == 0 or (choice == "kth" and len(occs_in_range) <= occ_idx):
            if kind == "keyword":
                q = f"{prefix}Which step is the {phrase} step whose reason mentions '{label}'?"
            elif kind == "terrain":
                q = f"{prefix}Which step is the {phrase} step which you stood on {label} terrain?"
            else: # action
                q = f"{prefix}Which step is the {phrase} step whose action is '{label}'?"
            p = paraphrase_text(q, rng)
            add_item(items, "A", "Adversarial", f"A_occ_{kind}_missing", q, p, hi - lo + 1, lo, hi, "not answerable")
            return
        # Determine target index
        if choice == "first":
            t_occ = occs_in_range[0]
        elif choice == "last":
            t_occ = occs_in_range[-1]
        else:
            t_occ = occs_in_range[occ_idx]
        if kind == "keyword":
            q = f"{prefix}Which step is the {phrase} step whose reason mentions '{label}'?"
        elif kind == "terrain":
            q = f"{prefix}Which step is the {phrase} step which you stood on {label} terrain?"
        else: # action
            q = f"{prefix}Which step is the {phrase} step whose action is '{label}'?"
        p = paraphrase_text(q, rng)
        # Convert zero‑based index to 1‑based step number for the answer
        answer_step = t_occ + 1
        add_item(items, "A", "Single‑Hop", f"A_occ_{kind}_{phrase}", q, p, hi - lo + 1, lo, hi, answer_step)

    # Generate a handful of occurrence queries for used values
    for act in sample_from_list(rng, used_actions, min(6, len(used_actions))):
        generate_occurrence_queries("action", act, action_usage.get(act, []))
    for kw in sample_from_list(rng, used_keywords, min(6, len(used_keywords))):
        generate_occurrence_queries("keyword", kw, keyword_usage.get(kw, []))
    for terr in sample_from_list(rng, used_terrains, min(6, len(used_terrains))):
        generate_occurrence_queries("terrain", user_name(terr), terrain_usage.get(terr, []))

    # Generate adversarial occurrence queries with unused values
    for act in sample_from_list(rng, [a for a in MOVE_TO_DELTA.keys() if a not in used_actions], 2):
        generate_occurrence_queries("action", act, [])
    for kw in sample_from_list(rng, unused_keywords, 2):
        generate_occurrence_queries("keyword", kw, [])
    for terr in sample_from_list(rng, unseen_terrains, 2):
        generate_occurrence_queries("terrain", user_name(terr), [])

    # Category B: Multi‑Hop Memory Retrieving
    # ---------------------------------------------------------------
    # Choose anchor types from used sets for normal queries
    def generate_multi_hop(kind: str, label: str, occs: List[int]) -> None:
        choice = rng.choice(["first", "kth", "last"])
        if choice == "first":
            occ_idx = 0
            phrase = "first"
        elif choice == "last":
            occ_idx = -1
            phrase = "last"
        else:
            k = rng.randint(2, 3)
            occ_idx = k - 1
            if k==2:
                phrase = "second"
            else:
                phrase = "third"
        forward = rng.choice([True, False])
        offset = rng.randint(1, 5)
        dir_str = "after" if forward else "before"
        # Pick a range and limit occs to range
        lo, hi, prefix = choose_range_and_prefix()
        occs_in_range = [t for t in occs if lo <= t <= hi]
        if len(occs_in_range) == 0 or (choice == "kth" and len(occs_in_range) <= occ_idx):
            if kind == "keyword":
                q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose reason mentions '{label}'?"
            elif kind == "terrain":
                q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose terrain you stand on is '{label}'?"
            else: # action
                q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose action is '{label}'?"
            p = paraphrase_text(q, rng)
            add_item(items, "B", "Multi‑Hop", f"B_missing_{kind}", q, p, hi - lo + 1, lo, hi, "not answerable")
            return
        # Anchor step
        t_occ = occs_in_range[0] if choice == "first" else (occs_in_range[-1] if choice == "last" else occs_in_range[occ_idx])
        t_target = t_occ + offset if forward else t_occ - offset
        if kind == "keyword":
            q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose reason mentions '{label}'?"
        elif kind == "terrain":
            q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose terrain you stand on is '{label}'?"
        else: # action
            q = f"{prefix}What is the action {offset} steps {dir_str} the {phrase} step whose action is '{label}'?"
        p = paraphrase_text(q, rng)
        if lo <= t_target <= hi:
            ans = actions_by_step[t_target]
            add_item(items, "B", "Multi‑Hop", f"B_{kind}_{phrase}", q, p, hi - lo + 1, lo, hi, ans)
        else:
            add_item(items, "B", "Multi‑Hop", f"B_{kind}_{phrase}", q, p, hi - lo + 1, lo, hi, "not answerable")

    # Normal multi‑hop queries
    for act in sample_from_list(rng, used_actions, min(5, len(used_actions))):
        generate_multi_hop("action", act, action_usage.get(act, []))
    for kw in sample_from_list(rng, used_keywords, min(5, len(used_keywords))):
        generate_multi_hop("keyword", kw, keyword_usage.get(kw, []))
    for terr in sample_from_list(rng, used_terrains, min(5, len(used_terrains))):
        generate_multi_hop("terrain", user_name(terr), terrain_usage.get(terr, []))
    
    # Adversarial multi‑hop queries (missing)
    for act in sample_from_list(rng, [a for a in MOVE_TO_DELTA.keys() if a not in used_actions], 2):
        generate_multi_hop("action", act, [])
    for kw in sample_from_list(rng, unused_keywords, 2):
        generate_multi_hop("keyword", kw, [])
    for terr in sample_from_list(rng, unseen_terrains, 2):
        generate_multi_hop("terrain", user_name(terr), [])

    # -------------------------------------------------------------------
    # Helper functions for dynamic pathfinding and path compression
    from collections import deque  
    WALKABLE_TILES = {1, 3, 4, 6}

    def dynamic_bfs_path(start: Tuple[int, int], target_tile_ids: List[int]) -> Optional[List[str]]:
        """Find the shortest path from ``start`` to the nearest tile with
        id in ``target_tile_ids`` on the dynamic grid.
        Returns a list of direction strings.
        """
        sr, sc = start
        H, W = dyn_grid.shape
        if not (0 <= sr < H and 0 <= sc < W):
            return None
        # If already on target
        if dyn_grid[sr, sc] in target_tile_ids:
            return []
        moves = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
        visited: set[Tuple[int, int]] = {(sr, sc)}
        queue: deque[Tuple[Tuple[int, int], List[str]]] = deque([((sr, sc), [])])
        while queue:
            (r, c), path = queue.popleft()
            for name, dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited:
                    continue
                tile_id = dyn_grid[nr, nc]
                # Skip unwalkable tiles
                if tile_id not in WALKABLE_TILES and tile_id not in target_tile_ids:
                    continue
                new_path = path + [name]
                if tile_id in target_tile_ids:
                    return new_path
                visited.add((nr, nc))
                queue.append(((nr, nc), new_path))
        return None

    def dynamic_bfs_distance(start: Tuple[int, int], target_tile_ids: List[int]) -> Optional[int]:
        """Compute the Manhattan distance via BFS from start to the nearest
        target tile on the dynamic grid. """
        sr, sc = start
        H, W = dyn_grid.shape
        if not (0 <= sr < H and 0 <= sc < W):
            return None
        if dyn_grid[sr, sc] in target_tile_ids:
            return 0
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        visited: set[Tuple[int, int]] = {(sr, sc)}
        queue: deque[Tuple[Tuple[int, int], int]] = deque([((sr, sc), 0)])
        while queue:
            (r, c), dist = queue.popleft()
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited:
                    continue
                tile_id = dyn_grid[nr, nc]
                # skip unwalkable
                if tile_id not in WALKABLE_TILES and tile_id not in target_tile_ids:
                    continue
                if tile_id in target_tile_ids:
                    return dist + 1
                visited.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        return None

    def dynamic_bfs_all_targets(start: Tuple[int, int], target_tile_ids: List[int]) -> Tuple[Optional[int], List[Tuple[int, int]], Dict[Tuple[int, int], Tuple[Tuple[int, int], str]]]:
        """
        Find all target tiles reachable by the shortest path and record parents for path reconstruction.
        """
        sr, sc = start
        H, W = dyn_grid.shape
        if not (0 <= sr < H and 0 <= sc < W):
            return None, [], {}
        # If already on a target tile, distance is zero and no movement is needed.
        if dyn_grid[sr, sc] in target_tile_ids:
            return 0, [(sr, sc)], {}
        # BFS queue of (r, c) positions
        from collections import deque as _deque  # local alias
        q = _deque()
        q.append((sr, sc))
        visited: set[Tuple[int, int]] = {(sr, sc)}
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]] = {}
        # Keep distances separately to identify the first layer containing targets
        dist_map: Dict[Tuple[int, int], int] = {(sr, sc): 0}
        targets: List[Tuple[int, int]] = []
        found_dist: Optional[int] = None
        # Define movement options with names for reconstruction
        moves_seq = [("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1)]
        while q:
            r, c = q.popleft()
            cur_dist = dist_map[(r, c)]
            # Stop exploring deeper layers once the first target layer is found
            if found_dist is not None and cur_dist + 1 > found_dist:
                continue
            for name, dr, dc in moves_seq:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if (nr, nc) in visited:
                    continue
                tile_id = dyn_grid[nr, nc]
                # Skip if unwalkable and not a target
                if tile_id not in WALKABLE_TILES and tile_id not in target_tile_ids:
                    continue
                visited.add((nr, nc))
                dist_map[(nr, nc)] = cur_dist + 1
                parent[(nr, nc)] = ((r, c), name)
                if tile_id in target_tile_ids:
                    if found_dist is None:
                        found_dist = cur_dist + 1
                    if cur_dist + 1 == found_dist:
                        targets.append((nr, nc))
                else:
                    q.append((nr, nc))
        return found_dist, targets, parent

    def reconstruct_path(parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], str]], start: Tuple[int, int], target: Tuple[int, int]) -> List[str]:
        """Reconstruct a path from start to target using the parent map.
        """
        path: List[str] = []
        cur = target
        while cur in parent and cur != start:
            (prev, name) = parent[cur]
            path.append(name)
            cur = prev
        path.reverse()
        return path

    def compress_path(path: List[str]) -> str:
        if not path:
            return "0"
        runs = []
        cur = path[0]
        cnt = 1
        for step in path[1:]:
            if step == cur:
                cnt += 1
            else:
                runs.append((cur, cnt))
                cur, cnt = step, 1
        runs.append((cur, cnt))

        parts = [f"{n} step{'s' if n != 1 else ''} {d}" for d, n in runs]
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


    # Category C: Memory Inducing (Counting, runs, resource tracking)
    # ---------------------------------------------------------------
    # C1: Count keyword occurrences
    for kw in sample_from_list(rng, used_keywords, min(5, len(used_keywords))):
        lo, hi, prefix = choose_range_and_prefix()
        cnt = sum(1 for t in range(lo, hi + 1) if contains_keyword(reasons_by_step[t], kw))
        q = f"{prefix}How many steps have reasons that mention '{kw}'?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_keyword_count", q, p, hi - lo + 1, lo, hi, cnt)
    
    # C2: Most common movement direction
    for _ in range(5):
        lo, hi, prefix = choose_range_and_prefix()
        cnts = Counter()
        for t in range(lo, hi + 1):
            act = actions_by_step[t]
            if act in MOVE_TO_DELTA:
                cnts[act] += 1
        if cnts:
            # break ties deterministically: highest count then alphabetical
            sorted_dirs = sorted(cnts.items(), key=lambda x: (-x[1], x[0]))
            most = sorted_dirs[0][0]
            q = f"{prefix}What was the most common movement direction?"
            p = paraphrase_text(q, rng)
            add_item(items, "C", "Inducing", "C_most_move", q, p, hi - lo + 1, lo, hi, most)
        else:
            q = f"{prefix}What was the most common movement direction?"
            p = paraphrase_text(q, rng)
            add_item(items, "C", "Inducing", "C_most_move", q, p, hi - lo + 1, lo, hi, "not answerable")
    
    # C3: Longest consecutive run of a particular action
    for act in sample_from_list(rng, [a for a in used_actions if a != "NOOP"], min(5, len(used_actions))):
        lo, hi, prefix = choose_range_and_prefix()
        run_max = 0
        cur = 0
        for t in range(lo, hi + 1):
            if actions_by_step[t] == act:
                cur += 1
                if cur > run_max:
                    run_max = cur
            else:
                cur = 0
        q = f"{prefix}What was the longest consecutive run of {act}?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_longest_run", q, p, hi - lo + 1, lo, hi, run_max)
    
    # C4: Resource collection count
    for res in sample_from_list(rng, used_resources, min(5, len(used_resources))):
        lo, hi, prefix = choose_range_and_prefix()
        count_collected = 0
        for t in range(lo + 1, hi + 1):
            prev = inventory_by_step[t - 1].get(res, 0)
            cur = inventory_by_step[t].get(res, 0)
            if cur > prev:
                count_collected += 1
        q = f"{prefix}How many times did you collect {res}?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_collect_res", q, p, hi - lo + 1, lo, hi, count_collected)
    
    # C5: Inventory extreme value and change
    for res in sample_from_list(rng, used_resources, min(5, len(used_resources))):
        # Peak step
        vals = [inv.get(res, 0) for inv in inventory_by_step]
        max_val = max(vals)
        t_peak = vals.index(max_val)
        q = f"At which step did {res} reach its maximum quantity (at least 1)?"
        p = paraphrase_text(q, rng)
        # For peak queries treat them as full‑episode queries
        # Convert index to 1‑based step number for the answer when applicable
        if max_val > 0:
            ans_peak = t_peak + 1
        else:
            ans_peak = "not answerable"
        add_item(items, "C", "Inducing", "C_resource_peak", q, p, total_steps, 0, total_steps - 1, ans_peak)
        # Change in a random range
        lo, hi, prefix = choose_range_and_prefix()
        delta = inventory_by_step[hi].get(res, 0) - inventory_by_step[lo].get(res, 0)
        q2 = f"{prefix}What was the change in {res} quantity?"
        p2 = paraphrase_text(q2, rng)
        add_item(items, "C", "Inducing", "C_resource_change", q2, p2, hi - lo + 1, lo, hi, delta)
    
    # C6: Visible terrain count
    for terr in sample_from_list(rng, seen_terrains, min(5, len(seen_terrains))):
        lo, hi, prefix = choose_range_and_prefix()
        cnt = sum(1 for t in range(lo, hi + 1) if visible_each_counts_by_step[t].get(terr, 0) > 0)
        q = f"{prefix}In how many steps could you see any {user_name(terr)} in the frame?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_visible_count", q, p, hi - lo + 1, lo, hi, cnt)
    
    # C7: Adjacent terrain count
    H, W = grid.shape
    for terr in sample_from_list(rng, seen_terrains, min(3, len(seen_terrains))):
        lo, hi, prefix = choose_range_and_prefix()
        cnt_adj = 0
        for t in range(lo, hi + 1):
            r0, c0 = pos_by_step[t]
            adj = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < H and 0 <= cc < W:
                    # Use discovered map to avoid peeking into unseen territory
                    tile = discovered_maps[t][rr, cc]
                    if tile >= 0 and TERRAIN_ID_TO_NAME[tile] == terr:
                        adj = True
                        break
            if adj:
                cnt_adj += 1
        q = f"{prefix}In how many steps were you adjacent to any {user_name(terr)}?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_adjacent_count", q, p, hi - lo + 1, lo, hi, cnt_adj)

    # C8: Distinct trees seen within a range
    for _ in range(3):
        lo, hi, prefix = choose_range_and_prefix()
        # Collect unique tree coordinates observed in the range
        seen_tree_coords: set[Tuple[int, int]] = set()
        for t in range(lo, hi + 1):
            seen_tree_coords.update(visible_tree_coords_by_step[t])
        count_trees = len(seen_tree_coords)
        q = f"{prefix}How many distinct trees did you see in total?"
        p = paraphrase_text(q, rng)
        add_item(items, "C", "Inducing", "C_distinct_trees", q, p, hi - lo + 1, lo, hi, count_trees)

    # Category D: Reasoning – Spatial
    # ---------------------------------------------------------------
    # D1: Total displacement in a range
    for _ in range(5):
        lo, hi, prefix = choose_range_and_prefix()
        # Compute displacement.  When difficulty is negative (full episode),
        # measure from the default start position (32,32) to the final
        # position at hi.  Otherwise measure between the endpoints of
        # the selected range.
        if difficulty is not None and difficulty < 0:
            dx = pos_by_step[hi][1] - DEFAULT_START_POS[1]
            dy = pos_by_step[hi][0] - DEFAULT_START_POS[0]
        else:
            dx = pos_by_step[hi][1] - pos_by_step[lo][1]
            dy = pos_by_step[hi][0] - pos_by_step[lo][0]
        disp_str = normalize_direction(dx, dy)
        q = f"{prefix}What was the total displacement of you? Answer in 'x step(s) left/right and/or x step(s) up/down.'"
        p = paraphrase_text(q, rng)
        add_item(items, "D", "Spatial", "D_displacement", q, p, hi - lo + 1, lo, hi, disp_str)
    
    # D2: Real path length (number of position changes)
    for _ in range(5):
        lo, hi, prefix = choose_range_and_prefix()
        moves_count = 0
        if lo == 0:
            # Compare starting position with the first logged position
            if pos_by_step[0] != DEFAULT_START_POS:
                moves_count += 1
            start_idx = 1
        else:
            start_idx = lo + 1
        for t_idx in range(start_idx, hi + 1):
            if pos_by_step[t_idx] != pos_by_step[t_idx - 1]:
                moves_count += 1
        q = f"{prefix}How many total movement steps did you successfully take?"
        p = paraphrase_text(q, rng)
        add_item(items, "D", "Spatial", "D_path_length", q, p, hi - lo + 1, lo, hi, moves_count)
    
    # D3: Predict terrain after K steps in a direction
    directions = ["right", "left", "up", "down"]
    dir_delta = {"right": (0, 1), "left": (0, -1), "up": (-1, 0), "down": (1, 0)}
    for _ in range(min(8, total_steps)):
        lo, hi, prefix = choose_range_and_prefix()
        t0 = rng.randint(lo, hi)
        K = rng.randint(1, 4)
        direction = rng.choice(directions)
        dr, dc = dir_delta[direction]
        r0, c0 = pos_by_step[t0]
        r1 = r0 + dr * K
        c1 = c0 + dc * K
        # Convert to 1‑based step number in the prompt
        step0_num = t0 + 1
        q = f"At step {step0_num}, if you walked {direction} {K} steps, what terrain would be underfoot?"
        p = paraphrase_text(q, rng)
        # Use discovered map at t0 to determine answer
        disc_map = discovered_maps[t0]
        ans: Any
        if 0 <= r1 < disc_map.shape[0] and 0 <= c1 < disc_map.shape[1]:
            tile = disc_map[r1, c1]
            if tile >= 0 and TERRAIN_ID_TO_NAME[tile] not in ["water", "lava"]:
                ans = user_name(TERRAIN_ID_TO_NAME[tile])
            else:
                ans = "not answerable"
        else:
            ans = "not answerable"
        add_item(items, "D", "Spatial", "D_predict_terrain", q, p, hi - lo + 1, t0, t0, ans)
    
    # D4: Relative direction to nearest terrain
    for _ in range(min(8, total_steps)):
        lo, hi, prefix = choose_range_and_prefix()
        t0 = rng.randint(lo, hi)
        r0, c0 = pos_by_step[t0]

        current_under = terrain_by_step[t0]
        candidate_terrains = [ter for ter in seen_terrains if ter != current_under]
        if not candidate_terrains:
            continue

        terr = rng.choice(candidate_terrains)

        disc_map = discovered_maps[t0]
        min_dist = None
        nearest = None
        H, W = disc_map.shape
        for rr in range(H):
            for cc in range(W):
                tile = disc_map[rr, cc]
                if tile >= 0 and TERRAIN_ID_TO_NAME[tile] == terr:
                    dist = abs(rr - r0) + abs(cc - c0)
                    if min_dist is None or dist < min_dist:
                        min_dist = dist
                        nearest = (rr, cc)

        # Convert to 1‑based step number for the prompt
        step0_num = t0 + 1
        q = f"At step {step0_num}, in which direction is the nearest {user_name(terr)} relative to you?"
        p = paraphrase_text(q, rng)

        if nearest is None:
            add_item(items, "D", "Spatial", "D_nearest_dir", q, p, 1, t0, t0, "not answerable")
        else:
            # Collect all target coordinates with the same minimal Manhattan distance
            all_nearest: List[Tuple[int, int]] = []
            # The nearest coordinate from the loop above is stored in ``nearest`` and its distance in ``min_dist``
            # We'll recompute all positions with this distance.
            all_nearest.append(nearest)
            for rr in range(H):
                for cc in range(W):
                    tile = disc_map[rr, cc]
                    if tile >= 0 and TERRAIN_ID_TO_NAME[tile] == terr:
                        if (rr, cc) == nearest:
                            continue
                        if min_dist is not None and abs(rr - r0) + abs(cc - c0) == min_dist:
                            all_nearest.append((rr, cc))
            # For each nearest coordinate, compute the direction string
            dir_set: set[str] = set()
            for nr, nc in all_nearest:
                dir_parts: List[str] = []
                if nr < r0:
                    dir_parts.append("up")
                if nr > r0:
                    dir_parts.append("down")
                if nc < c0:
                    dir_parts.append("left")
                if nc > c0:
                    dir_parts.append("right")
                dir_str = "-".join(dir_parts) if dir_parts else "here"
                dir_set.add(dir_str)
            # Convert to sorted list for deterministic output
            dir_list = sorted(dir_set)
            final_ans: Any
            if len(dir_list) == 1:
                final_ans = dir_list[0]
            else:
                final_ans = dir_list
            add_item(items, "D", "Spatial", "D_nearest_dir", q, p, 1, t0, t0, final_ans)

    # D5: Navigation to nearest terrain (using dynamic map)
    for terr in sample_from_list(rng, seen_terrains, min(4, len(seen_terrains))):
        # Determine target tile ids for this terrain
        target_id_list = [tid for tid, name in TERRAIN_ID_TO_NAME.items() if name == terr]
        for _ in range(2):
            lo, hi, prefix = choose_range_and_prefix()
            t0 = rng.randint(lo, hi)
            r0, c0 = pos_by_step[t0]
            # Use dynamic BFS to find all reachable targets at the minimal distance
            dist_val, targets_list, parent_map = dynamic_bfs_all_targets((r0, c0), target_id_list)
            # Convert to 1‑based step number for the prompt
            step0_num = t0 + 1
            q = (
                f"At step {step0_num}, starting from your current position, "
                f"how should you move to reach the nearest {user_name(terr)}? "
                "Answer in 'X steps left/right and Y steps up/down', or 'not answerable' if never see the target, or 0 if you are on the target"
            )
            p = paraphrase_text(q, rng)
            # Determine answer(s)
            if dist_val is None:
                ans_list: List[str] = ["not answerable"]
            elif dist_val == 0:
                ans_list = ["0"]
            else:
                seen_paths: set[str] = set()
                for tgt in targets_list:
                    path_dirs = reconstruct_path(parent_map, (r0, c0), tgt)
                    comp = compress_path(path_dirs)
                    seen_paths.add(comp)
                if not seen_paths:
                    ans_list = ["not answerable"]
                else:
                    ans_list = sorted(seen_paths)
            # Flatten answer to a single string if there is only one option
            final_ans: Any
            if len(ans_list) == 1:
                final_ans = ans_list[0]
            else:
                final_ans = ans_list
            add_item(items, "D", "Spatial", "D_nav_to_target", q, p, 1, t0, t0, final_ans)
    
    # D6: Distance extreme (closest/furthest step to terrain) using dynamic BFS
    # Restrict to salient terrain types for the "furthest" query to avoid
    # trivial answers on widespread walkable tiles. 
    allowed_furthest = [terr for terr in ["water", "tree", "stone", "diamond"] if terr in seen_terrains]
    for terr in sample_from_list(rng, allowed_furthest, min(3, len(allowed_furthest))):
        lo, hi, prefix = choose_range_and_prefix()
        distances: List[Optional[int]] = []
        # Determine target ids for this terrain on the dynamic grid
        target_id_list = [tid for tid, name in TERRAIN_ID_TO_NAME.items() if name == terr]
        for t in range(lo, hi + 1):
            r0, c0 = pos_by_step[t]
            dist = dynamic_bfs_distance((r0, c0), target_id_list)
            distances.append(dist)
        # If no finite distances found then query is unanswerable
        if not any(d is not None for d in distances):
            qmin = f"{prefix}At which step were you closest to the nearest {user_name(terr)}? If multiple steps tie, answer the first step"
            pmin = paraphrase_text(qmin, rng)
            add_item(items, "D", "Spatial", "D_min_dist", qmin, pmin, hi - lo + 1, lo, hi, "not answerable")
            qmax = f"{prefix}At which step were you furthest from the nearest {user_name(terr)}? If multiple steps tie, answer the first step."
            pmax = paraphrase_text(qmax, rng)
            add_item(items, "D", "Spatial", "D_max_dist", qmax, pmax, hi - lo + 1, lo, hi, "not answerable")
        else:
            # Replace None distances with a large number for max distance
            finite_vals = [d for d in distances if d is not None]
            max_val = max(finite_vals) if finite_vals else 0
            normalised = [d if d is not None else max_val + 1 for d in distances]
            # Determine minimal and maximal finite distances
            min_val = min(d for d in distances if d is not None)
            max_val2 = max(d for d in distances if d is not None)
            # Collect all indices (relative to range) that match the min and max distance
            min_indices: List[int] = [lo + i for i, d in enumerate(distances) if d is not None and d == min_val]
            max_indices: List[int] = [lo + i for i, d in enumerate(distances) if d is not None and d == max_val2]
            # Convert indices to 1‑based step numbers for the answers
            min_steps = [idx + 1 for idx in min_indices]
            max_steps = [idx + 1 for idx in max_indices]
            qmin = f"{prefix}At which step were you closest to the nearest {user_name(terr)}? If multiple steps tie, answer the first step."
            pmin = paraphrase_text(qmin, rng)

            qmax = f"{prefix}At which step were you furthest from the nearest {user_name(terr)}? If multiple steps tie, answer the first step."
            pmax = paraphrase_text(qmax, rng)

            ans_min = sorted(min_steps)[0]
            ans_max = sorted(max_steps)[0] 

            add_item(items, "D", "Spatial", "D_min_dist", qmin, pmin, hi - lo + 1, lo, hi, ans_min)
            add_item(items, "D", "Spatial", "D_max_dist", qmax, pmax, hi - lo + 1, lo, hi, ans_max)


    # Category E: Reasoning – Time
    # ---------------------------------------------------------------
    # The following templates implement temporal reasoning questions without
    # assumptions about constant rates or hypothetical replenishments.  We use
    # anchor events (first occurrences of specific conditions) to ask about
    # ordering, intervals, counts, and state queries relative to these events.

    # E_event_order: Did one event happen before or after another?
    # We generate multiple candidate questions and leave it to post‑processing
    # to cap the number per template.
    event_keys = list(anchor_events.keys())
    # Generate up to 5 candidate ordering questions
    for _ in range(5):
        if len(event_keys) < 2:
            break
        # Choose two distinct anchor events
        a_key, b_key = rng.sample(event_keys, 2)
        a_idx = anchor_events.get(a_key)
        b_idx = anchor_events.get(b_key)
        if a_idx is None or b_idx is None or a_key == b_key:
            continue
        # Formulate the question asking whether A occurred before or after B
        q = f"Did {event_phrases[a_key]} happen before or after {event_phrases[b_key]}?"
        p = paraphrase_text(q, rng)
        # Determine answer
        ans = "before" if a_idx < b_idx else "after"
        add_item(
            items,
            "E",
            "Time",
            "E_event_order",
            q,
            p,
            total_steps,
            0,
            total_steps - 1,
            ans,
        )

    # E_event_interval: Steps between two anchor events
    for _ in range(5):
        if len(event_keys) < 2:
            break
        a_key, b_key = rng.sample(event_keys, 2)
        a_idx = anchor_events.get(a_key)
        b_idx = anchor_events.get(b_key)
        if a_idx is None or b_idx is None or a_key == b_key:
            continue
        # Ensure the first event happens before the second; skip if equal or reverse
        if a_idx >= b_idx:
            continue
        diff = b_idx - a_idx
        q = f"After {event_phrases[a_key]}, after how many steps did {event_phrases[b_key]} occur?"
        p = paraphrase_text(q, rng)
        # Range is the segment between the two events
        add_item(
            items,
            "E",
            "Time",
            "E_event_interval",
            q,
            p,
            diff,
            a_idx,
            b_idx,
            diff,
        )

    # E_event_count: Count occurrences of a condition between two anchor events
    for _ in range(5):
        if len(event_keys) < 2:
            break
        a_key, b_key = rng.sample(event_keys, 2)
        a_idx = anchor_events.get(a_key)
        b_idx = anchor_events.get(b_key)
        if a_idx is None or b_idx is None or a_key == b_key:
            continue
        # Ensure correct ordering
        if a_idx >= b_idx:
            continue
        # Pick a condition to count
        count_key = rng.choice(list(count_events.keys()))
        events_list = count_events[count_key]
        # Only generate a question if there is at least one occurrence of this condition
        if not events_list:
            continue
        # Count occurrences in inclusive range [a_idx, b_idx]
        cnt = sum(1 for t in events_list if a_idx <= t <= b_idx)
        q = (
            f"Starting from the step where {event_phrases[a_key]} happened up to and including the step where {event_phrases[b_key]} happened, "
            f"how many steps were there where {count_event_phrases[count_key]}?"
        )
        p = paraphrase_text(q, rng)
        add_item(
            items,
            "E",
            "Time",
            "E_event_count",
            q,
            p,
            b_idx - a_idx + 1,
            a_idx,
            b_idx,
            cnt,
        )

    # E_state_after_event: Query state K steps after an anchor event
    state_keys = ["health", "food", "water", "rest", "terrain"]
    for _ in range(5):
        if not anchor_events:
            break
        a_key = rng.choice(event_keys)
        a_idx = anchor_events.get(a_key)
        if a_idx is None:
            continue
        # Choose an offset K such that a_idx + K is within bounds
        max_offset = total_steps - 1 - a_idx
        if max_offset <= 0:
            continue
        K = rng.randint(1, min(3, max_offset))
        state_key = rng.choice(state_keys)
        idx = a_idx + K
        # Retrieve the state value at the target step
        if state_key == "health":
            ans_val: Any = stats_by_step[idx].get("health", 0)
        elif state_key == "food":
            ans_val = stats_by_step[idx].get("food", 0)
        elif state_key == "water":
            ans_val = stats_by_step[idx].get("water", 0)
        elif state_key == "rest":
            ans_val = stats_by_step[idx].get("rest", 0)
        else:
            ans_val = terrain_by_step[idx]
        step_word = "step" if K == 1 else "steps"
        q = f"{K} {step_word} after {event_phrases[a_key]}, what was {state_phrases[state_key]}?"
        p = paraphrase_text(q, rng)
        add_item(
            items,
            "E",
            "Time",
            "E_state_after_event",
            q,
            p,
            1,
            idx,
            idx,
            ans_val,
        )

    # E_state_before_event: Query state K steps before an anchor event
    for _ in range(5):
        if not anchor_events:
            break
        b_key = rng.choice(event_keys)
        b_idx = anchor_events.get(b_key)
        if b_idx is None:
            continue
        # Choose an offset K such that b_idx - K >= 0
        if b_idx <= 0:
            continue
        max_offset = min(3, b_idx)
        K = rng.randint(1, max_offset)
        state_key = rng.choice(state_keys)
        idx = b_idx - K
        if state_key == "health":
            ans_val: Any = stats_by_step[idx].get("health", 0)
        elif state_key == "food":
            ans_val = stats_by_step[idx].get("food", 0)
        elif state_key == "water":
            ans_val = stats_by_step[idx].get("water", 0)
        elif state_key == "rest":
            ans_val = stats_by_step[idx].get("rest", 0)
        else:
            ans_val = terrain_by_step[idx]
        step_word = "step" if K == 1 else "steps"
        q = f"{K} {step_word} before {event_phrases[b_key]}, what was {state_phrases[state_key]}?"
        p = paraphrase_text(q, rng)
        add_item(
            items,
            "E",
            "Time",
            "E_state_before_event",
            q,
            p,
            1,
            idx,
            idx,
            ans_val,
        )

    # Category F: Reasoning – Logical
    # ---------------------------------------------------------------
    # F1: Crafting feasibility
    # Define crafting recipes and requirements
    craftables = {
        "wood_pickaxe": {"items": {"wood": 1}, "table": True, "furnace": False},
        "stone_pickaxe": {"items": {"wood": 1, "stone": 1}, "table": True, "furnace": False},
        "iron_pickaxe": {"items": {"wood": 1, "coal": 1, "iron": 1}, "table": True, "furnace": True},
        "wood_sword": {"items": {"wood": 1}, "table": True, "furnace": False},
        "stone_sword": {"items": {"wood": 1, "stone": 1}, "table": True, "furnace": False},
        "iron_sword": {"items": {"wood": 1, "coal": 1, "iron": 1}, "table": True, "furnace": True},
        # placing stations require resources too
        "place_table": {"items": {"wood": 2}, "table": False, "furnace": False},
        "place_furnace": {"items": {"stone": 2}, "table": False, "furnace": False},
    }
    for craft_name, req in craftables.items():
        for _ in range(3):
            lo, hi, prefix = choose_range_and_prefix()
            t0 = rng.randint(lo, hi)
            inv = inventory_by_step[t0]
            # Check resource counts
            has_items = all(inv.get(k, 0) >= v for k, v in req["items"].items())
            ans = "yes" if has_items else "no"
            # Convert to 1‑based step number for the prompt
            step0_num = t0 + 1
            # Formulate the query based on whether this is a placement or a build action
            if craft_name.startswith("place_"):
                # Ask about placing a station without the "build" prefix
                q = f"At step {step0_num}, are the collected resources enough to {craft_name}?"
            else:
                q = f"At step {step0_num}, are the collected resources enough to build a {craft_name}?"
            p = paraphrase_text(q, rng)
            add_item(items, "F", "Logical", "F_craft_feasibility", q, p, hi - lo + 1, lo, hi, ans)
    
    # F2: Event localisation (drink/eat/sleep) with string answers
    # For each event type (drink/eat/sleep), we ask about occurrences within a
    # selected range.  We skip cases with no occurrences instead of marking
    # them as unanswerable.  Answers for drink/eat are comma‑separated step
    # numbers; for sleep we report only the first occurrence in range.
    for event_name, event_list in [
        ("drink", drink_steps),
        ("eat", eat_steps),
        ("sleep", sleep_steps),
    ]:
        for _ in range(3):
            lo, hi, prefix = choose_range_and_prefix()
            # Collect event steps within the selected range
            event_steps = [t for t in event_list if lo <= t <= hi]
            if not event_steps:
                # Do not generate a question if the event never occurs in the range
                continue
            # Convert to 1‑based step numbers and sort for consistency
            event_steps_num = sorted([t_idx + 1 for t_idx in event_steps])
            if event_name == "sleep":
                q = f"{prefix}At which step did you sleep for the first time?"
                p = paraphrase_text(q, rng)
                ans_event = str(event_steps_num[0])
                add_item(items, "F", "Logical", "F_event_loc", q, p, hi - lo + 1, lo, hi, ans_event)
            else:
                q = f"{prefix}At which steps did you {event_name} any resource?"
                p = paraphrase_text(q, rng)
                ans_event = ", ".join(str(s) for s in event_steps_num)
                add_item(items, "F", "Logical", "F_event_loc", q, p, hi - lo + 1, lo, hi, ans_event)
    
    # F3: Attack count in a range
    for _ in range(4):
        lo, hi, prefix = choose_range_and_prefix()
        attacks = [t for t in attacked_steps if lo <= t <= hi]
        q = f"{prefix}How many times were you attacked?"
        p = paraphrase_text(q, rng)
        add_item(items, "F", "Logical", "F_attack_count", q, p, hi - lo + 1, lo, hi, len(attacks))
    # F4: Death reason classification
    # Only ask if episode ended in death
    q_dr = "What was the cause of your death at the end of the game?"
    p_dr = paraphrase_text(q_dr, rng)
    if death_reason:
        add_item(items, "F", "Logical", "F_death_reason", q_dr, p_dr, total_steps, 0, total_steps - 1, death_reason)
    else:
        add_item(items, "F", "Logical", "F_death_reason", q_dr, p_dr, total_steps, 0, total_steps - 1, "not answerable")

    # F5: First and last attack steps
    # First attack step (if any)
    if attacked_steps:
        first_attack_step_num = attacked_steps[0] + 1
        q_first = "At which step were you attacked for the first time?"
        p_first = paraphrase_text(q_first, rng)
        add_item(items, "F", "Logical", "F_first_attack_step", q_first, p_first, total_steps, 0, total_steps - 1, str(first_attack_step_num))
        # Last attack step
        last_attack_step_num = attacked_steps[-1] + 1
        q_last = "At which step were you attacked for the last time?"
        p_last = paraphrase_text(q_last, rng)
        add_item(items, "F", "Logical", "F_last_attack_step", q_last, p_last, total_steps, 0, total_steps - 1, str(last_attack_step_num))

    # F6: Inventory contents at a specific step
    # Generate a few questions about the player's inventory at random steps
    inventory_trials = min(4, total_steps)
    for _ in range(inventory_trials):
        t0 = rng.randint(0, total_steps - 1)
        step_num = t0 + 1
        inv = inventory_by_step[t0]
        # Filter out items with zero quantity and sort alphabetically by name
        nonzero_items = [(k, v) for k, v in inv.items() if isinstance(v, (int, float)) and v > 0]
        if not nonzero_items:
            ans_inv = "empty"
        else:
            nonzero_items.sort(key=lambda kv: kv[0])
            ans_inv = ", ".join(f"{name}:{int(count)}" for name, count in nonzero_items)
        q_inv = (
            f"At step {step_num}, what was the content of your inventory? "
            "Answer as a comma-separated list of 'item:quantity' pairs, or 'empty' if there were no items."
        )
        p_inv = paraphrase_text(q_inv, rng)
        add_item(items, "F", "Logical", "F_inventory_contents", q_inv, p_inv, 1, t0, t0, ans_inv)

    # Category G: Adversarial
    # ---------------------------------------------------------------
    # Construct adversarial queries by selecting slots that were never used
    # or target unreachable conditions.  
    
    # G1: Ask for the first occurrence of an unused action
    for act in sample_from_list(rng, [a for a in MOVE_TO_DELTA.keys() if a not in used_actions], 2):
        lo, hi, prefix = choose_range_and_prefix()
        q = f"{prefix}Which step is the first step whose action is '{act}'?" 
        p = paraphrase_text(q, rng)
        add_item(items, "G", "Adversarial", "G_unused_action", q, p, hi - lo + 1, lo, hi, "not answerable")
    
    # G2: Ask for the first occurrence of an unused keyword
    for kw in sample_from_list(rng, unused_keywords, 2):
        lo, hi, prefix = choose_range_and_prefix()
        q = f"{prefix}Which step is the first step whose reason mentions '{kw}'?"
        p = paraphrase_text(q, rng)
        add_item(items, "G", "Adversarial", "G_unused_keyword", q, p, hi - lo + 1, lo, hi, "not answerable")
    
    # G3: Ask for an unreachable navigation path (unknown terrain)
    for terr in sample_from_list(rng, unseen_terrains, 2):
        lo, hi, prefix = choose_range_and_prefix()
        t0 = rng.randint(lo, hi)
        # Convert index to 1‑based step number in the prompt
        step0_num = t0 + 1
        q = (
                f"{prefix}At step {step0_num}, starting from your current position, "
                f"how should you move to reach the nearest {user_name(terr)}? "
                "Answer in 'X steps left/right and Y steps up/down', or 'not answerable' if never see the target, or 0 if you are on the target"
            )
        p = paraphrase_text(q, rng)
        add_item(items, "G", "Adversarial", "G_unreachable_nav", q, p, hi - lo + 1, lo, hi, "not answerable")
    
    # G4: Ask for a resource peak for a resource never collected
    for res in sample_from_list(rng, unused_resources, 2):
        q = f"At which step did {res} reach its maximum quantity (at least 1)?"
        p = paraphrase_text(q, rng)
        add_item(items, "G", "Adversarial", "G_unused_resource_peak", q, p, total_steps, 0, total_steps - 1, "not answerable")

    return items


# -----------------------------------------------------------------------------
# Post‑processing helpers
#
# We limit the number of questions per template and selectively paraphrase only
# a subset of those questions.  This helper groups the generated items by
# ``template``, samples at most ``max_per_template`` items from each group using
# the provided RNG, and then paraphrases roughly half of the selected items.
def prune_and_paraphrase_items(
    items: List[Dict[str, Any]],
    max_per_template: int,
    rng: random.Random,
    paraphrase_func=paraphrase_text,
) -> List[Dict[str, Any]]:
    """
    Given a list of QA items, reduce the count of items for each template
    to at most ``max_per_template`` and paraphrase a subset of the kept items.

    Parameters
    ----------
    items : list of dict
        All generated QA items (each item must have a ``template`` key and
        ``question`` field).  The ``paraphrase`` field will be overwritten
        based on whether the item is selected for paraphrasing.
    max_per_template : int
        Maximum number of items to keep per distinct template name.  If a
        template has fewer than this many items, all of them are kept.
    rng : random.Random
        A random number generator for deterministic sampling and shuffling.
    paraphrase_func : callable, optional
        Function used to paraphrase a question.  It should accept
        ``(question: str, rng)`` and return a string.  Defaults to the
        imported ``paraphrase_text``.

    Returns
    -------
    List[Dict[str, Any]]
        A new list of items satisfying the per‑template limit and with
        paraphrase fields adjusted accordingly.
    """
    # Group items by template
    by_template: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        tpl = item.get("template")
        by_template[tpl].append(item)

    final_items: List[Dict[str, Any]] = []
    for tpl, group in by_template.items():
        # Shuffle group to avoid bias in selection order
        group_copy = list(group)
        rng.shuffle(group_copy)
        # Determine how many items to keep
        k = max_per_template if max_per_template is not None else len(group_copy)
        k = int(k)
        if k < 0:
            k = 0
        if k > len(group_copy):
            k = len(group_copy)
        selected = group_copy[:k]
        # Decide how many of the selected items to paraphrase.
        # We paraphrase roughly half of them (rounding up), so that at
        # least one item is paraphrased when k is odd and > 0.
        if k == 0:
            continue
        # Determine number to paraphrase.  We use ceil(k / 2) to ensure at
        # least half (and at least one when k is odd).
        n_para = (k + 1) // 2
        # Randomly select which indices will be paraphrased
        idxs = list(range(k))
        rng.shuffle(idxs)
        para_idx_set = set(idxs[:n_para])
        # For each selected item, update the paraphrase field
        for idx, it in enumerate(selected):
            # Make a shallow copy so we don't mutate original input
            new_it = it.copy()
            q_text = new_it.get("question", "")
            # If this item is chosen for paraphrasing, call paraphrase_func
            if idx in para_idx_set:
                try:
                    new_it["paraphrase"] = paraphrase_func(q_text, rng)
                except Exception:
                    new_it["paraphrase"] = q_text
            else:
                new_it["paraphrase"] = q_text
            final_items.append(new_it)
    return final_items


# ===== Dedup helper =====
def dedupe_items(items: List[Dict[str, Any]], mode: str = "question") -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        if mode == "question":
            key = it.get("question", "")
        else:
            key = (
                it.get("type", ""),
                it.get("template", ""),
                it.get("question", ""),
                tuple(it.get("range", [])) if isinstance(it.get("range", []), list) else it.get("range", []),
            )
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out



###############################################################################
# Main entry point
###############################################################################

def main():
    FOLDER_NAME = "gpt41_500steps_10rounds_20251029-164157"
    MAP_SEED = "1"
    DIFFICULTY = 100
    parser = argparse.ArgumentParser(description="Generate Crafter QA using the enhanced framework")
    # Default paths reflect the new directory layout where seed-specific runs are stored under log/seed{MAP_SEED}/... and generated_qa/seed{MAP_SEED}/...
    parser.add_argument(
        "--log-file",
        type=str,
        default=f"/Users/xinzeli/Documents/mem_eval_game/visual_game/log/seed{MAP_SEED}/{FOLDER_NAME}/logs.jsonl",
        help="Path to the logs.jsonl file",
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=f"/Users/xinzeli/Documents/mem_eval_game/visual_game/log/seed{MAP_SEED}/{FOLDER_NAME}/map_seed{MAP_SEED}.txt",
        help="Path to the semantic map file (map_seedXXXX.txt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=f"/Users/xinzeli/Documents/mem_eval_game/visual_game/generated_qa/seed{MAP_SEED}/{FOLDER_NAME}/DIF_{DIFFICULTY}",
        help="Directory where qa.jsonl and qa_context.json will be written",
    )
    parser.add_argument("--difficulty", type=int, default=DIFFICULTY, help="Length of step ranges for range‑based questions; -1 for full episode")
    parser.add_argument("--view-height", type=int, default=7, help="Local view height (odd) for visible and discovered maps (default 7)")
    parser.add_argument("--view-width", type=int, default=9, help="Local view width (odd) for visible and discovered maps (default 9)")
    args = parser.parse_args()

    # Load data
    steps = load_steps(args.log_file)
    if not steps:
        raise SystemExit(f"No steps loaded from {args.log_file}")
    grid, char_to_terrain = load_map(args.map_file)
    # Compute per‑step derived data using dynamic map tracking. 
    (actions_by_step, reasons_by_step, stats_by_step, inventory_by_step,
     pos_by_step, terrain_by_step, visible_counts_by_step,
     visible_each_counts_by_step, dynamic_maps) = compute_dynamic_step_data(
         steps, grid, char_to_terrain, args.view_height, args.view_width)

    # Build discovered maps based on the dynamic maps.  
    discovered_maps = compute_dynamic_discovered_maps(pos_by_step, dynamic_maps, args.view_height, args.view_width)
    # Detect attack events.  
    attacked_steps = detect_attack_steps(steps, stats_by_step)
    drink_steps, eat_steps, sleep_steps = compute_event_indices(steps, inventory_by_step, stats_by_step)
    # Achievement times and crafting station positions
    achievement_times = compute_achievement_times(steps)
    craft_positions = compute_craft_positions(steps, pos_by_step)
    # Estimate consumption rates for water, food, rest
    consumption_rates = {
        "water": compute_consumption_rate([s.get("water", 0) for s in stats_by_step], drink_steps, list(range(len(steps)))),
        "food": compute_consumption_rate([s.get("food", 0) for s in stats_by_step], eat_steps, list(range(len(steps)))),
        "rest": compute_consumption_rate([s.get("rest", 0) for s in stats_by_step], sleep_steps, list(range(len(steps)))),
    }
    # Compute death reason using updated heuristics. 
    death_reason = compute_death_reason(steps, stats_by_step)
    # Random generator with fixed seed for determinism
    rng = random.Random(1234)
    # Generate QA items.  
    items = generate_questions(
        steps, actions_by_step, reasons_by_step, stats_by_step, inventory_by_step,
        pos_by_step, terrain_by_step, visible_counts_by_step, visible_each_counts_by_step,
        discovered_maps, attacked_steps, drink_steps, eat_steps, sleep_steps,
        achievement_times, craft_positions, grid,
        args.difficulty, args.view_height, args.view_width,
        consumption_rates, death_reason, rng,
        dynamic_maps=dynamic_maps,
    )

    # ===== Deduplicate only when difficulty == -1 (global) =====
    if args.difficulty == -1:
        before = len(items)
        items = dedupe_items(items, mode="question")
        after = len(items)
        print(f"[dedupe] difficulty = -1, removed duplicates: {before} -> {after}")

    # ------------------------------------------------------------------
    # Limit the number of questions per template and paraphrase half of
    # the selected questions.  This post‑processing step applies a
    # uniform cap across all templates (using the global constant
    # TEMPLATE_QUESTIONS_PER_TEMPLATE) and rewrites the "paraphrase" field
    # for the retained items.  It uses the same RNG as generate_questions
    # for deterministic behavior.
    items = prune_and_paraphrase_items(items, TEMPLATE_QUESTIONS_PER_TEMPLATE, rng)

    # Prepare QA context: record of t, action, reason, frame, done and key resource stats
    # Extend each context entry with health, food, water and energy values derived from stats_by_step.
    qa_context: List[Dict[str, Any]] = []
    for idx, s in enumerate(steps):
        ctx = {
            "t": int(s.get("step", 0)),
            "action": s.get("action_name", ""),
            "reason": s.get("reason", ""),
            "frame": s.get("frame", ""),
            "done": bool(s.get("done", False)),
        }
        # Append vital resource values; if stats are unavailable, default to 0
        try:
            stat_dict = stats_by_step[idx] if idx < len(stats_by_step) else {}
        except Exception:
            stat_dict = {}
        ctx["health"] = int(stat_dict.get("health", 0))
        ctx["food"] = int(stat_dict.get("food", 0))
        ctx["water"] = int(stat_dict.get("water", 0))
        # In the raw statistics, 'rest' corresponds to player energy; rename it to energy for clarity
        ctx["energy"] = int(stat_dict.get("rest", 0))
        qa_context.append(ctx)
    # Write outputs
    os.makedirs(args.output_dir, exist_ok=True)
    qa_path = os.path.join(args.output_dir, "qa.jsonl")
    ctx_path = os.path.join(args.output_dir, "qa_context.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(qa_context, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(items)} QA items to {qa_path}")
    print(f"Wrote context for {len(qa_context)} steps to {ctx_path}")

    # Save dynamic maps only for steps where the map changed. 
    dyn_map_changes_path = os.path.join(args.output_dir, "dynamic_maps.txt")
    changed_steps: List[int] = []
    # Identify changes by comparing successive maps
    prev_map: Optional[np.ndarray] = None
    for idx, dm in enumerate(dynamic_maps):
        if prev_map is None:
            changed_steps.append(idx)
        else:
            # If any cell differs, mark this step as changed
            if dm.shape != prev_map.shape or not np.array_equal(dm, prev_map):
                changed_steps.append(idx)
        prev_map = dm
    with open(dyn_map_changes_path, "w", encoding="utf-8") as f:
        for idx in changed_steps:
            dm = dynamic_maps[idx]
            # Write 1‑based step number in the comment for clarity
            f.write(f"# Step {idx + 1}\n")
            # Convert numeric ids to string representation.  Use 't' for 10,
            # 'f' for 11, and digits for 0‑9.  Unknown ids are printed as
            # decimal numbers.
            for r in range(dm.shape[0]):
                line_chars: List[str] = []
                for c in range(dm.shape[1]):
                    val = int(dm[r, c])
                    if val == 10:
                        line_chars.append('t')
                    elif val == 11:
                        line_chars.append('f')
                    elif 0 <= val <= 9:
                        line_chars.append(str(val))
                    else:
                        line_chars.append(str(val))
                f.write("".join(line_chars) + "\n")
            f.write("\n")
    print(f"Wrote dynamic maps for {len(changed_steps)} steps to {dyn_map_changes_path}")


if __name__ == "__main__":
    main()