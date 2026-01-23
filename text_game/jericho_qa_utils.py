#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import random
import os
from collections import defaultdict, Counter, deque
from typing import Any, Dict, List, Optional, Tuple, Iterable

__all__ = [
    "load_jsonl",
    "JerichoQAUtils",
]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file, returning a list of Python dictionaries.

    Lines that do not parse as JSON are silently skipped.  The original row
    number (1-based) is stored in the ``_row`` key of each record for
    diagnostic purposes.

    Args:
        path: Path to the JSONL file on disk.

    Returns:
        A list of parsed JSON objects in the order they appear in the file.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # Skip malformed lines
                continue
            obj.setdefault("_row", i)
            records.append(obj)
    return records


class JerichoQAUtils:

    DIRECTION_ALIASES = {
        "north": "north", "n": "north",
        "south": "south", "s": "south",
        "east": "east", "e": "east",
        "west": "west", "w": "west",
        "northeast": "northeast", "ne": "northeast",
        "northwest": "northwest", "nw": "northwest",
        "southeast": "southeast", "se": "southeast",
        "southwest": "southwest", "sw": "southwest",
        "up": "up", "u": "up",
        "down": "down", "d": "down",
        "in": "in", "inside": "in",
        "out": "out", "outside": "out",
    }

    PURE_DIRECTIONS = set(DIRECTION_ALIASES.values())

    def __init__(self, records: List[Dict[str, Any]]):

        def sort_key(rec: Dict[str, Any], idx: int) -> Tuple[int, Any, int]:
            step = rec.get("step")
            if step is not None:
                try:
                    st = int(step)
                    return (0, st, idx)
                except Exception:
                    pass
            ts = rec.get("timestamp")
            if isinstance(ts, str):
                return (1, ts, idx)
            return (2, idx, idx)

        self.recs = [r for _, r in sorted(enumerate(records), key=lambda p: sort_key(p[1], p[0]))]
        self.n = len(self.recs)

        self.steps: List[int] = []
        self.actions: List[Optional[str]] = []
        self.reasons: List[Optional[str]] = []
        self.reason_first_sentence: List[Optional[str]] = []
        self.locations: List[Optional[str]] = []
        self.scores: List[Optional[int]] = []
        self.max_scores: List[Optional[int]] = []
        self.inventory: List[Optional[List[str]]] = []
        self.valid_actions: List[Optional[List[str]]] = []
        self.env_feedback: List[Optional[str]] = []
        self.observations: List[Optional[str]] = []
        self.rewards: List[Optional[float]] = []
        self.dones: List[Optional[bool]] = []

        self.enter_events: Dict[str, List[int]] = defaultdict(list)
        self.leave_events: Dict[str, List[int]] = defaultdict(list)
        self.gain_events: Dict[str, List[int]] = defaultdict(list)
        self.first_gain: Dict[str, int] = {}
        self.last_gain: Dict[str, int] = {}
        self.item_gain_locations: Dict[str, Counter] = defaultdict(Counter)
        self.total_dwell: Dict[str, int] = defaultdict(int)
        self.longest_run: Tuple[str, int, int] = ("", -1, -1)  # (room, start_idx, end_idx)

        # Graph representation (nodes, edges) with discovery steps
        self.dynamic_map: Dict[str, Any] = {}

        # Distances between locations (computed lazily)
        self._distances: Dict[Tuple[str, str], int] = {}

        self._preprocess_records()
        self._build_dynamic_map()
        self._compute_location_runs()
        self._compute_inventory_events()

    # ------------------------------------------------------------------
    # Internal preprocessing helpers

    @staticmethod
    def _parse_action(rec: Dict[str, Any]) -> Optional[str]:
        """Extract the agent action from a record.
        """
        a = rec.get("agent")
        if isinstance(a, dict):
            act = a.get("action")
            if isinstance(act, str) and act.strip():
                return act.strip().lower()
        m = rec.get("model")
        if isinstance(m, dict):
            raw = m.get("raw_text")
            if isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    j = json.loads(raw)
                    act = j.get("action")
                    if isinstance(act, str) and act.strip():
                        return act.strip().lower()
                except Exception:
                    pass
        return None

    @staticmethod
    def _parse_reason(rec: Dict[str, Any]) -> Optional[str]:
        """Extract the reasoning sentences from a record.
        """
        a = rec.get("agent")
        if isinstance(a, dict):
            r = a.get("reason")
            if isinstance(r, str) and r.strip():
                return r.strip()
        m = rec.get("model")
        if isinstance(m, dict):
            raw = m.get("raw_text")
            if isinstance(raw, str) and raw.strip().startswith("{"):
                try:
                    j = json.loads(raw)
                    r = j.get("reason")
                    if isinstance(r, str) and r.strip():
                        return r.strip()
                except Exception:
                    pass
        # Fallback: use observation text if available
        obs = rec.get("observation")
        if isinstance(obs, str) and obs.strip():
            return obs.strip()
        return None

    @staticmethod
    def _first_sentence(text: Optional[str]) -> Optional[str]:
        """Return the first sentence of the supplied text.
        """
        if not text:
            return None
        # Split on period or newline; keep punctuation with the sentence
        m = re.split(r"(?<=[.!?])\s+|\n+", text.strip(), maxsplit=1)
        if m:
            return m[0].strip()
        return text.strip()

    @staticmethod
    def _parse_location(rec: Dict[str, Any]) -> Optional[str]:
        """Extract the current location from a record.
        """
        g = rec.get("game") or {}
        loc = g.get("location")
        if loc is None:
            return None
        s = str(loc).strip()
        return s if s else None

    @staticmethod
    def _parse_score(rec: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
        """Extract the cumulative score and max score from a record.
        """
        score = None
        max_score = None
        # env_feedback.info.score is more up to date than game.score
        ef = rec.get("env_feedback")
        if isinstance(ef, dict):
            info = ef.get("info")
            if isinstance(info, dict):
                sc = info.get("score")
                if sc is not None:
                    try:
                        score = int(sc)
                    except Exception:
                        pass
        # fallback to game.score
        g = rec.get("game") or {}
        if score is None:
            sc = g.get("score")
            if sc is not None:
                try:
                    score = int(sc)
                except Exception:
                    pass
        ms = g.get("max_score")
        if ms is not None:
            try:
                max_score = int(ms)
            except Exception:
                pass
        return score, max_score

    @staticmethod
    def _parse_inventory(rec: Dict[str, Any]) -> Optional[List[str]]:
        """Extract the player's inventory list from a record.
        """
        g = rec.get("game") or {}
        inv = g.get("inventory")
        if inv is None:
            return None
        if isinstance(inv, list):
            out: List[str] = []
            for item in inv:
                try:
                    s = str(item).strip().lower()
                    if s:
                        out.append(s)
                except Exception:
                    continue
            return out
        return None

    @staticmethod
    def _parse_valid_actions(rec: Dict[str, Any]) -> Optional[List[str]]:
        """Extract the list of valid (admissible) actions from a record.
        """
        g = rec.get("game") or {}
        va = g.get("valid_actions")
        if va is None:
            return None
        out: List[str] = []
        if isinstance(va, list):
            for act in va:
                try:
                    s = str(act).strip().lower()
                    if s:
                        out.append(s)
                except Exception:
                    continue
        return out if out else None

    @staticmethod
    def _parse_env_feedback(rec: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[bool]]:
        """Extract the env_feedback text, reward and done flag from a record."""
        ef = rec.get("env_feedback")
        if not isinstance(ef, dict):
            return None, None, None
        text = ef.get("text")
        if isinstance(text, str):
            text_val = text
        else:
            text_val = None
        reward = ef.get("reward")
        if reward is not None:
            try:
                reward_val = float(reward)
            except Exception:
                reward_val = None
        else:
            reward_val = None
        done = ef.get("done")
        if isinstance(done, bool):
            done_val = done
        else:
            done_val = None
        return text_val, reward_val, done_val

    def _preprocess_records(self) -> None:
        """Populate per-step arrays and initial caches."""
        for idx, rec in enumerate(self.recs):
            # Step index (1-based in the log, 0-based in arrays)
            st = rec.get("step")
            try:
                self.steps.append(int(st))
            except Exception:
                self.steps.append(idx + 1)
            # Action and reasoning
            act = self._parse_action(rec)
            reason = self._parse_reason(rec)
            self.actions.append(act)
            self.reasons.append(reason)
            self.reason_first_sentence.append(self._first_sentence(reason))
            # Location
            loc = self._parse_location(rec)
            self.locations.append(loc)
            # Scores
            score, max_score = self._parse_score(rec)
            self.scores.append(score)
            self.max_scores.append(max_score)
            # Inventory
            inv = self._parse_inventory(rec)
            self.inventory.append(inv)
            # Valid actions
            va = self._parse_valid_actions(rec)
            self.valid_actions.append(va)
            # Env feedback
            ef_text, rew, done = self._parse_env_feedback(rec)
            self.env_feedback.append(ef_text)
            self.rewards.append(rew)
            self.dones.append(done)
            # Observation (text before performing current action).  This is
            # directly ``rec['observation']`` when present.
            obs = rec.get("observation")
            if isinstance(obs, str) and obs.strip():
                self.observations.append(obs)
            else:
                self.observations.append(None)

        # After populating per-step arrays, compute cumulative reward.  Each entry
        # ``self.cum_rewards[i]`` represents the total reward accumulated up to
        # and including step ``i``.  Missing reward values are treated as
        # zero and do not reset the cumulative sum.  
        cum = 0.0
        self.cum_rewards: List[float] = []
        for r in self.rewards:
            if r is not None:
                cum += r
            # Append the running total so far; this always yields a float
            self.cum_rewards.append(cum)

    # ------------------------------------------------------------------
    # Dynamic map builder

    def _build_dynamic_map(self) -> None:
        """Construct a dynamic map capturing when each transition is first observed.
        """
        nodes_seen: Dict[str, int] = {}
        edges_seen: Dict[Tuple[str, str, str], int] = {}

        # Add initial location (if any) as discovered at step 1
        if self.n > 0:
            loc0 = self.locations[0]
            if loc0:
                nodes_seen.setdefault(loc0, self.steps[0] if self.steps else 1)

        for i in range(1, self.n):
            prev = self.recs[i - 1]
            curr = self.recs[i]
            # Determine if still in same episode/session by comparing game name and rom
            g_prev = prev.get("game") or {}
            g_curr = curr.get("game") or {}
            if not (g_prev.get("name") == g_curr.get("name") and g_prev.get("rom") == g_curr.get("rom")):
                # Episode boundary; skip this transition
                continue
            src = self.locations[i - 1]
            dst = self.locations[i]
            act = self.actions[i - 1]
            if src and dst and act:
                # Record node discovery for both endpoints
                nodes_seen.setdefault(src, self.steps[i - 1])
                nodes_seen.setdefault(dst, self.steps[i])
                # Skip self-loops entirely (ignore transitions that do not change location)
                if src == dst:
                    continue
                key = (src, act, dst)
                # Record edge discovery only the first time seen
                if key not in edges_seen:
                    edges_seen[key] = self.steps[i - 1]

        # Build lists of node/edge dicts
        node_list = [
            {"id": name, "first_step": step_num}
            for name, step_num in sorted(nodes_seen.items(), key=lambda x: (x[1], x[0]))
        ]
        edge_list = [
            {"source": s, "action": a, "target": t, "first_step": step_num}
            for (s, a, t), step_num in sorted(edges_seen.items(), key=lambda x: (x[1], x[0][0], x[0][1], x[0][2]))
        ]
        self.dynamic_map = {
            "directed": True,
            "multigraph": True,
            "nodes": node_list,
            "edges": edge_list,
        }

        # Build adjacency for shortest path queries
        self.adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for (s, a, t), step_num in edges_seen.items():
            self.adj[s].append((a, t))

    # ------------------------------------------------------------------
    # Run-length encoding and dwell time computation

    def _compute_location_runs(self) -> None:
        """Compute contiguous runs of locations to determine dwell times.
        """
        current_loc = None
        run_start = 0
        for i, loc in enumerate(self.locations):
            if loc != current_loc:
                # handle end of previous run
                if current_loc is not None:
                    # run from run_start to i-1 inclusive
                    length = i - run_start
                    self.total_dwell[current_loc] += length
                    if length > (self.longest_run[2] - self.longest_run[1] + 1 if self.longest_run[1] >= 0 else 0):
                        self.longest_run = (current_loc, run_start, i - 1)
                    self.leave_events[current_loc].append(i)
                # handle start of new run
                if loc is not None:
                    self.enter_events[loc].append(i)
                current_loc = loc
                run_start = i
        # handle final run
        if current_loc is not None and run_start < self.n:
            length = self.n - run_start
            self.total_dwell[current_loc] += length
            if length > (self.longest_run[2] - self.longest_run[1] + 1 if self.longest_run[1] >= 0 else 0):
                self.longest_run = (current_loc, run_start, self.n - 1)

    # ------------------------------------------------------------------
    # Inventory gain events and counts per location

    def _compute_inventory_events(self) -> None:
        """Compute when items are first gained and counts by location.
        """
        for i in range(self.n):
            # Inventory before step i (i.e. at index i)
            before_inv = set(self.inventory[i]) if (i < self.n and self.inventory[i] is not None) else set()
            # Inventory after step i (i.e. at index i+1) when within bounds
            after_inv = set(self.inventory[i + 1]) if (i + 1 < self.n and self.inventory[i + 1] is not None) else set()
            gained = after_inv - before_inv
            for item in gained:
                # Record that ``item`` was gained at step i
                self.gain_events[item].append(i)
                # Record first and last gain indices
                self.first_gain.setdefault(item, i)
                self.last_gain[item] = i
                # Record location where item was gained (use location at step i)
                loc = self.locations[i]
                if loc is not None:
                    self.item_gain_locations[item][loc] += 1

    # ------------------------------------------------------------------
    # Helper functions for QA answering

    def first_occurrence(self, keyword: str, in_actions: bool = False, in_obs: bool = True) -> Optional[int]:
        """Return the index (0-based) of the first occurrence of ``keyword``.
        """
        kw = keyword.lower()
        for i in range(self.n):
            if in_actions and self.actions[i] and kw in self.actions[i]:
                return i
            if in_obs and self.observations[i] and kw in self.observations[i].lower():
                return i
        return None

    def nth_occurrence(self, keyword: str, k: int, in_actions: bool = False, in_obs: bool = True) -> Optional[int]:
        """Return the index (0-based) of the k-th occurrence of ``keyword``.
        """
        kw = keyword.lower()
        count = 0
        for i in range(self.n):
            hit = False
            if in_actions and self.actions[i] and kw in self.actions[i]:
                hit = True
            if in_obs and self.observations[i] and kw in self.observations[i].lower():
                hit = True
            if hit:
                count += 1
                if count == k:
                    return i
        return None

    def last_occurrence(self, keyword: str, in_actions: bool = False, in_obs: bool = True) -> Optional[int]:
        """Return the index of the last occurrence of ``keyword``."""
        kw = keyword.lower()
        last = None
        for i in range(self.n):
            if in_actions and self.actions[i] and kw in self.actions[i]:
                last = i
            if in_obs and self.observations[i] and kw in self.observations[i].lower():
                last = i
        return last

    def direction_of_action(self, act: Optional[str]) -> Optional[str]:
        """Normalise a command to its pure direction, if applicable.
        """
        if not act:
            return None
        a = act.lower().strip()
        return self.DIRECTION_ALIASES.get(a)

    def compute_shortest_path_length(self, src: str, dst: str) -> Optional[int]:
        """Compute the length of the shortest path between two locations.
        """
        if src == dst:
            return 0
        key = (src, dst)
        if key in self._distances:
            return self._distances[key]
        # BFS search
        visited = set()
        q = deque([(src, 0)])
        visited.add(src)
        while q:
            node, dist = q.popleft()
            for action, nxt in self.adj.get(node, []):
                if nxt == dst:
                    self._distances[key] = dist + 1
                    return dist + 1
                if nxt not in visited:
                    visited.add(nxt)
                    q.append((nxt, dist + 1))
        # No path
        self._distances[key] = None
        return None

    def reachable_within(self, src: str, max_steps: int) -> List[str]:
        """Return the list of distinct locations reachable from ``src`` within ``max_steps`` transitions.
        """
        if max_steps <= 0 or src not in self.adj:
            return []
        visited = set([src])
        reachable = set()
        q = deque([(src, 0)])
        while q:
            node, dist = q.popleft()
            if dist == max_steps:
                continue
            for act, nxt in self.adj.get(node, []):
                if nxt not in visited:
                    visited.add(nxt)
                    reachable.add(nxt)
                    q.append((nxt, dist + 1))
        return list(reachable)

    # ------------------------------------------------------------------
    # Additional helper methods used by QA generation

    def next_score_increase(self, idx: int) -> Optional[int]:
        """Return the number of steps until the next score increase after index ``idx``.
        """
        if idx < 0 or idx >= self.n:
            return None
        base = self.scores[idx]
        if base is None:
            return None
        for j in range(idx + 1, self.n):
            sc = self.scores[j]
            if sc is not None and sc > base:
                return j - idx
        return None

    def next_reward_increase(self, idx: int) -> Optional[int]:
        """Return the number of steps until the cumulative reward increases after index ``idx``.
        """
        if idx < 0 or idx >= self.n:
            return None
        base = self.cum_rewards[idx]
        for j in range(idx + 1, self.n):
            # Since cum_rewards is monotonic, a strictly larger value indicates an increase
            if self.cum_rewards[j] > base:
                return j - idx
        return None

    def longest_contiguous_stay_in_range(self, start: int, end: int) -> Optional[Tuple[str, int, int]]:
        """Compute the longest contiguous run of a location within [start, end].
        """
        if start < 0:
            start = 0
        if end >= self.n:
            end = self.n - 1
        best_loc = None
        best_len = 0
        best_start = None
        cur_loc = None
        cur_start = None
        for i in range(start, end + 1):
            loc = self.locations[i]
            if loc != cur_loc:
                # close previous run
                if cur_loc is not None:
                    run_end = i - 1
                    length = run_end - cur_start + 1
                    if length > best_len:
                        best_len = length
                        best_loc = cur_loc
                        best_start = cur_start
                # start new run
                cur_loc = loc
                cur_start = i
        # handle final run
        if cur_loc is not None and cur_start is not None:
            run_end = end
            length = run_end - cur_start + 1
            if length > best_len:
                best_len = length
                best_loc = cur_loc
                best_start = cur_start
        if best_loc is None:
            return None
        return (best_loc, best_start, best_start + best_len - 1)

    def longest_total_dwell_in_range(self, start: int, end: int) -> Optional[Tuple[str, int]]:
        """Compute which location has the longest total dwell time within [start, end].
        """
        if start < 0:
            start = 0
        if end >= self.n:
            end = self.n - 1
        if start > end:
            return None
        dwell_counts: Dict[str, int] = defaultdict(int)
        cur_loc = None
        cur_start = None
        for i in range(start, end + 1):
            loc = self.locations[i]
            if loc != cur_loc:
                # close previous run
                if cur_loc is not None:
                    run_end = i - 1
                    length = run_end - cur_start + 1
                    dwell_counts[cur_loc] += length
                # start new run
                cur_loc = loc
                cur_start = i
        # final run
        if cur_loc is not None and cur_start is not None:
            length = end - cur_start + 1
            dwell_counts[cur_loc] += length
        if not dwell_counts:
            return None
        # choose location with maximum total dwell time; tie break by earliest appearance
        max_loc = None
        max_len = -1
        for loc, length in dwell_counts.items():
            if length > max_len:
                max_len = length
                max_loc = loc
        return (max_loc, max_len)

    def first_enter_step(self, loc: str) -> Optional[int]:
        """Return the index (0-based) of the first time the player enters ``loc``.
        """
        events = self.enter_events.get(loc)
        if events:
            return events[0]
        return None

    def first_leave_step(self, loc: str) -> Optional[int]:
        """Return the index of the first time the player leaves ``loc``."""
        events = self.leave_events.get(loc)
        if events:
            return events[0]
        return None

    def item_first_gain(self, item: str) -> Optional[int]:
        """Return the index of the first time an item is gained."""
        events = self.gain_events.get(item)
        if events:
            return events[0]
        return None

    def item_nth_gain(self, item: str, k: int) -> Optional[int]:
        """Return the index of the k-th time an item is gained."""
        events = self.gain_events.get(item)
        if events and len(events) >= k:
            return events[k - 1]
        return None

    def item_last_gain(self, item: str) -> Optional[int]:
        """Return the index of the last time an item is gained."""
        events = self.gain_events.get(item)
        if events:
            return events[-1]
        return None

    def has_item_at(self, item: str, idx: int) -> Optional[bool]:
        """Check if item is in inventory at index ``idx``. Returns True/False or None if inventory unknown."""
        if idx < 0 or idx >= self.n:
            return None
        inv = self.inventory[idx]
        if inv is None:
            return None
        return item in inv

    def valid_actions_contain(self, idx: int, pattern: str) -> Optional[bool]:
        """Check if the valid actions list at index ``idx`` contains the given pattern (substring)."""
        if idx < 0 or idx >= self.n:
            return None
        va = self.valid_actions[idx]
        if not va:
            return None
        p = pattern.lower()
        return any(p in a for a in va)

    def keyword_occurrences(self, keyword: str, within_range: Optional[Tuple[int, int]] = None) -> List[int]:
        """Return indices of steps where keyword appears in actions or observations.
        """
        kw = keyword.lower()
        hits: List[int] = []
        start = 0
        end = self.n - 1
        if within_range is not None:
            start, end = within_range
            if start < 0:
                start = 0
            if end >= self.n:
                end = self.n - 1
        for i in range(start, end + 1):
            hit = False
            if self.actions[i] and kw in self.actions[i]:
                hit = True
            obs = self.observations[i]
            if obs and kw in obs.lower():
                hit = True
            if hit:
                hits.append(i)
        return hits

    def keyword_count_in_obs(self, keyword: str, L: int, R: int) -> int:
        """Return the number of steps between L and R where observation contains keyword."""
        if L < 0:
            L = 0
        if R >= self.n:
            R = self.n - 1
        kw = keyword.lower()
        count = 0
        for i in range(L, R + 1):
            obs = self.observations[i]
            if obs and kw in obs.lower():
                count += 1
        return count

    def direction_mode_in_range(self, L: int, R: int) -> Optional[str]:
        """Return the most frequent direction in actions between L and R.
        """
        if L < 0:
            L = 0
        if R >= self.n:
            R = self.n - 1
        counter: Dict[str, int] = defaultdict(int)
        for i in range(L, R + 1):
            act = self.actions[i]
            d = self.direction_of_action(act)
            if d and d in self.PURE_DIRECTIONS:
                counter[d] += 1
        if not counter:
            return None
        # find mode; tie break lexicographically
        max_count = max(counter.values())
        candidates = [d for d, c in counter.items() if c == max_count]
        return sorted(candidates)[0]

    def locations_stats_in_range(self, L: int, R: int) -> Tuple[int, Optional[str]]:
        """Return the number of distinct locations and the most frequent location in [L,R].
        """
        if L < 0:
            L = 0
        if R >= self.n:
            R = self.n - 1
        counts: Dict[str, int] = defaultdict(int)
        order: List[str] = []
        for i in range(L, R + 1):
            loc = self.locations[i]
            if loc is None:
                continue
            counts[loc] += 1
            if loc not in order:
                order.append(loc)
        distinct_count = len(counts)
        mode_loc = None
        max_count = 0
        for loc in order:
            c = counts.get(loc, 0)
            if c > max_count:
                max_count = c
                mode_loc = loc
        return (distinct_count, mode_loc)

    def score_peak_and_net_change(self, L: int, R: int) -> Tuple[Optional[int], Optional[int]]:
        """Return (max_score, net_change) in score between indices L and R (inclusive).
        """
        if L < 0:
            L = 0
        if R >= self.n:
            R = self.n - 1
        max_score = None
        for i in range(L, R + 1):
            sc = self.scores[i]
            if sc is not None:
                if max_score is None or sc > max_score:
                    max_score = sc
        net_change = None
        if L <= R and self.scores[L] is not None and self.scores[R] is not None:
            net_change = self.scores[R] - self.scores[L]
        return (max_score, net_change)

    def region_stay_duration(self, region_kw: str) -> Optional[int]:
        """Return the number of steps from entering any location containing ``region_kw`` to leaving the region.
        """
        kw = region_kw.lower()
        in_region = False
        enter_idx = None
        for i, loc in enumerate(self.locations):
            if loc and kw in loc.lower():
                if not in_region:
                    in_region = True
                    enter_idx = i
            else:
                if in_region:
                    # leaving region
                    if enter_idx is not None:
                        return i - enter_idx
                    in_region = False
        return None

    def item_order(self, item1: str, item2: str) -> Optional[bool]:
        """Return True if item2 was obtained before item1; False if opposite; None if either missing."""
        i1 = self.item_first_gain(item1)
        i2 = self.item_first_gain(item2)
        if i1 is None or i2 is None:
            return None
        return i2 < i1

    def item_before_leave(self, item: str, loc: str) -> Optional[bool]:
        """Return True if first gain of ``item`` occurs before first leave of ``loc``.  None if either missing."""
        gi = self.item_first_gain(item)
        li = self.first_leave_step(loc)
        if gi is None or li is None:
            return None
        return gi < li


    # ------------------------------------------------------------------
    # QA generation

    def generate_qa(self, max_per_type: int = 2) -> List[Dict[str, str]]:
        """Generate a list of question–answer pairs across supported categories.
        """
        qas: List[Dict[str, str]] = []

        # Helper to add tasks up to limit
        def add_some(tasks: List[Tuple[str, str]], cap: int) -> None:
            for q, a in tasks[:cap]:
                qas.append({"question": q, "answer": a})

        # A1: Action + reasoning first sentence at step t
        tasks_a1: List[Tuple[str, str]] = []
        for i in range(self.n):
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            act = self.actions[i] or "unknown"
            reason = self.reason_first_sentence[i] or "unknown"
            q = f"At step {step_num}, what action did the player execute, and what is the first sentence of its reasoning?"
            a = f"Action: {act}; Reasoning: {reason}"
            tasks_a1.append((q, a))
        add_some(tasks_a1, max_per_type)

        # A2: Location at step t
        tasks_a2: List[Tuple[str, str]] = []
        for i in range(self.n):
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            loc = self.locations[i] or "unknown"
            q = f"At step {step_num}, what is the player's location?"
            a = loc
            tasks_a2.append((q, a))
        add_some(tasks_a2, max_per_type)

        # A3: Score at step t
        tasks_a3: List[Tuple[str, str]] = []
        for i in range(self.n):
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            sc = self.scores[i]
            if sc is None:
                continue
            q = f"At step {step_num}, what is the cumulative score?"
            max_sc = self.max_scores[i]
            if max_sc is not None:
                a = f"{sc} (max {max_sc})"
            else:
                a = str(sc)
            tasks_a3.append((q, a))
        add_some(tasks_a3, max_per_type)

        # A7 / F1: Valid actions contain pattern? We choose a few patterns from early steps
        tasks_a7: List[Tuple[str, str]] = []
        candidate_patterns: List[str] = []
        # Extract up to a handful of unique actions from the log
        for act in self.actions:
            if act and act not in candidate_patterns:
                candidate_patterns.append(act)
            if len(candidate_patterns) >= 5:
                break
        for pattern in candidate_patterns:
            # find a step where pattern appears in valid_actions
            for i in range(self.n):
                va = self.valid_actions[i] or []
                step_num = self.steps[i] if i < len(self.steps) else i + 1
                contains = any(pattern in a for a in va)
                answer = "yes" if contains else "no"
                q = f"At step {step_num}, do valid actions include '{pattern}'? (yes/no)"
                tasks_a7.append((q, answer))
        add_some(tasks_a7, max_per_type)

        # New single-hop: Observation before/after performing action at step t
        tasks_obs_single: List[Tuple[str, str]] = []
        for i in range(self.n):
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            # After performing: env_feedback at index i
            after_obs = self.env_feedback[i]
            if after_obs:
                q_after = f"At step {step_num}, what observation did you see after performing the action?"
                a_after = after_obs.strip()
                tasks_obs_single.append((q_after, a_after))
            # Before performing: env_feedback at index i-1
            if i > 0:
                before_obs = self.env_feedback[i - 1]
                if before_obs:
                    q_before = f"At step {step_num}, what observation did you see before performing the action?"
                    a_before = before_obs.strip()
                    tasks_obs_single.append((q_before, a_before))
        add_some(tasks_obs_single, max_per_type)

        # Multi-hop new task: first/n-th/last time performing some action: what observation before/after?
        tasks_obs_multi: List[Tuple[str, str]] = []
        # Build action occurrences map
        action_indices: Dict[str, List[int]] = defaultdict(list)
        for i, act in enumerate(self.actions):
            if act:
                action_indices[act].append(i)
        # For each action with at least one occurrence, create before/after tasks for first and last occurrence
        for act, idxs in list(action_indices.items())[:3]:  # limit number of actions
            # first
            first_idx = idxs[0]
            last_idx = idxs[-1]
            # Before first
            if first_idx > 0:
                before = self.env_feedback[first_idx - 1]
                if before:
                    q = f"Before the first time performing '{act}', what observation did you see?"
                    a = before.strip()
                    tasks_obs_multi.append((q, a))
            # After first
            after = self.env_feedback[first_idx]
            if after:
                q = f"After the first time performing '{act}', what observation did you see?"
                a = after.strip()
                tasks_obs_multi.append((q, a))
            # Before last
            if last_idx > 0:
                before = self.env_feedback[last_idx - 1]
                if before:
                    q = f"Before the last time performing '{act}', what observation did you see?"
                    a = before.strip()
                    tasks_obs_multi.append((q, a))
            # After last
            after = self.env_feedback[last_idx]
            if after:
                q = f"After the last time performing '{act}', what observation did you see?"
                a = after.strip()
                tasks_obs_multi.append((q, a))
        add_some(tasks_obs_multi, max_per_type)

        # F3: Inventory contains item?  Choose a few common items and query
        tasks_f3: List[Tuple[str, str]] = []
        # Determine candidate items by frequency of gain events
        item_counts = sorted(
            ((item, len(idxs)) for item, idxs in self.gain_events.items()),
            key=lambda x: -x[1]
        )
        for item, _ in item_counts[:3]:
            # pick the first occurrence of gain event as the query step
            idxs = self.gain_events[item]
            if not idxs:
                continue
            i = idxs[0]
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            has_item = self.inventory[i] and (item in (self.inventory[i] or []))
            answer = "yes" if has_item else "no"
            q = f"At step {step_num}, did you have '{item}' in inventory? (yes/no)"
            tasks_f3.append((q, answer))
        add_some(tasks_f3, max_per_type)

        # F4: List inventory at a given step
        tasks_f4: List[Tuple[str, str]] = []
        for i in range(self.n):
            inv = self.inventory[i]
            if inv:
                step_num = self.steps[i] if i < len(self.steps) else i + 1
                inv_list = ", ".join(inv)
                q = f"At step {step_num}, what are all the items you carry?"
                a = inv_list
                tasks_f4.append((q, a))
        add_some(tasks_f4, max_per_type)

        # F5: Step with maximum inventory size
        tasks_f5: List[Tuple[str, str]] = []
        max_inv_count = -1
        max_inv_step = None
        for i, inv in enumerate(self.inventory):
            count = len(inv) if inv else 0
            if count > max_inv_count:
                max_inv_count = count
                max_inv_step = i
        if max_inv_step is not None:
            step_num = self.steps[max_inv_step] if max_inv_step < len(self.steps) else max_inv_step + 1
            q = "At which step does the inventory contain the most items?"
            a = str(step_num)
            tasks_f5.append((q, a))
        add_some(tasks_f5, 1)

        # F6: Location with most item pickups
        tasks_f6: List[Tuple[str, str]] = []
        # Sum item gains per location across all items
        location_gain_counts: Counter = Counter()
        for item, loc_counter in self.item_gain_locations.items():
            for loc, c in loc_counter.items():
                location_gain_counts[loc] += c
        if location_gain_counts:
            # Pick the location with maximum gain count
            loc, _ = location_gain_counts.most_common(1)[0]
            q = "At which location did you pick up the largest number of new items?"
            a = loc
            tasks_f6.append((q, a))
        add_some(tasks_f6, 1)

        # D3: Number of distinct pure directions available at a step
        tasks_d3: List[Tuple[str, str]] = []
        for i in range(self.n):
            loc = self.locations[i]
            if not loc:
                continue
            # Determine directions available from current location
            dirs = set()
            for action, tgt in self.adj.get(loc, []):
                d = self.direction_of_action(action)
                if d and d in self.PURE_DIRECTIONS:
                    dirs.add(d)
            count = len(dirs)
            step_num = self.steps[i] if i < len(self.steps) else i + 1
            q = f"At step {step_num}, how many distinct directions can you move from the current location from your knowledge?"
            a = str(count)
            tasks_d3.append((q, a))
        add_some(tasks_d3, max_per_type)

        return qas



# -------------------------
# Negative sampling from pooled items/locations with clustering
# Pool file: game_envs/items_locations.json
# -------------------------

_POOL_CACHE: Optional[Dict[str, Any]] = None


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _load_items_locations_pool(pool_path: str = "game_envs/items_locations.json") -> Dict[str, Any]:
    global _POOL_CACHE
    if _POOL_CACHE is not None:
        return _POOL_CACHE
    if not os.path.isfile(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")
    with open(pool_path, "r", encoding="utf-8") as f:
        _POOL_CACHE = json.load(f)
    return _POOL_CACHE


def pick_negative_from_pool(
    anchor: str,
    pool_type: str,                 # "item" or "location"
    seen_terms: Iterable[str],      # terms that appeared in the current log, any casing
    pool_path: str = "game_envs/items_locations.json",
    rng: Optional[random.Random] = None,
) -> str:
    """
    Pick a negative term:
    - Prefer: same cluster as anchor, but NOT in seen_terms.
    - Fallback: any cluster term NOT in seen_terms.
    - Fallback: any term in pool (even if seen), if pool is exhausted.

    If anchor is not found in term2cluster, we skip the 'same cluster' preference.
    """
    rng = rng or random
    pool = _load_items_locations_pool(pool_path)

    if pool_type not in pool:
        # Robust fallback: pick from any available type
        pool_type = "item" if "item" in pool else list(pool.keys())[0]

    entry = pool[pool_type]
    clusters: Dict[str, list] = entry.get("clusters", {}) or {}
    term2cluster: Dict[str, str] = entry.get("term2cluster", {}) or {}
    term2cluster_norm = {_norm_key(k): str(v) for k, v in term2cluster.items()}

    # Build a normalized seen set
    seen_set: Set[str] = set(_norm_key(x) for x in (seen_terms or []))

    def pick_from_cluster(cid: str) -> Optional[str]:
        arr = clusters.get(cid, [])
        if not arr:
            return None
        cands = [t for t in arr if _norm_key(t) not in seen_set]
        if cands:
            return rng.choice(cands)
        return None

    anchor_k = _norm_key(anchor)
    cid = term2cluster_norm.get(anchor_k)

    # 1) Same-cluster preference
    if cid is not None:
        x = pick_from_cluster(str(cid))
        if x is not None:
            return x

    # 2) Any cluster, try to find something not seen
    cluster_ids = list(clusters.keys())
    rng.shuffle(cluster_ids)
    for c in cluster_ids:
        x = pick_from_cluster(c)
        if x is not None:
            return x

    # 3) Pool exhausted w.r.t seen_terms: return any term at all
    all_terms = []
    for arr in clusters.values():
        all_terms.extend(arr)
    if all_terms:
        return rng.choice(all_terms)

    # Final fallback
    return ""
