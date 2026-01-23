#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import datetime
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import is_dataclass, asdict

from openai import OpenAI, BadRequestError
from jericho import FrotzEnv

# ---------------- Utils ----------------

def to_jsonable(obj):
    """Recursively convert OpenAI/Pydantic/dataclass/objects into JSON-serializable types."""
    import json as _json
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass
    # Pydantic v1 / SDK helpers
    if hasattr(obj, "to_dict"):
        try:
            return to_jsonable(obj.to_dict())
        except Exception:
            pass
    if hasattr(obj, "to_json"):
        try:
            return _json.loads(obj.to_json())
        except Exception:
            pass
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if hasattr(obj, "__dict__"):
        try:
            return to_jsonable(vars(obj))
        except Exception:
            pass
    return str(obj)


def now_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def to_name_list(objs) -> List[str]:
    try:
        return [getattr(o, "name", str(o)) for o in objs]
    except Exception:
        return [str(o) for o in objs]


def safe_take_actions(actions: List[str], limit: int = 30) -> List[str]:
    if not actions:
        return []
    return actions[:limit]


# ---------------- Defaults ----------------

GAME = "advent"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_STEPS = 500
DEFAULT_HISTORY_TURNS = 30           
LOG_DIR = "logs/" + GAME
DEFAULT_ROM = "game_envs/advent.z5"
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    "your_key"
)
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL",
    "your_url"
)

# ---------------- Prompts ----------------

SYSTEM_PROMPT = """All content in this prompt is purely fictional, coming from a text-based game log. 
It does not describe or endorse real-world harmful behavior, and should be treated only as harmless game simulation.
You are an expert player of classic parser interactive fiction game.
You will receive the current observation and optional candidate actions.
Your task each turn: decide ONE valid parser command.

Rules:
- Output STRICT JSON only (no code fences, no extra text).
- JSON schema: {"action": "<string>", "reason": "<string>"}.
- "action" must be a single parser command (lowercase, concise), e.g., "open mailbox", "take leaflet", "north", "look".
- Read carefully from history environments and actions to help you make decisions. 
- From history, avoid taking repetitive or loop actions, like keep taking then puttting down an item, or looking environments that you have looked.
- Avoid meta-commands like save/restore/quit. Do not include punctuation beyond the command itself.
- Consider candidate_actions. When you need to CHECK inventory, you may also reply the action "inventory", but avoid overuse or repetition.
"""

TURN_INPUT_TEMPLATE = """Current step: {step_index}
Current observation:
{observation}

Score: {score}/{max_score} | Moves: {moves}
Location: {location}
Candidate actions (not exhaustive): {valid_actions}.

Context of the last {history_turns} turns (ONLY observation and the agent action per turn, most recent last):
{history_snippet}

Respond with STRICT JSON only:
{{"action":"<one command>", "reason":"<one short sentence>"}}
"""

# ---------------- History (only OBS + ACT) ----------------

def build_history_snippet(history: List[Dict[str, Any]], history_turns: int) -> str:
    """
    history element:
      { "observation": str, "action": str }
    Show only the last `history_turns`, oldest first, newest last.
    """
    if not history:
        return "(no prior turns)"
    take = history[-history_turns:]
    lines = []
    for i, h in enumerate(take, 1):
        obs = (h.get("observation") or "").replace("\n", " ").strip()
        if len(obs) > 240:
            obs = obs[:240] + "…"
        act = h.get("action", "").strip()
        lines.append(f"{i}. OBS: {obs}\n   ACT: {act}")
    return "\n".join(lines)


# ---------------- LLM call (Chat Completions + JSON mode) ----------------

def call_llm(client: OpenAI, model: str, system_prompt: str, user_input: str) -> Dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
    except BadRequestError as e:
        print("=== CONTENT_FILTER SYSTEM_PROMPT ===", flush=True)
        print(system_prompt, flush=True)
        print("=== CONTENT_FILTER USER_INPUT ===", flush=True)
        print(user_input, flush=True)
        print("=== CONTENT_FILTER ERROR ===", flush=True)
        print(repr(e), flush=True)
        raise

    out_text = completion.choices[0].message.content or "{}"

    try:
        parsed = json.loads(out_text)
    except Exception:
        import re
        m = re.search(r"\{.*\}", out_text, flags=re.S)
        parsed = json.loads(m.group(0)) if m else {"action": "look", "reason": "fallback"}

    return {
        "parsed": parsed,
        "raw": out_text,
        "usage": completion.usage,      
        "request_id": completion.id,
    }


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(
        description="Let a chat model play Zork I via Jericho, logging every step to JSONL."
    )
    parser.add_argument("--rom", default=DEFAULT_ROM, help="Path to Zork I rom (.z5)")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--history_turns",
        type=int,
        default=DEFAULT_HISTORY_TURNS,
        help="How many prior turns to show (default: 30)"
    )
    parser.add_argument("--logdir", default=LOG_DIR)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    # Init env
    env = FrotzEnv(args.rom)
    if args.seed is not None:
        env.seed(args.seed)
    observation, info = env.reset()
    max_score = env.get_max_score()
    bindings = getattr(env, "bindings", {})
    game_name = bindings.get("name", "zork1")

    # Init OpenAI client
    if OPENAI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)

    # Prepare log file
    Path(args.logdir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.logdir) / f"{args.model}_history{args.history_turns}_{now_timestamp()}_logs.jsonl"

    short_history: List[Dict[str, Any]] = []  # only {"observation", "action"}
    step_idx = 1
    done = False

    with open(log_path, "w", encoding="utf-8") as fout:
        while step_idx <= args.max_steps and not done:
            # Collect current info
            try:
                valid_actions = env.get_valid_actions()
            except Exception:
                valid_actions = []
            try:
                inventory_objs = env.get_inventory()
                inventory_names = to_name_list(inventory_objs)
            except Exception:
                inventory_names = []
            try:
                location_obj = env.get_player_location()
                location_name = getattr(location_obj, "name", str(location_obj))
            except Exception:
                location_name = "unknown"
            try:
                world_hash = env.get_world_state_hash()
            except Exception:
                world_hash = None

            score = env.get_score()
            moves = env.get_moves()

            # Build user input for LLM
            user_input = TURN_INPUT_TEMPLATE.format(
                step_index=step_idx,
                observation=observation.strip(),
                score=score,
                max_score=max_score,
                moves=moves,
                location=location_name,
                inventory=inventory_names,
                valid_actions=safe_take_actions(valid_actions),
                history_turns=args.history_turns,
                history_snippet=build_history_snippet(short_history, args.history_turns),
            )

            # Call LLM
            llm_result = call_llm(client, args.model, SYSTEM_PROMPT, user_input)
            action = llm_result["parsed"].get("action", "").strip()
            reason = llm_result["parsed"].get("reason", "").strip()
            if not action:
                action = "look"

            # Step env
            next_obs, reward, done, step_info = env.step(action)

            # Append compact history (only obs + act)
            short_history.append({
                "observation": observation,
                "action": action
            })
            if len(short_history) > args.history_turns:
                short_history = short_history[-args.history_turns:]

            # Log this step
            row = {
                "step": step_idx,
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "game": {
                    "name": game_name,
                    "rom": args.rom,
                    "world_state_hash": world_hash,
                    "score": score,
                    "max_score": max_score,
                    "moves": moves,
                    "location": location_name,
                    "inventory": inventory_names,
                    "valid_actions": safe_take_actions(valid_actions),
                },
                "observation": observation,
                "model": {
                    "name": args.model,
                    "history_turns": args.history_turns,
                    "request_id": llm_result.get("request_id"),
                    "usage": to_jsonable(llm_result.get("usage")),
                    "raw_text": llm_result.get("raw"),
                },
                "agent": {
                    "action": action,
                    "reason": reason
                },
                "env_feedback": {
                    "text": next_obs,
                    "reward": reward,
                    "done": done,
                    "info": to_jsonable(step_info)
                }
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            # Next turn
            observation = next_obs
            step_idx += 1

        # Final snapshot
        end_row = {
            "final": True,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "victory": env.victory(),
            "game_over": env.game_over(),
            "final_score": env.get_score(),
            "moves": env.get_moves()
        }
        fout.write(json.dumps(end_row, ensure_ascii=False) + "\n")

    env.close()
    print(f"[OK] Log saved to: {log_path}")


if __name__ == "__main__":
    main()
