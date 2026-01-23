#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm


# ---- List of Jericho games ----
'''
JERICHO_GAMES: List[str] = [
    "advent",
    "awaken",
    "balances",
    "dragon",
    "gold",
    "jewel",
    "karn",
    "ludicorp",
    "moonlit",
    "pentari",
    "reverb",
    "sorcerer",
    "zork1",
    "zork2",
    "zork3",
]
'''
JERICHO_GAMES: List[str] = [
    "zork3",
]


# ---- Map game name -> ROM path ----
ROM_MAP: Dict[str, str] = {
    "advent": "game_envs/advent.z5",
    "awaken": "game_envs/awaken.z5",
    "balances": "game_envs/balances.z5",
    "dragon": "game_envs/dragon.z5",
    "gold": "game_envs/gold.z5",
    "jewel": "game_envs/jewel.z5",
    "karn": "game_envs/karn.z5",
    "ludicorp": "game_envs/ludicorp.z5",
    "moonlit": "game_envs/moonlit.z5",
    "pentari": "game_envs/pentari.z5",
    "reverb": "game_envs/reverb.z5",
    "sorcerer": "game_envs/sorcerer.z3",
    "zork1": "game_envs/zork1.z5",
    "zork2": "game_envs/zork2.z5",
    "zork3": "game_envs/zork3.z5",
}


def run_cmd(cmd: List[str]) -> None:
    """Run a subprocess command with basic logging."""
    print("\n[CMD]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def find_latest_log(log_dir: Path) -> Path:
    """Return the newest *_logs.jsonl file in a directory."""
    candidates = list(log_dir.glob("*_logs.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No *_logs.jsonl found in {log_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Run Jericho full pipeline (play -> QA gen -> answer) for all games."
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="Model used both to play the game and to answer QA (e.g. gpt-5.1 / gpt-4.1).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per Jericho run.",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=30,
        help="Number of history turns passed to the model when playing.",
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory for Jericho logs.",
    )
    parser.add_argument(
        "--qa-root",
        default="generated_qa",
        help="Root directory for generated QA.",
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        type=int,
        default=[-1, 50],
        help="Difficulty values to generate QA for (e.g. -1 50).",
    )
    parser.add_argument(
        "--max-per-type",
        type=int,
        default=2,
        help="max-per-type passed to generate_jericho_qa.py.",
    )

    args = parser.parse_args()

    logs_root = Path(args.logs_root)
    qa_root = Path(args.qa_root)

    # Ensure roots exist
    logs_root.mkdir(parents=True, exist_ok=True)
    qa_root.mkdir(parents=True, exist_ok=True)

    print("Games to run:", ", ".join(JERICHO_GAMES))
    print(f"Model for play & answer: {args.model}")
    print(f"Logs root: {logs_root.resolve()}")
    print(f"QA root:   {qa_root.resolve()}")

    # Progress bar over games
    for game in tqdm(JERICHO_GAMES, desc="Jericho games"):
        print("\n" + "=" * 80)
        print(f"== Game: {game} ==")

        if game not in ROM_MAP:
            raise KeyError(f"No ROM path configured for game '{game}' in ROM_MAP.")
        rom_path = ROM_MAP[game]

        # ----------------------
        # 1) Run the game
        # ----------------------
        game_logs_dir = logs_root / game
        game_logs_dir.mkdir(parents=True, exist_ok=True)

        run_cmd(
            [
                "python",
                "run_jericho_openai.py",
                "--rom",
                rom_path,
                "--model",
                args.model,
                "--max_steps",
                str(args.max_steps),
                "--history_turns",
                str(args.history_turns),
                "--logdir",
                str(game_logs_dir),
            ]
        )

        latest_log = find_latest_log(game_logs_dir)
        # strip trailing "_logs.jsonl"
        run_name = latest_log.name.replace("_logs.jsonl", "")
        print(f"[INFO] Latest log for {game}: {latest_log}")
        print(f"[INFO] Run name (default-folder): {run_name}")

        # ----------------------
        # 2) Generate QA for multiple difficulties
        # ----------------------
        for dif in args.difficulties:
            print(f"\n[INFO] Generating QA for {game}, difficulty={dif}")
            run_cmd(
                [
                    "python",
                    "generate_jericho_qa.py",
                    "--input-dir",
                    str(game_logs_dir),
                    "--default-folder",
                    run_name,
                    "--game",
                    game,
                    "--output-dir",
                    str(qa_root),
                    "--max-per-type",
                    str(args.max_per_type),
                    "--difficulty",
                    str(dif),
                    "--paraphrase",
                    "True",
                ]
            )

        # ----------------------
        # 3) Answer QA for this run (all DIF_* under the run folder)
        # ----------------------
        run_folder = qa_root / game / run_name
        print(f"\n[INFO] Answering QA for {game} in run-folder: {run_folder}")

        run_cmd(
            [
                "python",
                "answer_jericho_qa.py",
                "--run-folder",
                str(run_folder),
                "--model",
                args.model,
                "--temperature",
                "0.0",
                "--max-tokens",
                "1024",
                "--source",
                "paraphrase",
                "--batch-size",
                "8",
            ]
        )

    print("\nAll games finished.")


if __name__ == "__main__":
    main()
