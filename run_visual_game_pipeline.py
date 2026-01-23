#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end Crafter QA pipeline.

This script runs the full pipeline for one or more Crafter seeds:

  1. Play Crafter with a vision LLM policy, producing logs + frames + map.
  2. Generate QA for each specified difficulty level from the logs.
  3. Answer and evaluate all QA using the answer_crafter_qa.py script.

It assumes you have the following scripts in the same project root:

  - run_crafter_openai.py
  - generate_crafter_qa.py
  - answer_crafter_qa.py

and that they use the directory layout discussed in our modifications:

  log/seed{SEED}/{RUN_NAME}/logs.jsonl
  log/seed{SEED}/{RUN_NAME}/map_seed{SEED}.txt
  log/seed{SEED}/{RUN_NAME}/frames/*.png

  generated_qa/seed{SEED}/{RUN_NAME}/DIF_{DIFFICULTY}/qa.jsonl
  generated_qa/seed{SEED}/{RUN_NAME}/qa_context.json

  eval/seed{SEED}/{RUN_NAME}/...

You normally run this script from the project root (the directory that
contains these scripts plus the log/generated_qa/eval folders).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> None:
    """Run a subprocess and raise if it fails."""
    cmd_str = " ".join(cmd)
    print(f"\n[CMD] {cmd_str}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None)
    if result.returncode != 0:
        raise SystemExit(f"[ERROR] command failed with code {result.returncode}: {cmd_str}")


def find_latest_run_dir(log_root: Path, seed: int) -> Path:
    """
    Find the most recently modified run directory for a given seed.

    We expect logs to be in:

        log/seed{SEED}/{RUN_NAME}/logs.jsonl
    """
    seed_dir = log_root / f"seed{seed}"
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"[ERROR] seed log directory not found: {seed_dir}")

    candidates = [p for p in seed_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"[ERROR] no run directories found under {seed_dir}")

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"[INFO] Latest run for seed{seed}: {latest.name} ({latest})")
    return latest


def parse_seeds(seeds_arg: Iterable[str]) -> List[int]:
    """
    Parse seeds from CLI.

    Supports:
      --seeds 1 2 3
      --seeds 1,2,3
    """
    out: List[int] = []
    for token in seeds_arg:
        for part in str(token).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(part))
            except ValueError:
                raise SystemExit(f"[ERROR] invalid seed value: {part!r}")
    if not out:
        raise SystemExit("[ERROR] no seeds given after parsing")
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end Crafter pipeline: play -> generate QA -> answer & eval."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seeds to run, e.g. '--seeds 100 123' or '--seeds 100,123'.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Number of environment steps per seed (passed to run_crafter_openai.py).",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=10,
        help="How many past turns to include in the policy prompt (passed to run_crafter_openai.py).",
    )
    parser.add_argument(
        "--difficulties",
        type=int,
        nargs="+",
        default=[-1, 50],
        help="Difficulties to generate QA for (same semantics as --difficulty in generate_crafter_qa.py).",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (where the scripts and log/generated_qa/eval folders live).",
    )
    parser.add_argument(
        "--run-script",
        type=str,
        default="run_crafter_openai.py",
        help="Filename of the Crafter playing script.",
    )
    parser.add_argument(
        "--gen-script",
        type=str,
        default="generate_crafter_qa.py",
        help="Filename of the QA generation script.",
    )
    parser.add_argument(
        "--answer-script",
        type=str,
        default="answer_crafter_qa.py",
        help="Filename of the QA answering/eval script.",
    )
    # Answering / eval parameters (forwarded to answer_crafter_qa.py)
    parser.add_argument(
        "--qa-model",
        type=str,
        default=None,
        help=(
            "Model name override for answering (passed as --model to answer_crafter_qa.py; "
            "if omitted, that script uses its own default/OPENAI_MODEL)."
        ),
    )
    parser.add_argument(
        "--qa-temperature",
        type=float,
        default=0.0,
        help="Temperature when answering QA (passed to answer_crafter_qa.py).",
    )
    parser.add_argument(
        "--qa-max-tokens",
        type=int,
        default=4096,
        help="Max tokens when answering QA (passed to answer_crafter_qa.py).",
    )
    parser.add_argument(
        "--qa-source",
        type=str,
        choices=["question", "paraphrase"],
        default="paraphrase",
        help="Use 'question' or 'paraphrase' as the prompt text (passed to answer_crafter_qa.py).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for answering QA (passed to answer_crafter_qa.py).",
    )
    parser.add_argument(
        "--frames-mode",
        type=str,
        choices=["mosaic", "all", "sample"],
        default="mosaic",
        help="How to include frames in prompts (passed to answer_crafter_qa.py).",
    )

    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)

    root = Path(args.project_root).expanduser().resolve()
    print(f"[INFO] Project root: {root}")

    run_script = (root / args.run_script).resolve()
    gen_script = (root / args.gen_script).resolve()
    answer_script = (root / args.answer_script).resolve()

    if not run_script.is_file():
        raise SystemExit(f"[ERROR] run script not found: {run_script}")
    if not gen_script.is_file():
        raise SystemExit(f"[ERROR] QA generation script not found: {gen_script}")
    if not answer_script.is_file():
        raise SystemExit(f"[ERROR] QA answering script not found: {answer_script}")

    log_root = root / "log"
    qa_root = root / "generated_qa"

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"[SEED] {seed}")
        print("=" * 80)

        # ------------------------------------------------------------------
        # 1) Play Crafter (vision agent) to produce logs / map / frames
        # ------------------------------------------------------------------
        play_cmd = [
            sys.executable,
            str(run_script),
            "--steps",
            str(args.steps),
            "--seed",
            str(seed),
            "--history-turns",
            str(args.history_turns),
        ]

        run_cmd(play_cmd, cwd=root)

        # Find the newest run directory for this seed under log/seed{seed}
        run_dir = find_latest_run_dir(log_root, seed)
        run_name = run_dir.name

        log_file = run_dir / "logs.jsonl"
        map_file = run_dir / f"map_seed{seed}.txt"

        if not log_file.is_file():
            raise SystemExit(f"[ERROR] log file not found: {log_file}")
        if not map_file.is_file():
            raise SystemExit(f"[ERROR] map file not found: {map_file}")

        print(f"[INFO] Using log file: {log_file}")
        print(f"[INFO] Using map file: {map_file}")

        # ------------------------------------------------------------------
        # 2) Generate QA for each difficulty
        # ------------------------------------------------------------------
        qa_run_root = qa_root / f"seed{seed}" / run_name

        for diff in args.difficulties:
            dif_dir = qa_run_root / f"DIF_{diff}"
            dif_dir.parent.mkdir(parents=True, exist_ok=True)

            gen_cmd = [
                sys.executable,
                str(gen_script),
                "--log-file",
                str(log_file),
                "--map-file",
                str(map_file),
                "--output-dir",
                str(dif_dir),
                "--difficulty",
                str(diff),
            ]
            # view window: rely on defaults in generate_crafter_qa.py unless you want to override here.
            run_cmd(gen_cmd, cwd=root)
            print(f"[INFO] Generated QA for difficulty {diff} in {dif_dir}")

        # ------------------------------------------------------------------
        # 3) Answer & evaluate all DIF_* folders under this run
        # ------------------------------------------------------------------
        answer_cmd: List[str] = [
            sys.executable,
            str(answer_script),
            "--run-folder",
            str(qa_run_root),
            "--source",
            args.qa_source,
            "--temperature",
            str(args.qa_temperature),
            "--max-tokens",
            str(args.qa_max_tokens),
            "--batch-size",
            str(args.batch_size),
            "--frames-mode",
            args.frames_mode,
        ]
        if args.qa_model:
            answer_cmd.extend(["--model", args.qa_model])

        run_cmd(answer_cmd, cwd=root)
        print(f"[INFO] Finished answering/evaluating for seed {seed} (run_name={run_name})")

    print("\n[DONE] All seeds finished.")


if __name__ == "__main__":
    main()
