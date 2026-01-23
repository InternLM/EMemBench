#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Reusable DocVQA-style eval logic
from eval_score import eval_score, eval_acc_and_f1, infer_answer_type, canonicalize_not_answerable

# ===================== Constants and defaults =====================
DEFAULT_FOLDER = ""

# ===================== I/O helpers =====================
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows

def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# ===================== OpenAI client helper =====================
def openai_client(api_key: str, base_url: str):
    """
    Construct an OpenAI client. This wrapper defers the import so
    that the dependency is optional until runtime.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required. Try: pip install openai") from e
    return OpenAI(api_key=api_key, base_url=base_url)

# ===================== System prompt and icons =====================
def load_icons_data_url(icons_path: str) -> str:
    """Read an image file and return a base64 data URL. Returns empty string if fail."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        Image = None  # type: ignore
    if not icons_path or Image is None:
        return ""
    try:
        with Image.open(icons_path) as im:
            import io
            import base64
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

def build_instruction_prelude(icons_path: str) -> List[Dict[str, Any]]:
    prelude: List[Dict[str, Any]] = []
    sys_text = "You answer Jericho QA concisely. Reply with JSON only."
    prelude.append({"role": "system", "content": sys_text})
    data_url = load_icons_data_url(icons_path)
    if data_url:
        prelude.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "ICON LEGEND for Jericho entities (reference image)."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        })
    return prelude

# ===================== Context and messages =====================
def get_steps_array(qa_context: Any) -> List[Dict[str, Any]]:
    if isinstance(qa_context, list):
        return qa_context
    if isinstance(qa_context, dict):
        return qa_context.get("steps", []) or []
    return []

def build_context_timeline(qa_context: Any) -> str:
    steps = get_steps_array(qa_context)
    lines: List[str] = []
    for s in steps:
        t = s.get("t")
        fields: List[str] = []
        action = s.get("action", "")
        reason_raw = s.get("reason", "") or ""
        reason = " ".join(str(reason_raw).split()).strip().lower()
        fields.append(f"action={action}")
        fields.append(f"reason={reason}")
        for key, value in s.items():
            if key in {"t", "action", "reason"}:
                continue
            if isinstance(value, list):
                val_str = ", ".join(str(v) for v in value)
                formatted = f"[{val_str}]"
            else:
                if isinstance(value, str):
                    val_str = " ".join(value.split())
                else:
                    val_str = str(value)
                formatted = val_str
            fields.append(f"{key}={formatted}")
        line = f"step {t}: " + "; ".join(fields)
        lines.append(line)
    return "\n".join(lines)

USER_BATCH_HEADER = (
    "You will answer multiple questions about the same episode.\n"
    "Return strictly JSON.\n"
    "For a single question: {\"answer\": \"<short>\", \"explanation\": \"<brief>\"}\n"
    "For batched questions: an array of objects, each: {\"id\": \"<qid>\", \"answer\": \"<short>\", \"explanation\": \"<brief>\"}\n"
    "Keep explanation short.\n"
    "If your answer is a step number, answer only the number.\n"
    "For any question you find not answerable, strictly reply 'Not answerable' as answer, and give explanation.\n"
    "Questions:\n"
)

def build_batch_messages(batch_items: List[Tuple[str, str]], prelude: List[Dict[str, Any]], timeline_text: str) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    msgs.extend(prelude)
    lines = [USER_BATCH_HEADER]
    for qid, qtext in batch_items:
        lines.append(f"- id={qid} :: {qtext}")
    if timeline_text:
        lines.append("\nContext timeline:\n" + timeline_text)
    content = [{"type": "text", "text": "\n".join(lines)}]
    msgs.append({"role": "user", "content": content})
    return msgs

def chat_once(client, model: str, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def parse_batch_json(s: str) -> List[Dict[str, str]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            out = []
            for it in obj:
                if not isinstance(it, dict):
                    continue
                out.append({
                    "id": str(it.get("id", "")).strip(),
                    "answer": str(it.get("answer", "")).strip(),
                    "explanation": str(it.get("explanation", "")).strip(),
                })
            return out
    except Exception:
        pass
    return []

# ===================== Grouping metrics =====================
from collections import defaultdict

def _group_metrics(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, float]]:
    buckets = defaultdict(list)
    for r in rows:
        k = r.get(key_name)
        if key_name == "difficulty":
            k = "null" if k is None else str(k)
        k = str(k)
        buckets[k].append(r)
    out: Dict[str, Dict[str, float]] = {}
    for k, sub in buckets.items():
        acc, f1 = eval_acc_and_f1(sub)
        out[k] = {"count": len(sub), "acc": acc, "f1": f1}
    return out

# ===================== Directory discovery =====================
def discover_dif_dirs(run_folder: Path) -> List[Path]:
    if run_folder.name.startswith("DIF_"):
        return [run_folder]
    difs = [p for p in run_folder.iterdir() if p.is_dir() and p.name.startswith("DIF_")]
    if difs:
        return sorted(difs, key=lambda p: (len(p.name), p.name))
    if (run_folder / "qa.jsonl").is_file():
        return [run_folder]
    raise FileNotFoundError(f"No DIF_* folders or qa.jsonl found under: {run_folder}")

# ===================== Evaluation root for Jericho =====================
def compute_eval_root_from_run(run_folder: Path) -> Path:
    if run_folder.name.startswith("DIF_"):
        actual_run = run_folder.parent
    else:
        actual_run = run_folder
    run_name = actual_run.name
    game_folder = actual_run.parent
    game_name = game_folder.name
    cur = actual_run.resolve()
    jericho_root = None
    for _ in range(10):
        if cur.name == "jericho":
            jericho_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if jericho_root is None:
        jericho_root = game_folder.parent.parent if game_folder.parent else actual_run.parent
    return jericho_root / "eval" / game_name / run_name

# ===================== Evaluation per DIF folder =====================
def evaluate_dif_folder(
    client,
    dif_dir: Path,
    eval_dif_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    source: str,
    prelude: List[Dict[str, Any]],
    batch_size: int,
):
    qa_path = dif_dir / "qa.jsonl"
    ctx_path = dif_dir / "qa_context.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing qa.jsonl in {dif_dir}")
    if not ctx_path.is_file():
        raise FileNotFoundError(f"Missing qa_context.json in {dif_dir}")

    qa_rows = load_jsonl(qa_path)
    qa_context = load_json(ctx_path)
    timeline_text = build_context_timeline(qa_context)

    items: List[Dict[str, Any]] = []
    for idx, row in enumerate(qa_rows):
        if source == "question":
            qtext = row.get("question")
        elif source == "paraphrase":
            qtext = row.get("paraphrase")
        else:
            raise SystemExit(f"[ERROR] Unknown source: {source}")
        qtext = (qtext or "").strip()
        qid = f"{row.get('template', 'Q')}_{idx:05d}"
        items.append({"qid": qid, "qtext": qtext, "gold": row.get("gt", ""), "raw": row})

    answers: List[Dict[str, Any]] = []

    for bstart in range(0, len(items), batch_size):
        batch = items[bstart : bstart + batch_size]
        batch_qs = [(it["qid"], it["qtext"]) for it in batch]
        messages = build_batch_messages(batch_qs, prelude, timeline_text)
        raw_resp = chat_once(client, model, messages, temperature, max_tokens)
        parsed = parse_batch_json(raw_resp)
        pred_by_id: Dict[str, Dict[str, str]] = {d.get("id", ""): d for d in parsed if d.get("id")}

        for it in batch:
            qid = it["qid"]
            gold = it["gold"]
            row_rec = it["raw"]
            p = pred_by_id.get(qid, {})
            pred_answer = canonicalize_not_answerable(p.get("answer", ""))
            pred_expl = p.get("explanation", "")

            gold_answer = canonicalize_not_answerable(gold)

            ans_type = row_rec.get("answer_type", None)
            if ans_type is None or str(ans_type).strip() == "":
                ans_type = infer_answer_type(gold_answer, pred_answer)

            score = eval_score(gold_answer, pred_answer, ans_type)

            answers.append({
                "id": qid,
                "type": row_rec.get("type"),
                "template": row_rec.get("template"),
                "source": source,
                "question": it["qtext"],
                "answer": gold_answer,
                "pred": pred_answer,
                "answer_type": ans_type,
                "score": score,
                "explanation": pred_expl,
                "difficulty": row_rec.get("difficulty"),
                "range": row_rec.get("range"),
            })

    eval_dif_dir.mkdir(parents=True, exist_ok=True)
    answers_path = eval_dif_dir / "answers.jsonl"
    save_jsonl(answers_path, answers)

    acc, f1 = eval_acc_and_f1(answers)
    by_type = _group_metrics(answers, "type")
    by_tmpl = _group_metrics(answers, "template")
    by_diff = _group_metrics(answers, "difficulty")

    eval_report = {
        "dif": dif_dir.name,
        "num_items": len(answers),
        "metrics": {"acc": acc, "f1": f1},
        "breakdown": {"by_type": by_type, "by_template": by_tmpl, "by_difficulty": by_diff},
        "inputs": {"qa_jsonl": str(qa_path), "qa_context": str(ctx_path)},
        "outputs": {"answers_jsonl": str(answers_path)},
        "source": source,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
    }
    with (eval_dif_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    return acc, f1, len(answers)

# ===================== Main CLI =====================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Jericho QA per-DIF with batched queries to an OpenAI model (DocVQA-style scoring)."
    )
    ap.add_argument(
        "--run-folder",
        default=f"/Users/xinzeli/Documents/mem_eval_game/jericho/generated_qa/advent/{DEFAULT_FOLDER}",
    )
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "your_key"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "your_url"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--source",
        choices=["question", "paraphrase"],
        default="paraphrase",
        help="Use 'question' or 'paraphrase' field as the prompt.",
    )
    ap.add_argument("--icons-path", type=str, default="", help="Path to an icons image for the game (optional).")
    ap.add_argument("--batch-size", type=int, default=8, help="How many questions per request batch.")
    args = ap.parse_args()

    run_folder = Path(args.run_folder).expanduser().resolve()
    dif_dirs = discover_dif_dirs(run_folder)
    eval_root = compute_eval_root_from_run(run_folder)

    client = openai_client(args.api_key, args.base_url)
    prelude = build_instruction_prelude(args.icons_path)

    print(f"Found {len(dif_dirs)} DIF folders under: {run_folder}")

    summary_index: List[Dict[str, Any]] = []
    for dif_dir in dif_dirs:
        eval_dif_dir = (eval_root / dif_dir.name) if dif_dir.name.startswith("DIF_") else (eval_root / "DIF_SINGLE")
        print(f"Evaluating {dif_dir.name} -> {eval_dif_dir}")
        acc, f1, n = evaluate_dif_folder(
            client=client,
            dif_dir=dif_dir,
            eval_dif_dir=eval_dif_dir,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            source=args.source,
            prelude=prelude,
            batch_size=args.batch_size,
        )
        summary_index.append({"dif": dif_dir.name, "num_items": n, "metrics": {"acc": acc, "f1": f1}, "eval_dir": str(eval_dif_dir)})

    index_path = eval_root / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"run_folder": str(run_folder), "per_dif": summary_index, "source": args.source}, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Per-DIF results written under: {eval_root}")

if __name__ == "__main__":
    main()
