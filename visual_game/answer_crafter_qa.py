#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple


# Reusable DocVQA-style eval logic
from eval_score import eval_score, eval_acc_and_f1, infer_answer_type, canonicalize_not_answerable
# ===================== 常量 =====================
DEFAULT_FOLDER = "gpt41_500steps_10rounds_20251107-191027"

# ===================== 可选：PIL用于图例PNG与拼图（建议安装） =====================
try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None

# ===================== tqdm 进度条（可选） =====================
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x, **kwargs):
        return x

# ===================== OpenAI 客户端（保持你的用法） =====================
def openai_client(api_key: str, base_url: str):
    """
    构造 OpenAI 客户端。保持与你原脚本一致（需要 pip install openai）。
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package is required. Try: pip install openai") from e
    return OpenAI(api_key=api_key, base_url=base_url)

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

# ===================== Instruction & 图例 =====================
def load_game_instructions(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    games = obj.get("games", {})
    if not games:
        raise SystemExit(f"[ERROR] No 'games' key found in instruction file: {path}")
    out: Dict[str, str] = {}
    for k, v in games.items():
        if isinstance(v, list):
            out[k.lower()] = "\n".join(str(x) for x in v)
        else:
            out[k.lower()] = str(v)
    return out

def load_icons_data_url(icons_path: str) -> str:
    if not icons_path or Image is None:
        return ""
    try:
        with Image.open(icons_path) as im:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
    except Exception:
        return ""

# ===================== 归一化（用于时间线文本展示） =====================
def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

# ===================== 评测指标（DocVQA-style；复用 eval_score.py） =====================
# NOTE: Do NOT change inference/prompting logic elsewhere; we only swap evaluation here.
# Per-sample: score = eval_score(answer, pred, answer_type)
# Overall: acc = mean(score), f1 = dataset-level F1 treating "Not answerable" as negative.

# ===================== Prompt 模板 =====================
# Extend the system prompt to emphasize that all steps come from a single continuous episode.
# The model should use the entire history when answering and never guess when information is missing.
SYSTEM_PROMPT = (
    "You are a concise answerer for questions about the Crafter environment. "
    "All provided steps are part of a single continuous episode of the game. "
    "The timeline presents actions, reasons, and health related stats for each step in order. "
    "Use the entire history to answer questions; recall what was seen or collected earlier even if it is not visible in the current frame. "
    "If there is not enough information in the given steps to answer, reply with 'not answerable' as the answer and briefly explain why. "
    "Always reply in strict JSON format and do not guess."
)

INSTR_SCHEMA = (
    "Return strictly JSON.\n"
    "For a single question: {\"answer\": \"<short>\", \"explanation\": \"<brief>\"}\n"
    "For batched questions: an array of objects, each: {\"id\": \"<qid>\", \"answer\": \"<short>\", \"explanation\": \"<brief>\"}\n"
    "Keep explanation short."
    "For any question you find not answerable, strictly reply 'not answerable' as answer, and give explanation."
)

USER_BATCH_HEADER = (
    "You will answer multiple questions about the same episode.\n"
    + INSTR_SCHEMA
    + "\nQuestions:\n"
)

def build_instruction_prelude(env_name: str, instr_file: str, icons_path: str) -> List[Dict[str, Any]]:
    prelude: List[Dict[str, Any]] = []
    game_instr = ""
    if instr_file:
        try:
            games = load_game_instructions(instr_file)
            game_instr = games.get(env_name.lower(), "").strip()
        except Exception as e:
            print(f"[WARN] Failed to load instructions from {instr_file}: {e}")
    sys_text = SYSTEM_PROMPT
    if game_instr:
        sys_text = sys_text + "\nGame rules: " + game_instr
    prelude.append({"role": "system", "content": sys_text})

    data_url = load_icons_data_url(icons_path)
    if data_url:
        prelude.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "ICON LEGEND for Crafter tiles and HUD icons (reference image)."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        })
    return prelude

# ===================== 帧图与 Mosaic 工具 =====================
def png_to_data_url(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def derive_frames_dir(dif_dir: Path, override: str | None) -> Path:
    """
    Compute the frames directory for a given DIF directory, respecting seed-based layouts.

    If ``override`` is provided, it is returned as an absolute path.  Otherwise, the
    directory is inferred from the run folder structure:

      <pc_root>/generated_qa/<game_dir>/<run_name>/[DIF_xxx]

    Frames are assumed to reside at:

      <pc_root>/log/<game_dir>/<run_name>/frames

    where ``game_dir`` may be 'crafter' or 'seedXXX'.
    """
    if override:
        return Path(override).expanduser().resolve()
    # Determine run_name
    if dif_dir.name.startswith("DIF_"):
        run_name = dif_dir.parent.name
    else:
        run_name = dif_dir.name
    # Locate visual_game root
    cur = dif_dir.resolve()
    pc_root = None
    for _ in range(12):
        if cur.name == "visual_game":
            pc_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if pc_root is None:
        raise FileNotFoundError("Cannot locate 'visual_game' root from dif_dir: " + str(dif_dir))
    # Determine game_dir by inspecting generated_qa path
    # Start from dif_dir or its parent to locate <game_dir>
    cur_dir = dif_dir
    if dif_dir.name.startswith("DIF_"):
        cur_dir = dif_dir.parent
    # cur_dir should be run_name; ascend once to <game_dir>
    if cur_dir.name == run_name:
        cur_dir = cur_dir.parent
    game_dir = None
    try:
        if cur_dir.parent.name == "generated_qa":
            game_dir = cur_dir.name
    except Exception:
        game_dir = None
    if not game_dir:
        game_dir = "crafter"
    # Compose frames directory
    frames_dir = pc_root / "log" / game_dir / run_name / "frames"
    return frames_dir.resolve()

def list_all_frames(frames_dir: Path) -> List[Path]:
    if not frames_dir.is_dir():
        return []
    return sorted(frames_dir.glob("*.png"))

def get_steps_array(qa_context: Any) -> List[Dict[str, Any]]:
    """
    支持两种结构：
      - 列表:  [ {t, action, reason, frame, ...}, ... ]
      - 字典:  {"steps": [ {...}, ... ], ...}
    """
    if isinstance(qa_context, list):
        return qa_context
    if isinstance(qa_context, dict):
        return qa_context.get("steps", []) or []
    return []

def map_frame_to_t(qa_context: Any) -> Dict[str, int]:
    """
    从 qa_context 建立 filename -> t 的映射（如 'crafter_t000063.png' -> 63）
    """
    steps = get_steps_array(qa_context)
    m: Dict[str, int] = {}
    for s in steps:
        t = s.get("t")
        fr = s.get("frame")
        if isinstance(fr, str) and isinstance(t, int):
            m[fr] = t
    return m

def resize_keep_ratio(im: Any, max_side: int) -> Any:
    w, h = im.size
    scale = min(max_side / max(w, h), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), Image.BICUBIC)

def _measure_text(draw, text: str, font) -> Tuple[int, int]:
    """Pillow 兼容的文字尺寸测量。"""
    if hasattr(draw, "textbbox"):
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            return max(1, right - left), max(1, bottom - top)
        except Exception:
            pass
    if hasattr(font, "getsize"):
        try:
            w, h = font.getsize(text)
            return max(1, w), max(1, h)
        except Exception:
            pass
    return max(1, 8 * len(text)), 12

def build_mosaic_images(
    frames: List[Tuple[Path, int]],
    cols: int = 10,
    cell: int = 160,
    font_size: int = 18,
    per_mosaic_max: int = 200,
) -> List[bytes]:
    """
    把 (frame_path, t) 列表打包成若干张网格图；每格叠字 't=<num>'。
    返回 PNG 二进制列表。
    """
    if Image is None:
        return []

    mosaics: List[bytes] = []
    for start in range(0, len(frames), per_mosaic_max):
        chunk = frames[start: start + per_mosaic_max]
        if not chunk:
            continue

        rows = ceil(len(chunk) / cols)
        tiles: List[Tuple[Any, int]] = []
        max_w = max_h = 1
        for p, t in chunk:
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im2 = resize_keep_ratio(im, cell)
                    tiles.append((im2.copy(), t))
                    max_w = max(max_w, im2.size[0])
                    max_h = max(max_h, im2.size[1])
            except Exception:
                pass
        if not tiles:
            continue

        pad = 8
        grid_w = cols * (max_w + pad) + pad
        grid_h = rows * (max_h + pad) + pad
        canvas = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        for idx, (im2, t) in enumerate(tiles):
            r = idx // cols
            c = idx % cols
            x = pad + c * (max_w + pad)
            y = pad + r * (max_h + pad)
            canvas.paste(im2, (x, y))

            label = f"t={t}"
            tw, th = _measure_text(draw, label, font)
            bx, by = x + 4, y + 4
            # 背景黑条
            draw.rectangle((bx - 2, by - 2, bx + tw + 2, by + th + 2), fill=(0, 0, 0))
            draw.text((bx, by), label, fill=(255, 255, 255), font=font)

        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        mosaics.append(buf.getvalue())

        for im2, _ in tiles:
            im2.close()
        canvas.close()

    return mosaics

def mosaics_to_blocks(mosaics: List[bytes]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for data in mosaics:
        b64 = base64.b64encode(data).decode("utf-8")
        blocks.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    return blocks

def build_all_frame_blocks(
    qa_context: Any,
    frames_dir: Path,
    frames_mode: str,
    mosaic_cols: int,
    mosaic_cell: int,
    mosaic_per_image: int,
    mosaic_per_batch_limit: int,
) -> List[Dict[str, Any]]:
    """
    根据模式生成图片 blocks：
      - mosaic: 把所有帧拼图，按 per-image 张数拆分，多张 mosaic，但每个 batch 最多 mosaic_per_batch_limit 张
      - all   : 最多 50 张（API 限制）
      - sample: 仅 3 张（头/中/尾）
    """
    m = map_frame_to_t(qa_context)
    all_paths = list_all_frames(frames_dir)
    frames_with_t: List[Tuple[Path, int]] = []
    for p in all_paths:
        t = m.get(p.name, None)
        if t is None:
            m2 = re.search(r"t0*([0-9]+)\.png$", p.name)
            if m2:
                t = int(m2.group(1))
        if t is None:
            continue
        frames_with_t.append((p, t))
    frames_with_t.sort(key=lambda x: x[1])

    if frames_mode == "mosaic":
        mosaics = build_mosaic_images(
            frames_with_t,
            cols=mosaic_cols,
            cell=mosaic_cell,
            per_mosaic_max=mosaic_per_image,
        )
        return mosaics_to_blocks(mosaics[:mosaic_per_batch_limit])

    # fallback: all / sample
    blocks: List[Dict[str, Any]] = []
    if frames_mode == "all":
        for p, _ in frames_with_t[:50]:
            url = png_to_data_url(str(p))
            if url:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
        return blocks

    # sample
    n = len(frames_with_t)
    picks: List[Path] = []
    if n > 0:
        picks.append(frames_with_t[0][0])
    if n > 2:
        picks.append(frames_with_t[n // 2][0])
    if n > 1:
        picks.append(frames_with_t[-1][0])
    for p in picks:
        url = png_to_data_url(str(p))
        if url:
            blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks

# ===================== 目录发现 & 输出根定位 =====================
def discover_dif_dirs(run_folder: Path) -> List[Path]:
    """
    允许传入：<RUN> 或 <RUN>/DIF_xxx
    """
    if run_folder.name.startswith("DIF_"):
        return [run_folder]
    difs = [p for p in run_folder.iterdir() if p.is_dir() and p.name.startswith("DIF_")]
    if difs:
        return sorted(difs, key=lambda p: (len(p.name), p.name))
    if (run_folder / "qa.jsonl").is_file():
        return [run_folder]
    raise FileNotFoundError(f"No DIF_* folders or qa.jsonl found under: {run_folder}")

def compute_eval_root_from_run(run_folder: Path) -> Path:
    """
    输出固定为：/Users/.../visual_game/eval/crafter/<RUN>
    """
    # 解析 RUN 名
    if run_folder.name.startswith("DIF_"):
        run_name = run_folder.parent.name
    else:
        run_name = run_folder.name

    # 向上找 visual_game 根
    cur = run_folder.resolve()
    pc_root = None
    for _ in range(12):
        if cur.name == "visual_game":
            pc_root = cur
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    if pc_root is None:
        raise FileNotFoundError("Cannot locate 'visual_game' root from run_folder: " + str(run_folder))

    # Determine game_dir by inspecting the generated_qa path structure
    # We expect run_folder to be of the form
    #   <pc_root>/generated_qa/<game_dir>/<run_name>[/(DIF_xxx)]
    # where <game_dir> can be 'crafter' or 'seedXXX'.
    cur_dir = run_folder
    if run_folder.name.startswith("DIF_"):
        # skip the DIF layer
        cur_dir = run_folder.parent
    # cur_dir should be <run_name>
    if cur_dir.name == run_name:
        cur_dir = cur_dir.parent  # move up to <game_dir>
    game_dir = None
    # Check that the parent is generated_qa
    try:
        if cur_dir.parent.name == "generated_qa":
            game_dir = cur_dir.name
    except Exception:
        game_dir = None
    if not game_dir:
        # fallback to legacy 'crafter' if structure is unexpected
        game_dir = "crafter"
    eval_root = pc_root / "eval" / game_dir / run_name
    return eval_root

# ===================== 时间线文本 =====================
def build_context_timeline(qa_context: Any) -> str:
    steps = get_steps_array(qa_context)
    lines: List[str] = []
    for s in steps:
        t = s.get("t")
        action = s.get("action", "")
        reason = norm_text(s.get("reason", "") or "")
        # Gather resource stats if present
        health = s.get("health")
        food = s.get("food")
        water = s.get("water")
        energy = s.get("energy")
        stats_parts: List[str] = []
        def _append_stat(name: str, val: Any) -> None:
            if val is not None and val != "":
                stats_parts.append(f"{name}={val}")
        _append_stat("health", health)
        _append_stat("food", food)
        _append_stat("water", water)
        _append_stat("energy", energy)
        stats_str = "".join(["; stats: ", ", ".join(stats_parts)]) if stats_parts else ""
        lines.append(f"step {t}: action={action}; reason={reason}{stats_str}")
    return "\n".join(lines)

# ===================== 消息构造（批量） =====================
def build_batch_messages(batch_items: List[Tuple[str, str]], prelude: List[Dict[str, Any]],
                         timeline_text: str, frame_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    msgs.extend(prelude)
    lines = [USER_BATCH_HEADER]
    for qid, qtext in batch_items:
        lines.append(f"- id={qid} :: {qtext}")
    if timeline_text:
        lines.append("\nContext timeline:\n" + timeline_text)
    content = [{"type": "text", "text": "\n".join(lines)}]
    content.extend(frame_blocks)
    msgs.append({"role": "user", "content": content})
    return msgs

# ===================== 模型调用与解析 =====================
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

# ===================== 分类聚合 =====================
from collections import defaultdict
def _group_metrics(rows: List[Dict[str, Any]], key_name: str) -> Dict[str, Dict[str, float]]:
    """
    rows: answers 列表；key_name 取 'type' / 'template' / 'difficulty'
    返回: {group_value: {"count": n, "acc": mean(score), "f1": dataset-level F1}}
    """
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

# ===================== 针对单个 DIF 目录的评测（批量 + mosaic） =====================
def evaluate_dif_folder(
    client,
    dif_dir: Path,
    eval_dif_dir: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    source: str,
    prelude: List[Dict[str, Any]],
    frames_dir_override: str | None,
    frames_mode: str,
    mosaic_cols: int,
    mosaic_cell: int,
    mosaic_per_image: int,
    mosaic_per_batch_limit: int,
    batch_size: int,
) -> Tuple[float, float, int]:
    qa_path = dif_dir / "qa.jsonl"
    ctx_path = dif_dir / "qa_context.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Missing qa.jsonl in {dif_dir}")
    if not ctx_path.is_file():
        raise FileNotFoundError(f"Missing qa_context.json in {dif_dir}")

    qa_rows = load_jsonl(qa_path)
    qa_context = load_json(ctx_path)

    # 解析 RUN 名，推导 frames 目录；并拼图/取样
    # Determine frames directory; pass dif_dir for proper seed-based lookup
    frames_dir = derive_frames_dir(dif_dir, frames_dir_override)
    frame_blocks = build_all_frame_blocks(
        qa_context,
        frames_dir,
        frames_mode=frames_mode,
        mosaic_cols=mosaic_cols,
        mosaic_cell=mosaic_cell,
        mosaic_per_image=mosaic_per_image,
        mosaic_per_batch_limit=mosaic_per_batch_limit,
    )

    timeline_text = build_context_timeline(qa_context)

    # 整理 items
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
        # Preserve raw ground truth without converting to string; may be list or numeric
        gt_raw = row.get("gt", "")
        items.append({
            "qid": qid,
            "qtext": qtext,
            "gold": gt_raw,
            "raw": row,
        })

    answers: List[Dict[str, Any]] = []
    n_scored = 0
    for bstart in tqdm(range(0, len(items), batch_size), desc=f"[{dif_dir.name}] batches"):
        batch = items[bstart: bstart + batch_size]
        batch_qs = [(it["qid"], it["qtext"]) for it in batch]
        messages = build_batch_messages(batch_qs, prelude, timeline_text, frame_blocks)
        raw = chat_once(client, model, messages, temperature, max_tokens)
        parsed = parse_batch_json(raw)

        pred_by_id: Dict[str, Dict[str, str]] = {d.get("id", ""): d for d in parsed if d.get("id")}
        for it in batch:
            qid = it["qid"]
            gold = it["gold"]
            row = it["raw"]
            p = pred_by_id.get(qid, {})
            pred_answer = p.get("answer", "")
            pred_expl = p.get("explanation", "")

            # DocVQA-style scoring (ANLS + numeric tolerance + NA-aware dataset F1)
            pred_answer = canonicalize_not_answerable(pred_answer)
            gold_eval = gold
            if isinstance(gold_eval, str):
                gold_eval = canonicalize_not_answerable(gold_eval)

            ans_type = row.get("answer_type", None)
            if ans_type is None or str(ans_type).strip() == "":
                ans_type = infer_answer_type(gold_eval, pred_answer)

            score = eval_score(gold_eval, pred_answer, ans_type)

            answers.append(
                {
                    "id": qid,
                    "type": row.get("type"),
                    "template": row.get("template"),
                    "source": source,
                    "question": it["qtext"],
                    # keep raw GT for debugging/backward-compat
                    "gt": gold,
                    # keys required by eval_acc_and_f1:
                    "answer": gold_eval,
                    "pred": pred_answer,
                    "answer_type": ans_type,
                    "score": score,
                    "explanation": pred_expl,
                    "difficulty": row.get("difficulty"),
                    "range": row.get("range"),
                }
            )
            n_scored += 1

    # 写出
    eval_dif_dir.mkdir(parents=True, exist_ok=True)
    answers_path = eval_dif_dir / "answers.jsonl"
    save_jsonl(answers_path, answers)

    acc, f1 = eval_acc_and_f1(answers)

    # 分类统计
    by_type = _group_metrics(answers, "type")
    by_tmpl = _group_metrics(answers, "template")
    by_diff = _group_metrics(answers, "difficulty")

    eval_report = {
        "dif": dif_dir.name,
        "num_items": n_scored,
        "metrics": {"acc": acc, "f1": f1},
        "breakdown": {                   # 分类统计
            "by_type": by_type,
            "by_template": by_tmpl,
            "by_difficulty": by_diff,
        },
        "inputs": {
            "qa_jsonl": str(qa_path),
            "qa_context": str(ctx_path),
            "frames_dir": str(frames_dir),
            "frames_mode": frames_mode,
        },
        "outputs": {"answers_jsonl": str(answers_path)},
        "source": source,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "mosaic": {
            "cols": mosaic_cols,
            "cell": mosaic_cell,
            "per_image": mosaic_per_image,
            "per_batch_limit": mosaic_per_batch_limit,
        },
    }
    with (eval_dif_dir / "eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    return acc, f1, n_scored

# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate Crafter QA per-DIF with selectable question source, instructions, and frames mosaics (batched)."
    )
    # OpenAI 设置保持不变
    ap.add_argument(
        "--run-folder",
        default=f"/Users/xinzeli/Documents/mem_eval_game/visual_game/generated_qa/crafter/{DEFAULT_FOLDER}",
        help="Path to .../generated_qa/crafter/<RUN_NAME> or a specific DIF_* folder",
    )
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "sk-keKB2Y0DPJFFqWPblW6pWueLFkrq6gk66QkbqgzLc5sAxC46"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096)

    # 题面来源（按你给的片段）
    ap.add_argument("--source", choices=["question", "paraphrase"], default="paraphrase",
                    help="Use 'question' or 'paraphrase' field as the prompt.")

    # Instruction 注入（可选）
    ap.add_argument("--env", type=str, default="crafter")
    ap.add_argument("--instr-file", type=str, default="instructions/crafter_instructions.json")
    ap.add_argument("--icons-path", type=str, default="instructions/crafter_icons.png")

    # 帧图控制（增加 mosaic）
    ap.add_argument("--frames-dir", type=str, default=None,
                    help="Override frames directory; otherwise use /Users/.../visual_game/log/crafter/<RUN>/frames")
    ap.add_argument("--frames-mode", type=str, choices=["mosaic", "all", "sample"], default="mosaic",
                    help="mosaic (default) to avoid 50-image limit; 'all' caps at 50; 'sample' uses 3 images.")
    ap.add_argument("--mosaic-cols", type=int, default=10, help="Number of columns in each mosaic grid.")
    ap.add_argument("--mosaic-cell", type=int, default=160, help="Max side length of each tile (pixels).")
    ap.add_argument("--mosaic-per-image", type=int, default=200, help="Max frames per mosaic image.")
    ap.add_argument("--mosaic-per-batch", type=int, default=10, help="Max number of mosaic images attached per batch.")

    # 批量
    ap.add_argument("--batch-size", type=int, default=8, help="How many questions per request.")

    args = ap.parse_args()

    run_folder = Path(args.run_folder).expanduser().resolve()
    dif_dirs = discover_dif_dirs(run_folder)

    # 输出根目录：/visual_game/eval/crafter/<RUN>/
    eval_root = compute_eval_root_from_run(run_folder)
    client = openai_client(args.api_key, args.base_url)

    # 构造一次前置 instruction 消息
    prelude = build_instruction_prelude(args.env, args.instr_file, args.icons_path)

    print(f"Found {len(dif_dirs)} DIF folders under: {run_folder}")
    summary_index: List[Dict[str, Any]] = []

    for dif_dir in dif_dirs:
        if dif_dir.name.startswith("DIF_"):
            eval_dif_dir = eval_root / dif_dir.name
        else:
            eval_dif_dir = eval_root / "DIF_SINGLE"

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
            frames_dir_override=args.frames_dir,
            frames_mode=args.frames_mode,
            mosaic_cols=args.mosaic_cols,
            mosaic_cell=args.mosaic_cell,
            mosaic_per_image=args.mosaic_per_image,
            mosaic_per_batch_limit=args.mosaic_per_batch,
            batch_size=args.batch_size,
        )
        summary_index.append(
            {
                "dif": dif_dir.name,
                "num_items": n,
                "metrics": {"acc": acc, "f1": f1},
                "eval_dir": str(eval_dif_dir),
            }
        )

    # 顶层索引（各 DIF 各算）
    index_path = eval_root / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({"run_folder": str(run_folder), "per_dif": summary_index, "source": args.source}, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Per-DIF results written under: {eval_root}")

if __name__ == "__main__":
    main()
