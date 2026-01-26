# EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents

EMemBench is a **programmatic benchmark framework** for evaluating **episodic (experience-grounded) memory** in interactive agents.  
Instead of using a fixed, static QA set, EMemBench generates questions **from each agent’s own interaction trajectory** and computes **verifiable ground-truth answers** from underlying game signals.

This repo provides an end-to-end pipeline for:
- **Jericho** (text-only interactive fiction)
- **Crafter** (visual, partially observed survival & crafting)

> EMemBench is not a single fixed dataset. It is a **benchmark generator + evaluation harness**: run an agent → log → generate QA with programmatic GT → answer & score.

---

## Paper

**EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents**  
Xinze Li, Ziyue Zhu, Siyuan Liu, Yubo Ma, Yuhang Zang, Yixin Cao†, Aixin Sun†

---

## Key Ideas

- **Trajectory-conditioned QA**: questions are derived from the agent’s **own** interaction trace.
- **Programmatic, verifiable ground truth**: answers are computed from game signals / structured logs.
- **Query Horizon Control (QHC)**: templates can optionally restrict evidence selection and answer computation to a **prefix window** (e.g., steps 1..50) to reduce confounds from variable episode lengths.  
  - **Legacy naming note**: current code passes QHC values via flags named `--difficulties` / `--difficulty`, and writes to folders like `DIF_-1`, `DIF_50`. These values correspond to **QHC settings**.

---

## Repository Layout (Expected)

### Text (Jericho)

```
text_game/
  game_envs/                 # Jericho ROMs (.z3/.z5/...)
  run_jericho_openai.py      # play + log
  generate_jericho_qa.py     # QA generation (+ indices/maps)
  answer_jericho_qa.py       # answer + eval
  run_text_game_pipeline.py  # E2E entry (play -> gen -> answer)

logs/
  <game>/..._logs.jsonl

generated_qa/
  <game>/<run_name>/
    DIF_-1/                  # legacy folder name = QHC=-1
    DIF_50/                  # legacy folder name = QHC=50
    ...

eval/
  <game>/<run_name>/...
```

### Visual (Crafter)

```
visual_game/
  instructions/
  run_crafter_openai.py       # play + log + frames + map file
  generate_crafter_qa.py      # QA generation
  answer_crafter_qa.py        # answer + eval
  run_visual_game_pipeline.py # E2E entry (play -> gen -> answer)

log/
  seed{SEED}/{RUN_NAME}/
    logs.jsonl
    map_seed{SEED}.txt
    frames/*.png

generated_qa/
  seed{SEED}/{RUN_NAME}/
    qa_context.json
    DIF_-1/qa.jsonl           # legacy folder name = QHC=-1
    DIF_50/qa.jsonl           # legacy folder name = QHC=50
    ...

eval/
  seed{SEED}/{RUN_NAME}/...
```

---

## Installation

### 1) Python environment

```bash
conda create -n emembench python=3.10
conda activate emembench
pip install -r requirements.txt
```

### 2) Jericho (text games)

Jericho typically requires Linux + basic build tools. Install and download the spaCy model:

```bash
pip install jericho
python -m spacy download en_core_web_sm
```

You must place Jericho ROM files under `text_game/game_envs/` (they are not included in this repo).

### 3) Crafter (visual game)

```bash
pip install crafter
```

### 4) Model API (OpenAI-compatible)

The provided runners assume an **OpenAI-compatible chat API**.

```bash
export OPENAI_API_KEY="YOUR_KEY"
# Optional (if your code supports OpenAI-compatible endpoints):
export OPENAI_BASE_URL="https://YOUR_ENDPOINT"
# Optional:
export OPENAI_MODEL="gpt-5.1"
```

---

## Quickstart: End-to-End Pipelines

### A) Jericho (Text) — one command

From the `text_game/` directory (or repo root, depending on your working directory):

```bash
python run_text_game_pipeline.py \
  --model gpt-5.1 \
  --max-steps 200 \
  --history-turns 30 \
  --difficulties -1 50 \
  --max-per-type 2 \
  --logs-root logs \
  --qa-root generated_qa
```

What it does (per game):
1. **Play & log** → `logs/<game>/*_logs.jsonl`  
2. **Generate QA** (QHC values) → `generated_qa/<game>/<run_name>/DIF_*`  
3. **Answer & evaluate** → `eval/<game>/<run_name>/...`

**Notes**
- `--history-turns` controls how many recent turns are included in the policy prompt during play.
- The list of games is defined in `run_text_game_pipeline.py` (edit `JERICHO_GAMES` to run more/fewer titles).

---

### B) Crafter (Visual) — one command (multi-seed)

From the `visual_game/` directory (or repo root):

```bash
python run_visual_game_pipeline.py \
  --seeds 1 42 43 100 123 \
  --steps 500 \
  --history-turns 10 \
  --difficulties -1 50 \
  --qa-source paraphrase \
  --qa-temperature 0.0 \
  --qa-max-tokens 4096 \
  --batch-size 8 \
  --frames-mode mosaic
```

Override the answering model (optional):

```bash
python run_visual_game_pipeline.py \
  --seeds 42 \
  --qa-model gpt-5.1
```

**Notes**
- `--frames-mode` controls how frames are packaged into evaluation prompts (`mosaic` is typically the most economical).
- Outputs are grouped by seed: `log/seed{SEED}/{RUN_NAME}/...`

---

## Stage-by-Stage Usage (Debug / Research Workflow)

You can run each stage separately to inspect logs/frames, regenerate QA deterministically, or swap in a different answering model.

### 1) Play & log

**Jericho**
```bash
python run_jericho_openai.py \
  --rom game_envs/zork3.z5 \
  --model gpt-5.1 \
  --max_steps 200 \
  --history_turns 30 \
  --logdir logs/zork3
```

**Crafter**
```bash
python run_crafter_openai.py \
  --steps 500 \
  --seed 42 \
  --history-turns 10
```

### 2) Generate QA (Query Horizon Control)

**Jericho**
```bash
python generate_jericho_qa.py \
  --input-dir logs/zork3 \
  --default-folder <RUN_NAME> \
  --game zork3 \
  --output-dir generated_qa \
  --max-per-type 2 \
  --difficulty -1 \
  --paraphrase True
```

**Crafter**
```bash
python generate_crafter_qa.py \
  --log-file log/seed42/<RUN_NAME>/logs.jsonl \
  --map-file log/seed42/<RUN_NAME>/map_seed42.txt \
  --output-dir generated_qa/seed42/<RUN_NAME>/DIF_-1 \
  --difficulty -1
```

### 3) Answer & evaluate

**Jericho**
```bash
python answer_jericho_qa.py \
  --run-folder generated_qa/zork3/<RUN_NAME> \
  --model gpt-5.1 \
  --temperature 0.0 \
  --max-tokens 1024 \
  --source paraphrase \
  --batch-size 8
```

**Crafter**
```bash
python answer_crafter_qa.py \
  --run-folder generated_qa/seed42/<RUN_NAME> \
  --source paraphrase \
  --temperature 0.0 \
  --max-tokens 4096 \
  --batch-size 8 \
  --frames-mode mosaic \
  --model gpt-5.1
```

---

## Outputs

### Logs
- Jericho: `logs/<game>/*_logs.jsonl`
- Crafter: `log/seed{SEED}/{RUN_NAME}/logs.jsonl` + `frames/` + `map_seed{SEED}.txt`

### QA artifacts
- `qa_context.json`: agent-observable context used to build evaluation prompts
- `qa.jsonl`: one QA per line (question, metadata, GT answer, evidence pointers, etc.)

### Evaluation
- per-question predictions: `answers.jsonl` (or equivalent)
- aggregated metrics: `index.json` (or equivalent)

---

## Upstream Environments

- Jericho: https://github.com/microsoft/jericho
- Crafter: https://github.com/danijar/crafter
