<p align="center">
  <h1 align="center">EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents</h1>
    <p align="center">
    <a href="https://lixinze777.github.io/"><strong>Xinze Li</strong></a>
    ·
    <strong>Ziyue Zhu</strong>
    ·
    <strong>Siyuan Liu</strong>
    ·
    <a href="https://mayubo2333.github.io"><strong>Yubo Ma</strong></a>
    ·
    <a href="https://yuhangzang.github.io/"><strong>Yuhang Zang</strong></a>
    ·
    <a href="https://sites.google.com/view/yixin-homepage"><strong>Yixin Cao</strong></a>
    ·
    <a href="https://personal.ntu.edu.sg/axsun/"><strong>Aixin Sun</strong></a>
  </p>


EMemBench is a **programmatic benchmark framework** for evaluating **episodic (experience-grounded) memory** in interactive agents.  
Instead of using a fixed, static QA set, EMemBench generates questions **from each agent’s own interaction trajectory** and computes **verifiable ground-truth answers** from underlying game signals.

This repo provides an end-to-end pipeline for:
- **Jericho** (text-only interactive fiction)
- **Crafter** (visual, partially observed survival & crafting)

> EMemBench is not a single fixed dataset. It is a **benchmark generator + evaluation harness**: run an agent → log → generate QA with programmatic GT → answer & score.


<p align="center">
  <img src="paper/emembench concept.png" width="600" />
</p>
<p align="center">
  <em>Figure 1: EMemBench overview. An agent interacts with game environment to produce an episode trajectory. We log agent-observable signals and all underlying game signals. A carefully designed algorithm converts each episode into a QA set with calculated ground truths, and the same agent then answers these questions using only agent-observable context plus its own memory.</em>
</p>


---

## Key Ideas

- **Trajectory-conditioned QA**: questions are derived from the agent’s **own** interaction trace.
- **Programmatic, verifiable ground truth**: answers are computed from game signals / structured logs.
- **Query Horizon Control (QHC)**: templates can optionally restrict evidence selection and answer computation to a **prefix window** (e.g., steps 1..50) to reduce confounds from variable episode lengths.  
  - **Legacy naming note**: current code passes QHC values via flags named `--difficulties` / `--difficulty`, and writes to folders like `DIF_-1`, `DIF_50`. These values correspond to **QHC settings**.

---

## Repository Layout

### Text (Jericho)

```
text_game/
  game_envs/                 # Jericho ROMs (.z3/.z5/...)
  run_jericho_openai.py      # play + log
  generate_jericho_qa.py     # QA generation (+ indices/maps)
  answer_jericho_qa.py       # answer + eval
  run_text_game_pipeline.py  # E2E entry (play -> gen -> answer)

game_envs/
  advent.z5
  ...
  zork3.z5

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

## ✒️Citation
```
@misc{li2026emembenchinteractivebenchmarkingepisodic,
      title={EMemBench: Interactive Benchmarking of Episodic Memory for VLM Agents}, 
      author={Xinze Li and Ziyue Zhu and Siyuan Liu and Yubo Ma and Yuhang Zang and Yixin Cao and Aixin Sun},
      year={2026},
      eprint={2601.16690},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.16690}, 
}
```

## 📄 License
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg) **Usage and License Notices**: The data and code are intended and licensed for research use only.
License: Attribution-NonCommercial 4.0 International It should abide by the policy of OpenAI: https://openai.com/policies/terms-of-use
