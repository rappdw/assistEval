# Objective Benchmark Repo — ChatGPT vs Microsoft Copilot

A repo specification for building a fair, extensible, and repeatable evaluation harness that compares general-purpose assistants (initially Microsoft Copilot vs ChatGPT) on offline and online tasks with a clear scoring rubric.

---

## Goals & Non‑Goals

**Goals**
- Provide an open, reproducible harness to compare assistants on well-specified tasks.
- Make adding/modifying test cases trivial via declarative YAML/JSON.
- Enforce strict constraints (e.g., no browsing for offline tests; JSON-only outputs).
- Produce an objective numeric score and a concise report per run.
- Support multiple providers (ChatGPT, Microsoft Copilot) and manual-paste mode if APIs are unavailable.

**Non‑Goals**
- Build UI automation to scrape vendor UIs (phase 2 at most).
- Judge subjective creativity; we focus on enterprise-relevant skills, structure compliance, and correctness.

---

## High-Level Architecture

1. **Test Definitions (declarative)** — YAML files describing tasks, inputs, constraints, expected outputs, and scoring rules.
2. **Runner** — Executes prompts (or guides manual collection), collects model outputs, and stores raw artifacts.
3. **Validators** — Schema validation, structural checks (e.g., word counts), and content normalization.
4. **Evaluators** — Task-specific logic to compute correctness (e.g., metric diffs, regex evaluation).
5. **Scorer** — Aggregates task scores per provider with weight configuration.
6. **Report Generator** — Builds markdown/JSON summary, leaderboard, and per-task breakdown with failure reasons.

All components are modular and pluggable; adding a new test type is a matter of implementing an evaluator and schema extension.

---

## Repository Layout

```
bench/
  adapters/
    base.py               # Provider interface (sync abstraction)
    chatgpt.py            # ChatGPT adapter (OpenAI API or local wrapper)
    copilot_manual.py     # Manual paste adapter (no API) – prompts evaluator to paste raw text
    # copilot_api.py      # Placeholder for future API-based Copilot adapter
  core/
    runner.py             # Orchestrates runs across providers & test sets
    validators.py         # JSON schema, structure, field checks
    evaluators/
      __init__.py
      metrics_csv.py      # Task 1 evaluator (precision/recall/F1/accuracy)
      regex_match.py      # Task 2 evaluator (SSN regex + line matches)
      exec_summary.py     # Task 3 evaluator (title/word count/bullets/tone heuristics)
      deep_research.py    # Optional evaluator (plan structure, sources recency)
    scoring.py            # Aggregation, weights, stability bonus
    reporting.py          # Markdown/JSON output, per-run artifacts
    utils.py              # Tokenization, word counting, seed mgmt
  tests/
    offline/
      task1_metrics.yaml
      task2_ssn_regex.yaml
      task3_exec_summary.yaml
    online/
      deep_research_agentic_ai.yaml
  fixtures/
    csv/
      phishing_sample.csv
    text/
      ssn_validation_lines.txt
  answer_keys/
    offline/
      task1_metrics.json
      task2_lines.json
  schemas/
    test_case.schema.json
    rubric.schema.json
  configs/
    providers.yaml        # Model settings, temperature, seeds, tool allowances
    weights.default.yaml  # Default scoring weights per task
    runmatrix.yaml        # Which providers × which test sets, repetitions
results/
  .gitkeep
scripts/
  bench.py                # CLI entrypoint (thin wrapper around runner)
  make_report.py          # Generate combined report from prior runs
README.md
pyproject.toml / requirements.txt
.github/workflows/benchmark.yml
```

---

## Provider Abstraction

```python
# bench/adapters/base.py
class Provider:
    name: str
    def invoke(self, system: str, user: str, *, options: dict, capabilities: dict) -> dict:
        """Return {"raw_text": str}. Enforce capabilities (e.g., disable browsing) upstream."""
```

- **ChatGPT adapter**: Calls OpenAI API with deterministic options (temperature=0, seed if available, max_tokens, response_format=json when supported).
- **Copilot manual adapter**: The runner prints the prompt and waits for paste of the response (stored as artifact). Useful until an official API is available.
- **Capabilities**: `{ "web": "forbidden|required|allowed", "tools": ["code", "files"], "json_required": true }` are enforced by prompt preamble and adapter options.

---

## Test Case Schema (YAML)

```yaml
# schemas/test_case.schema.json governs validation
id: "offline.task1.metrics"
name: "Task 1 — Metrics from CSV"
category: "offline"  # offline|online
capability_profile:
  web: "forbidden"    # forbidden|allowed|required
  json_required: true
  retries: 1
prompt:
  system: |
    You are an enterprise assistant. Follow instructions exactly. Do not browse the web. Do not fabricate sources.
  user: |
    [TASK SPEC HERE]

# What to extract from model response (JSONPath-like selectors)
expectation:
  schema_name: "task1_metrics"  # maps to a JSON schema in validators
  fields:
    - path: $.task1_data_metrics.precision
      type: number
    - path: $.task1_data_metrics.recall
      type: number
    - path: $.task1_data_metrics.f1
      type: number
    - path: $.task1_data_metrics.accuracy
      type: number
    - path: $.task1_data_metrics.confusion_matrix.tp
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.fp
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.fn
      type: integer
    - path: $.task1_data_metrics.confusion_matrix.tn
      type: integer

scoring:
  evaluator: metrics_csv
  config:
    # Numbers must match expected to within +/- 0.0005; integers exact
    tolerance: 0.0005
    round_to: 4
    weights:
      precision: 6
      recall: 6
      f1: 6
      accuracy: 6
      confusion_matrix: { tp: 3, fp: 3, fn: 3, tn: 3 }
```

> Other tasks (regex, exec summary, deep research) follow the same top-level structure with task-specific evaluator configs.

---

## Sample Test Definitions

### 1) Offline — Metrics from CSV
`tests/offline/task1_metrics.yaml`

- **Input**: `fixtures/csv/phishing_sample.csv`
- **Prompt**: Inlined CSV + required JSON output schema.
- **Answer Key**: `answer_keys/offline/task1_metrics.json`

```json
{
  "precision": 0.75,
  "recall": 0.6,
  "f1": 0.6667,
  "accuracy": 0.625,
  "confusion_matrix": {"tp": 3, "fp": 1, "fn": 2, "tn": 2}
}
```

### 2) Offline — DLP Regex for U.S. SSN
`tests/offline/task2_ssn_regex.yaml`

- **Input**: `fixtures/text/ssn_validation_lines.txt` (12 lines)
- **Evaluator**: Compiles provided regex, checks matches vs expected lines.
- **Answer Key**: `answer_keys/offline/task2_lines.json` ⇒ `{ "matches": [1,8,9,12] }`
- **Additional Checks**: Regex must anchor start/end, avoid catastrophic backtracking (timeout guard), and not use inline code execution.

### 3) Offline — Executive Summary
`tests/offline/task3_exec_summary.yaml`

- **Evaluator**: Structural constraints + tone heuristics.
  - Title ≤ 6 words
  - Word count 120–160 (summary only)
  - Exactly 3 bullets
  - Tone: reject hype terms (configurable denylist) and require concise sentences (avg ≤ 24 words)

### 4) Online (Optional) — Deep Research
`tests/online/deep_research_agentic_ai.yaml`

- **Capability**: `web: required`
- **Evaluator**: Validates 7–10 steps with goal/method/deliverable; risk register ≥5 with likelihood & impact 1–5; 5–8 sources with ≥3 in last 3 years (parse years from citations); requires explicit assumptions/limitations fields.
- **Note**: If the provider cannot browse, evaluator awards partial credit for structure and penalizes missing verified sources.

---

## Scoring Rubric (Default Weights)

Weights are configurable via `configs/weights.default.yaml`.

- **Task 1 (40 pts)**: precision 6, recall 6, F1 6, accuracy 6, CM (tp/fp/fn/tn) 3 each.
  - **Numeric Rule**: absolute error ≤ 0.0005 from key (rounded to 4 decimals) ⇒ full points; else 0 for that sub-metric.
- **Task 2 (30 pts)**: Regex validity 18, line matches 12 (1 pt per correct line; must not over-match).
  - Validity checks (18 pts): anchors, area/group/serial constraints, no 000/666/9xx, no 00, no 0000.
- **Task 3 (20 pts)**: Structure 12 pts (title length 3, word count 3, exactly 3 bullets 3, JSON schema compliance 3); Tone/clarity 8 pts (heuristics + optional LLM-judge for tiebreaks).
- **Deep Research (10 pts)**: Plan quality & sequencing 5; source quality & recency or explicit limitations 5.
- **Stability Bonus (+0–5)**: Run each offline task 3×; if Task 1 numbers exact across runs and Task 2/3 structures consistent (no schema failures), award 5; partial consistency prorated.

**Overall Score** = sum of task scores (and optional bonus). The report shows per-task breakdown and reasons for deductions.

---

## Runner Configuration

`configs/providers.yaml`
```yaml
providers:
  - name: chatgpt
    adapter: chatgpt
    options:
      temperature: 0
      max_tokens: 1200
      seed: 42
      response_format: json
  - name: copilot
    adapter: copilot_manual
    options: {}
```

`configs/runmatrix.yaml`
```yaml
matrix:
  - provider: chatgpt
    test_set: offline
    repetitions: 3
  - provider: copilot
    test_set: offline
    repetitions: 3
  - provider: chatgpt
    test_set: online
    repetitions: 1
  - provider: copilot
    test_set: online
    repetitions: 1
```

---

## CLI Usage

```bash
# Install
pip install -r requirements.txt

# Run full matrix
python scripts/bench.py run --config configs/runmatrix.yaml --weights configs/weights.default.yaml

# Run a single test for a single provider
python scripts/bench.py run --provider chatgpt --test tests/offline/task2_ssn_regex.yaml

# Generate consolidated report from latest runs
python scripts/make_report.py --results results/
```

When using `copilot_manual`, the runner will display the prompt and pause to let you paste the raw model output. The tool saves the raw text and proceeds with validation.

---

## Validators & Evaluators (Details)

### JSON/Structure Validation
- Enforce exact JSON shape per task schema (strict mode), with clear error messages.
- Word counts: tokenized on whitespace; configurable stopword trimming is **off** by default.
- Bullet count: parse top-level array; forbid nested bullets.

### Task 1: Metrics from CSV
- Load expected metrics from answer key.
- Check numeric fields to tolerance and rounding.
- Check confusion matrix integers for exact match.
- Optional: recalc from CSV to verify internal consistency if the model also returns per-row labels (future extension).

### Task 2: SSN Regex
- Compile with `re.fullmatch` per token; use timeout guard (e.g., 100 ms per line) to prevent catastrophic backtracking.
- Evaluate against lines; award points for correct matches and validity rules.

### Task 3: Executive Summary
- Count title words, summary words (exclude bullets), bullet count.
- Tone heuristics: denylist (e.g., "revolutionary", "game‑changing", "world‑class"); sentence length stats.
- Optional LLM-as-judge: if enabled, run a small model locally or via configured provider and map to 0–8 pts with a rubric prompt.

### Online — Deep Research
- Structure checks: 7–10 ordered steps with fields {goal, method, deliverable}.
- Risk register: ≥5 items each with {risk, likelihood 1–5, impact 1–5, mitigation}.
- Sources: 5–8 entries; parse years; require ≥3 with year ≥ (current_year-3).
- If `web` is not actually used (declared but absent), auto-penalize source points.

---

## Results & Reporting

- Each run produces a directory `results/run_<timestamp>/` containing:
  - `config.json` (provider options, seeds, test list)
  - `raw/<provider>/<test_id>.txt` (verbatim model output)
  - `parsed/<provider>/<test_id>.json` (normalized JSON)
  - `scores/<provider>.json` (per-test & totals)
  - `report.md` (human-readable summary + leaderboard)
- A consolidated `results/latest/` symlink or pointer aids automation.
- GitHub Action uploads the markdown as an artifact and prints a summary table in the job log.

---

## Adding New Test Cases

1. Copy a template from `tests/_templates/<type>.yaml`.
2. Put any fixtures into `fixtures/` and update the test YAML paths.
3. If objective scoring is possible, add an answer key under `answer_keys/`.
4. Validate locally: `python scripts/bench.py validate --test tests/...yaml`.
5. Commit and run the matrix.

Design guidance:
- Prefer tasks with objective scoring (numeric, regex, exact string, structural constraints).
- Keep prompts vendor-neutral; avoid product-specific instructions.
- Tag tasks with domain labels (e.g., `security`, `dlp`, `metrics`, `writing`).

---

## CI/CD (GitHub Actions)

`.github/workflows/benchmark.yml`
- Triggers on `push` to main and on `workflow_dispatch`.
- Job matrix runs selected providers and tests (manual mode for Copilot prompts the maintainer to paste results into an artifact step — documented in README).
- Publishes `report.md` and `scores/*.json` as artifacts.

---

## Data Handling & Security

- Repository contains **synthetic** or non-sensitive fixtures only.
- No production data, secrets, or PII.
- Logs redact any tokens/keys; manual pastes are stored locally in the run folder.
- Optional: hash raw outputs for integrity; include a `NOTICE` about evaluation bias and limitations.

---

## Roadmap (Phased)

**Phase 1 (MVP)**
- Offline tasks 1–3 with scoring.
- ChatGPT adapter + Copilot manual adapter.
- CLI + basic report.

**Phase 2**
- Online Deep Research task + source recency checks.
- Stability bonus (multi-run variance analysis).
- HTML report (charts via static JS).

**Phase 3**
- Add provider API adapter for Copilot (if available) or UI automation runner (Playwright).
- Expand task library (code comprehension, table synthesis, policy extraction).
- Human-in-the-loop adjudication UI for borderline cases.

---

## Acceptance Criteria (for this spec)

- A new engineer can clone the repo, run `bench.py run` against ChatGPT and Manual Copilot, and produce a `report.md` with per-task scores.
- Adding a new offline test requires **no Python changes** (just YAML + answer key) for existing evaluators.
- Scores are deterministic for ChatGPT (temp=0, seed if supported) and repeatable for manual Copilot given the same pasted output.

---

## Appendix — Example Prompts (Embedded in Tests)

- Offline tasks embed the JSON output schema verbatim and instruct “Do not browse the web.”
- Online task instructs “Browsing required; provide sources with dates; if browsing disabled, mark sources as placeholders and state limitations.”

---

*End of spec.*
