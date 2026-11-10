# How the Language of NLP Research Changes Over Time

**CS 584 — Natural Language Processing | Stevens Institute of Technology**

A reproducible research pipeline for studying how NLP paper topics and scientific-claim language evolve over time. The project combines ACL Anthology metadata, keyword-based longitudinal analysis, multi-label topic classification, sentence-level claim-strength classification, local Qwen inference, Gemini API inference, and venue-sensitivity checks.

> **Project status:** Midterm experiments completed. Additional annotation, matched-baseline evaluation, and contextual topic–claim analysis are planned. The commands below distinguish verified setup instructions from experiment entry points that must be checked against the actual scripts before use.

## 1. What the project does

The pipeline answers four questions:

1. Which NLP research terms become more or less common over time?
2. How do lexical hedging and strong-claim cues change across publication periods?
3. How accurately do baseline classifiers and LLMs assign paper topics and claim-strength labels?
4. Are the main temporal trends directionally stable across venue families present in both periods?

**Important:** Keyword prevalence is a descriptive proxy. A keyword match does not necessarily prove that a paper belongs to a topic or that a scientific claim is unusually strong. We do not infer causation from model release dates.

## 2. Verified midterm findings

- **93,261** ACL Anthology XML-derived records initially processed.
- **67,288** cleaned papers from 2010–2025.
- **64,583** papers in the primary 2016–2025 analysis.
- **45,109** papers in the shared-venue-family sensitivity subset.
- **64** reviewed papers for topic evaluation and **128** reviewed sentences for claim evaluation.

### Model evaluation

All metrics below were independently recalculated from the six saved prediction CSV files. Failed or invalid outputs were included in scoring.

| Model | Topic micro-F1 | Topic macro-F1 | Claim accuracy | Claim macro-F1 |
|---|---:|---:|---:|---:|
| Qwen 2.5 3B zero-shot | 0.3926 | 0.2701 | 0.8359 | **0.6481** |
| Qwen 2.5 3B few-shot (3 examples) | 0.4654 | 0.3898 | **0.8828** | 0.6113 |
| Gemini 3.1 Flash Lite zero-shot | **0.6239** | **0.6471** | 0.6250 | 0.4203 |

The claim reference labels are imbalanced: **113 Neutral, 13 Strong, and 2 Hedged**. Class-level conclusions should therefore be interpreted cautiously.

### Temporal comparison

Changes below are **absolute percentage-point differences** in lexical indicator prevalence, comparing **2016–2022** with **2023–2025**:

| Indicator | Change (percentage points) |
|---|---:|
| Language modeling | +40.26 |
| Evaluation and benchmarks | +11.98 |
| Reasoning | +10.50 |
| Multimodal NLP | +4.78 |
| Machine translation | −4.21 |
| Strong-claim cues | +10.32 |
| Hedging cues | +7.25 |

The 2023 split is a descriptive comparison motivated by the adoption of conversational/generative LLMs following late-2022 releases, **not** a causal impact study. Directional patterns for selected indicators also persisted in the shared-venue-family subset.

## 3. Repository layout

Your checkout should contain the project code and any permitted inputs. The layout below describes the known project paths and a recommended organization; not every folder is necessarily included in a fresh GitHub clone.

```text
NLP-Research-Evolution/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                 # optional template; no real secrets
├── scripts/                     # corpus, baselines, inference, analysis, verification
├── data/                        # local input files, if present (often gitignored)
├── results/
│   ├── cleaned_corpus.csv
│   ├── phase3/
│   ├── phase6/
│   │   ├── independent_annotation/
│   │   │   ├── annotation_manifest.json
│   │   │   ├── papers_for_review_annotated.csv
│   │   │   └── sentences_for_review_annotated.csv
│   │   ├── qwen_zero/zero-shot/qwen2.5_3b/
│   │   ├── qwen_few/few-shot/qwen2.5_3b/
│   │   └── gemini_zero/gemini-3.1-flash-lite/
│   ├── phase7/period_comparison.csv
│   ├── phase8/venue_sensitivity.csv
│   └── phase9/independently_verified_metrics.csv
└── reports/                     # optional PDF/LaTeX report
```

> The original corpus and reviewed annotations are not automatically downloadable from this README. A teammate needs access to the authorized input files or must run the project's actual corpus-building scripts. Do not commit restricted data or any private API keys.

## 4. Requirements

- **Python:** use a compatible Python 3 version (Python **3.10 or 3.11 is a reasonable starting point**, but the full repository has not yet been compatibility-tested).
- **Git** for cloning.
- Internet access for installing packages and, if used, the Gemini API.
- Sufficient disk space and memory for corpus processing. **Local Qwen 2.5 3B** needs substantially more RAM/VRAM and model storage than CSV-based analysis; a GPU is helpful but hardware requirements depend on precision and loading method.
- A **Gemini API key** only if running Gemini inference. Existing saved results can be analyzed without one.

### Windows PowerShell — first-time setup

```powershell
git clone https://github.com/Laukik-Pawar/NLP-Research-Evolution.git
cd NLP-Research-Evolution

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If `py -3.11` is unavailable, use `python -m venv .venv` with your installed compatible Python.

If PowerShell blocks activation, **without changing machine-wide policy**, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Or skip activation and invoke ` .\.venv\Scripts\python.exe ` directly.

### macOS / Linux

```bash
git clone https://github.com/Laukik-Pawar/NLP-Research-Evolution.git
cd NLP-Research-Evolution

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

**Note:** The `requirements.txt` must be validated against the actual imports. GPU-enabled PyTorch may require a platform-specific installation; follow the [official PyTorch installer](https://pytorch.org/get-started/locally/) if needed.

## 5. Environment configuration (Gemini only)

Create a local `.env` file in the repository root if the inference script loads environment variables using `python-dotenv`:

```dotenv
GEMINI_API_KEY=replace_with_your_own_key
```

For PowerShell scripts that read the environment directly, you can instead set the variable for the current terminal:

```powershell
$env:GEMINI_API_KEY = "your_key_here"
```

For macOS/Linux:

```bash
export GEMINI_API_KEY="your_key_here"
```

**Never commit `.env`, API keys, access tokens, or raw API responses containing sensitive information.** Check the Gemini script's exact variable name before running; `GEMINI_API_KEY` is the recommended convention, not a verified code-level requirement here. API calls may incur charges or hit rate limits, and the originally used model name may not remain available.

## 6. Obtain the data and annotations

A clean clone will only reproduce results if its input files are present. Ask the project team for the approved dataset bundle or run the actual corpus-extraction/preprocessing scripts against authorized ACL Anthology source material.

Expected primary files after preprocessing:

```text
results/cleaned_corpus.csv
results/phase6/independent_annotation/annotation_manifest.json
results/phase6/independent_annotation/papers_for_review_annotated.csv
results/phase6/independent_annotation/sentences_for_review_annotated.csv
```

The cleaned CSV includes paper-level metadata such as year, title, abstract, language, venue, and source. The reviewed files are needed for the **64-paper/128-sentence** evaluation. Do not describe them as independent double-annotator ground truth; no agreement study has been verified.

### Check that required files exist (PowerShell)

```powershell
@(
  "results/cleaned_corpus.csv",
  "results/phase6/independent_annotation/annotation_manifest.json",
  "results/phase6/independent_annotation/papers_for_review_annotated.csv",
  "results/phase6/independent_annotation/sentences_for_review_annotated.csv"
) | ForEach-Object { "$_ : $(Test-Path $_)" }
```

## 7. Run the project in order

There are **two workflows**:

### Workflow A — Inspect and verify existing results (recommended for teammates)

This is the fastest path if your clone includes the saved result CSVs and reviewed reference files. **It does not require an LLM API key or rerunning Qwen.**

1. Complete the Python setup in Section 4.
2. Obtain the permitted results and annotation files described in Section 6.
3. Inspect the project scripts and run the verified evaluation entry point below.

The following command is based on the existing Phase 9 script name from the original working project:

```powershell
python .\scripts\phase9_recalculate_metrics.py
```

Expected success signal:

```text
Qwen zero-shot ... MATCHES PREVIOUS: True
Qwen few-shot  ... MATCHES PREVIOUS: True
Gemini zero-shot ... MATCHES PREVIOUS: True
All metrics match: True
```

It reads **six prediction CSV files** from these locations:

```text
results/phase6/qwen_zero/zero-shot/qwen2.5_3b/qwen_zero_topic_predictions.csv
results/phase6/qwen_zero/zero-shot/qwen2.5_3b/qwen_zero_claim_predictions.csv
results/phase6/qwen_few/few-shot/qwen2.5_3b/qwen_few_topic_predictions.csv
results/phase6/qwen_few/few-shot/qwen2.5_3b/qwen_few_claim_predictions.csv
results/phase6/gemini_zero/gemini-3.1-flash-lite/gemini_zero_topic_predictions.csv
results/phase6/gemini_zero/gemini-3.1-flash-lite/gemini_zero_claim_predictions.csv
```

**Optional integrity check** — verify the reviewed files against the frozen manifest (run from repository root):

```powershell
@'
from pathlib import Path
import hashlib, json
manifest = json.loads(Path("results/phase6/independent_annotation/annotation_manifest.json").read_text(encoding="utf-8"))
for name, item in manifest["files"].items():
    path = Path(item["path"])
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{name}: {'PASS' if actual == item['sha256'] else 'FAIL'}")
'@ | python -
```

Expected results when the same frozen files are supplied:

```text
papers: PASS
sentences: PASS
```

### Workflow B — Regenerate the experiments from scratch (advanced)

To fully reproduce the study, run the project's **actual** scripts in this logical order:

| Stage | Purpose | Inputs | Outputs |
|---|---|---|---|
| 1. Corpus extraction | Parse ACL Anthology XML and retain metadata | ACL source files | Raw records |
| 2. Cleaning | Deduplicate, remove missing/short abstracts and non-English records | Raw records | `results/cleaned_corpus.csv` |
| 3. Pilot and annotation | Build topic taxonomy, extract claim sentences, review labels | Cleaned corpus | Annotation CSVs |
| 4. Baselines | TF-IDF + one-vs-rest logistic regression; rule-based claims | Labeled samples | Baseline metrics |
| 5. Qwen zero-/few-shot | Run local Qwen 2.5 3B on fixed evaluation inputs | Reviewed set, model weights | Prediction CSVs |
| 6. Gemini zero-shot | Run Gemini API on fixed evaluation inputs | Reviewed set, API key | Prediction CSVs |
| 7. Temporal analysis | Calculate lexical indicators by period | Cleaned corpus | `results/phase7/period_comparison.csv` |
| 8. Venue sensitivity | Restrict to shared venue families | Corpus and venue metadata | `results/phase8/venue_sensitivity.csv` |
| 9. Verification | Recalculate and compare scores | Six prediction CSVs | `results/phase9/independently_verified_metrics.csv` |

**The precise commands for stages 1–8 are intentionally not invented.** Check available entry points with:

```powershell
Get-ChildItem .\scripts -Filter "*.py" | Select-Object Name
```

For any likely script, inspect its CLI options before running:

```powershell
python .\scripts\YOUR_SCRIPT_NAME.py --help
```

If a script has no `--help`, open it and inspect its `if __name__ == "__main__":` block, input paths, and any API calls. **Do not run the full pipeline blindly:** it may overwrite saved outputs, require large local models, or incur API charges. Once the actual script names and CLI options are audited, replace this section with copy-paste commands for every stage.

## 8. Reproducibility and evaluation rules

- **Topic task:** 15 possible labels; multiple labels allowed, but `Other` is mutually exclusive with substantive labels.
- **Claim task:** `Hedged`, `Neutral`, `Strong`.
- **Invalid topic outputs:** count as an empty prediction set; do not remove the paper.
- **Missing/invalid claim outputs:** count as incorrect; do not remove the sentence.
- **Metrics:** topic micro-F1 and macro-F1; claim accuracy, per-class F1, and macro-F1.
- **Fair comparisons:** the three LLM conditions use the same reviewed 64/128 evaluation set. Pilot baseline scores were obtained on a **different** sample and are **not** paired performance comparisons.
- **Integrity:** the frozen annotation manifest records SHA-256 hashes for the reviewed reference CSVs.

## 9. Common problems

| Problem | What to check |
|---|---|
| `ModuleNotFoundError` | Activate the right virtual environment; run `python -m pip install -r requirements.txt` |
| PowerShell blocks `.venv` activation | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or invoke `.venv\Scripts\python.exe` |
| Missing CSV / `FileNotFoundError` | Run commands from repository root; confirm the permitted data/results bundle is present |
| Qwen is very slow / out of memory | Check available RAM/VRAM, model precision, device selection, and actual inference script |
| Gemini `429` or `503` | Check quota, model availability, and transient service errors; avoid repeated costly retries |
| Scores do not match | Confirm identical reference files, prediction paths, taxonomy, failure handling, and sample IDs |
| SHA-256 check fails | Do not overwrite the manifest to force a pass; identify which reviewed file changed |

## 10. Limitations and next steps

The 2010–2015 coverage is uneven, so the main temporal analysis focuses on 2016–2025. Venue-family filtering is a sensitivity check, **not** complete control of venue composition. Keyword matches are proxies, and exploratory p-values are not corrected for multiple testing. The reviewed claim set is small and strongly imbalanced. API errors affect the observed Gemini scores.

Planned work includes expanding reviewed annotations, matched-test-set baseline comparisons, contextual topic–claim associations, alternative time windows, within-venue analysis, and error audits.



## 12. References and data source

- [ACL Anthology](https://aclanthology.org/)
- [Scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)

## 13. License and sharing

No license is implied by this README. Add a `LICENSE` file only after the team agrees on the terms. Confirm that shared datasets, annotations, and report materials can be redistributed before committing them to a public repository.

---

**Maintainer note:** This README is grounded in the completed midterm work and known file paths. To turn it into a fully tested one-command-from-scratch guide, audit the actual `scripts/` folder, input-data distribution, and `requirements.txt` against a fresh environment.
