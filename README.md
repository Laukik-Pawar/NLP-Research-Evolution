# How the Language of NLP Research Changes Over Time

**CS 584 — Natural Language Processing | Stevens Institute of Technology | Fall 2026**

A reproducible research project investigating how NLP publication vocabulary and scientific claim-language cues changed between 2016 and 2025. We combine lexical trend analysis with multi-label topic classification, sentence-level claim-strength classification, and venue-family sensitivity analysis.

**Authors:** Laukik Pawar, Shree Shingre, Mandar Bavdane, Atharva Raut.

## Research questions

1. Which NLP research-topic expressions became more or less common?
2. How did hedging and strong-claim lexical cues change over time?
3. How do TF–IDF/rule baselines and instruction-tuned LLMs perform on topic and claim classification?
4. Do temporal trends retain their direction when restricted to venue families represented in both periods?

## Dataset and methodology

The corpus was derived from [ACL Anthology](https://aclanthology.org/) XML records. Cleaning removed duplicate records, missing/short abstracts, and non-English records **sequentially**. Of 93,261 source records, 67,288 remained; 64,583 from 2016–2025 formed the primary longitudinal sample. The 2010–2015 years were excluded from primary trend conclusions due to uneven venue coverage.

- **Topic task:** 15-label multi-label classification of titles and abstracts (including an exclusive `Other` label).
- **Claim task:** `Hedged`, `Neutral`, or `Strong` classification of abstract sentences.
- **Temporal task:** paper-level regex prevalence of topic terms in title + abstract, and rhetorical cues in abstracts. The period comparison is **2016–2022 vs. 2023–2025**. Changes are **percentage points**, not relative percentage changes.
- **Evaluation:** reviewed reference labels for 64 papers and 128 sentences; failures count against scores. The claim reference distribution is 113 Neutral, 13 Strong, and 2 Hedged.
- **Robustness:** 45,109-paper shared-venue-family subset. This does not fully control for changing venue weights.

> Keyword prevalence is a lexical proxy, **not** a contextual topic assignment, a verified measure of claim strength, or evidence of causation.

## Results

### Temporal prevalence changes

| Indicator | Change (percentage points) |
| --- | ---: |
| Language modeling | +40.26 |
| Evaluation and benchmarks | +11.98 |
| Reasoning | +10.50 |
| Multimodal NLP | +4.78 |
| Machine translation | -4.21 |
| Strong-claim cues | +10.32 |
| Hedging cues | +7.25 |

### Model evaluation (reviewed 64-paper / 128-sentence set)

| Model | Topic micro-F1 | Topic macro-F1 | Claim accuracy | Claim macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5 3B zero-shot | 0.3926 | 0.2701 | 0.8359 | **0.6481** |
| Qwen2.5 3B few-shot (3 examples) | 0.4654 | 0.3898 | **0.8828** | 0.6113 |
| Gemini 3.1 Flash Lite zero-shot | **0.6239** | **0.6471** | 0.6250 | 0.4203 |

Topic failures: 12, 17, and 4 of 64, respectively. Claim failures: 0, 0, and 10 of 128. The pilot baselines were evaluated on a **different sample**, so their metrics are not directly comparable to the table above.

## Repository layout

This is the **recommended layout**; adapt it to your existing local project rather than renaming working scripts blindly.

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── scripts/                   # Existing phase1–phase9 scripts
├── results/
│   ├── phase2/                # Aggregated lexical trends
│   ├── phase7/                # Period comparison
│   ├── phase8/                # Venue sensitivity
│   └── phase9/                # Recomputed metrics
├── reports/                   # Midterm report (PDF / LaTeX)
└── data/                      # Local/raw data (not committed by default)
```

## Setup (Windows PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks virtual-environment activation, use `Set-ExecutionPolicy -Scope Process Bypass` for the current shell, or run `.\.venv\Scripts\python.exe` directly.

**Optional model dependencies:** For local Qwen experiments, install and run [Ollama](https://ollama.com/) and pull the specific Qwen2.5 3B model configured in your scripts. For Gemini experiments, configure the appropriate API key in `.env` and confirm your scripts read it. Do not commit `.env` or API responses containing credentials.

### Reproduce experiments

Run your **existing** scripts in phase order after verifying their filenames and CLI arguments:

1. Corpus extraction and cleaning.
2. Pilot labels and baseline classifiers.
3. Qwen/Gemini inference (requires model runtime/API; saved outputs can avoid repeat calls).
4. Temporal keyword trends and venue-family sensitivity.
5. Independent metrics verification and figure generation.

For the verification script used during the project, the command is:

```powershell
python .\scripts\phase9_recalculate_metrics.py
```

This command requires the six saved prediction CSVs at the paths expected by that script. If those files are not distributed, the evaluation cannot be rerun from the public repository alone. Document any alternative paths in the code before publishing.

## Reproducibility and limitations

- Independently recalculated scores matched prior results within `1e-5`.
- Frozen reviewed annotation files were verified using SHA-256 checksums.
- The small, imbalanced claim set limits generalization, especially for `Hedged`.
- Gemini API failures were included in evaluation rather than discarded.
- Early-year ACL coverage was uneven; 2016–2025 is the primary trend window.
- Regex prevalence is not contextual classification; shared-venue analysis does not equalize venue composition.
- A dual-annotator agreement study has **not** been established.

## Future work

Expand reviewed annotations, compare all baselines and LLMs on the same fixed test set, analyze contextual topic–claim relationships, and strengthen venue/time-window controls.

## References

- [ACL Anthology](https://aclanthology.org/)
- [scikit-learn: Machine Learning in Python](https://jmlr.org/papers/v12/pedregosa11a.html)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)
- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)

## License and data

No license has been selected yet. Add a `LICENSE` file only after the team agrees on a license. Follow ACL Anthology and any source-data usage terms; avoid publishing raw source datasets or annotation files without checking permissions and privacy considerations.
