
from pathlib import Path
import json
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results")
OUT = ROOT / "phase9"
OUT.mkdir(parents=True, exist_ok=True)

checks = []
warnings = []

def check_file(path, label):
    path = Path(path)
    exists = path.exists()
    checks.append({
        "check": label,
        "status": "PASS" if exists else "MISSING",
        "path": str(path)
    })
    return exists

print("\n=== PHASE 9: FINAL VERIFICATION ===")

# --------------------------------------------------
# 1. Required input files
# --------------------------------------------------

required = {
    "Cleaned corpus": ROOT / "cleaned_corpus.csv",
    "Reviewed topic annotations":
        ROOT / "phase6/independent_annotation/papers_for_review_annotated.csv",
    "Reviewed claim annotations":
        ROOT / "phase6/independent_annotation/sentences_for_review_annotated.csv",
    "Annotation manifest":
        ROOT / "phase6/independent_annotation/annotation_manifest.json",
    "Temporal claim trends":
        ROOT / "phase7/claim_cue_trends.csv",
    "Period comparison":
        ROOT / "phase7/period_comparison.csv",
    "Venue sensitivity":
        ROOT / "phase8/venue_sensitivity.csv",
}

for label, path in required.items():
    check_file(path, label)

# --------------------------------------------------
# 2. Corpus verification
# --------------------------------------------------

corpus_path = ROOT / "cleaned_corpus.csv"

if corpus_path.exists():
    corpus = pd.read_csv(
        corpus_path,
        keep_default_na=False,
        low_memory=False
    )

    corpus["year"] = pd.to_numeric(
        corpus["year"], errors="coerce"
    )

    total = len(corpus)
    main = corpus[
        corpus["year"].between(2016, 2025)
    ]

    print("\n=== CORPUS ===")
    print("Total cleaned papers:", total)
    print("Primary period papers:", len(main))

    if total != 67288:
        warnings.append(
            f"Corpus count differs from expected 67288: {total}"
        )

    if len(main) != 64583:
        warnings.append(
            "Primary-period count differs from expected 64583"
        )

# --------------------------------------------------
# 3. Reference annotation counts
# --------------------------------------------------

topic_path = required["Reviewed topic annotations"]
claim_path = required["Reviewed claim annotations"]

if topic_path.exists() and claim_path.exists():
    topics = pd.read_csv(topic_path)
    claims = pd.read_csv(claim_path)

    print("\n=== REVIEWED REFERENCE ANNOTATIONS ===")
    print("Topic papers:", len(topics))
    print("Claim sentences:", len(claims))

    if len(topics) != 64:
        warnings.append("Expected 64 reviewed topic papers")

    if len(claims) != 128:
        warnings.append("Expected 128 reviewed claim sentences")

# --------------------------------------------------
# 4. Phase 6 prediction file discovery
# --------------------------------------------------

print("\n=== PHASE 6 PREDICTION FILES ===")

for model in ["qwen_zero", "qwen_few", "gemini_zero"]:
    folder = ROOT / "phase6" / model

    if not folder.exists():
        warnings.append(f"Missing model folder: {folder}")
        print(model, "MISSING")
        continue

    files = sorted(folder.glob("*.csv"))

    print(f"\n{model}:")
    for file in files:
        print(" ", file.name)

    if not files:
        warnings.append(f"No prediction CSV files in {folder}")

# --------------------------------------------------
# 5. Load corrected comparison metrics
# --------------------------------------------------

# These are the previously verified fair-evaluation
# results, included as reference values. They are
# NOT independently recalculated in this script.

comparison = pd.DataFrame([
    {
        "model": "Qwen zero-shot",
        "topic_micro_f1": 0.392638,
        "topic_macro_f1": 0.270072,
        "claim_accuracy": 0.8359375,
        "claim_macro_f1": 0.648114,
        "topic_failures": 12,
        "claim_failures": 0
    },
    {
        "model": "Qwen few-shot",
        "topic_micro_f1": 0.465409,
        "topic_macro_f1": 0.389799,
        "claim_accuracy": 0.8828125,
        "claim_macro_f1": 0.611278,
        "topic_failures": 17,
        "claim_failures": 0
    },
    {
        "model": "Gemini zero-shot",
        "topic_micro_f1": 0.623853,
        "topic_macro_f1": 0.647091,
        "claim_accuracy": 0.625,
        "claim_macro_f1": 0.420255,
        "topic_failures": 4,
        "claim_failures": 10
    }
])

comparison.to_csv(
    OUT / "final_model_comparison.csv",
    index=False
)

print("\n=== CORRECTED MODEL COMPARISON ===")
print(comparison.round(4).to_string(index=False))

warnings.append(
    "Phase 6 metrics are reference values; "
    "automatic recomputation from predictions "
    "has not yet been performed."
)

# --------------------------------------------------
# 6. Verify venue sensitivity directions
# --------------------------------------------------

venue_path = ROOT / "phase8/venue_sensitivity.csv"

if venue_path.exists():
    venue = pd.read_csv(venue_path)

    pivot = venue.pivot(
        index="indicator",
        columns="sample",
        values="difference_pp"
    )

    pivot["same_direction"] = (
        pivot["Full corpus"] *
        pivot["Shared families"] > 0
    )

    print("\n=== VENUE SENSITIVITY ===")
    print(pivot.round(3).to_string())

    if not pivot["same_direction"].all():
        warnings.append(
            "At least one sensitivity trend changes direction"
        )

    pivot.to_csv(
        OUT / "venue_sensitivity_verified.csv"
    )

# --------------------------------------------------
# 7. Final temporal comparison figure
# --------------------------------------------------

period_path = ROOT / "phase7/period_comparison.csv"

if period_path.exists():
    period = pd.read_csv(period_path)

    selected = [
        "Language modeling",
        "Evaluation and benchmarks",
        "Reasoning",
        "Multimodal NLP",
        "Machine translation",
        "Strong-claim cues",
        "Hedging cues"
    ]

    plot = period[
        period["indicator"].isin(selected)
    ].copy()

    plot = plot.sort_values("difference_pp")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        plot["indicator"],
        plot["difference_pp"]
    )

    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Change in percentage points")
    ax.set_title(
        "NLP Research Language: 2023–2025 vs 2016–2022"
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "final_temporal_changes.png",
        dpi=250
    )
    plt.close(fig)

# --------------------------------------------------
# 8. Model comparison figure
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

x = range(len(comparison))
width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    comparison["topic_macro_f1"],
    width,
    label="Topic macro-F1"
)

ax.bar(
    [i + width / 2 for i in x],
    comparison["claim_macro_f1"],
    width,
    label="Claim macro-F1"
)

ax.set_xticks(list(x))
ax.set_xticklabels(comparison["model"])
ax.set_ylabel("Macro-F1")
ax.set_ylim(0, 1)
ax.set_title("Model Performance on Reviewed Reference Annotations")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUT / "final_model_comparison.png",
    dpi=250
)
plt.close(fig)

# --------------------------------------------------
# 9. Verification report
# --------------------------------------------------

report = {
    "checks": checks,
    "warnings": warnings,
    "limitations": [
        "2010–2015 venue coverage is uneven.",
        "Topic trends are keyword-based proxies.",
        "Claim trends measure lexical cues, not "
        "sentence-level claim-strength labels.",
        "Reviewed evaluation set is small.",
        "Claim reference labels are imbalanced.",
        "Venue-family sensitivity does not fully "
        "control for venue composition.",
        "Period comparison p-values are exploratory "
        "and unadjusted for multiple testing."
    ]
}

(OUT / "verification_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

pd.DataFrame(checks).to_csv(
    OUT / "verification_checks.csv",
    index=False
)

print("\n=== FINAL VERIFICATION SUMMARY ===")

for item in checks:
    print(f"{item['status']}: {item['check']}")

print("\nWarnings:", len(warnings))

for warning in warnings:
    print("WARNING:", warning)

print("\nOutputs saved to:", OUT.resolve())

