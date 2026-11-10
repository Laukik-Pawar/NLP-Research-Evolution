
from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path("results/cleaned_corpus.csv")
OUT = Path("results/phase7")
OUT.mkdir(parents=True, exist_ok=True)

START_YEAR = 2010
END_YEAR = 2025

TOPIC_PATTERNS = {
    "Machine translation": [
        r"\bmachine translation\b",
        r"\bneural machine translation\b"
    ],
    "Summarization": [
        r"\bsummarization\b",
        r"\bsummarisation\b"
    ],
    "Question answering": [
        r"\bquestion answering\b",
        r"\breading comprehension\b"
    ],
    "Reasoning": [
        r"\breasoning\b",
        r"\bchain.of.thought\b"
    ],
    "Dialogue systems": [
        r"\bdialogue systems?\b",
        r"\bdialog systems?\b",
        r"\bconversational agents?\b"
    ],
    "Information extraction": [
        r"\binformation extraction\b",
        r"\bnamed entity recognition\b",
        r"\brelation extraction\b"
    ],
    "Sentiment and opinion": [
        r"\bsentiment analysis\b",
        r"\bopinion mining\b",
        r"\bstance detection\b"
    ],
    "Syntax and parsing": [
        r"\bdependency parsing\b",
        r"\bsyntactic parsing\b",
        r"\bpart.of.speech tagging\b"
    ],
    "Language modeling": [
        r"\blanguage models?\b",
        r"\blarge language models?\b",
        r"\bpretrained language models?\b"
    ],
    "Multilingual / low-resource": [
        r"\bmultilingual\b",
        r"\bcross.lingual\b",
        r"\blow.resource\b"
    ],
    "Fairness, bias and ethics": [
        r"\bfairness\b",
        r"\bbias\b",
        r"\bethical\b"
    ],
    "Evaluation and benchmarks": [
        r"\bbenchmark(s|ing)?\b",
        r"\bevaluation framework\b"
    ],
    "Multimodal NLP": [
        r"\bmultimodal\b",
        r"\bvision.language\b",
        r"\bimage captioning\b"
    ],
    "Speech and spoken language": [
        r"\bspeech recognition\b",
        r"\bspoken language\b",
        r"\bspeech translation\b"
    ]
}

# "Other" is intentionally excluded:
# it cannot be measured using a reliable keyword pattern.

df = pd.read_csv(
    INPUT,
    keep_default_na=False,
    low_memory=False
)

df["year"] = pd.to_numeric(
    df["year"], errors="coerce"
)

df = df[
    df["year"].between(START_YEAR, END_YEAR)
].copy()

df["year"] = df["year"].astype(int)

df["text"] = (
    df["title"].astype(str)
    + " "
    + df["abstract"].astype(str)
)

counts = (
    df.groupby("year")
    .size()
    .rename("papers")
    .reset_index()
)

counts.to_csv(
    OUT / "publication_counts.csv",
    index=False
)

topic_rows = []

for topic, patterns in TOPIC_PATTERNS.items():
    combined = "(?:" + "|".join(patterns) + ")"

    matches = df["text"].str.contains(
        combined,
        case=False,
        regex=True,
        na=False
    )

    yearly = (
        df.assign(matched=matches)
        .groupby("year")
        .agg(
            papers=("matched", "size"),
            matched_papers=("matched", "sum")
        )
        .reset_index()
    )

    yearly["topic"] = topic

    yearly["prevalence_pct"] = (
        100 * yearly["matched_papers"]
        / yearly["papers"]
    )

    topic_rows.append(yearly)

trends = pd.concat(
    topic_rows, ignore_index=True
)

trends.to_csv(
    OUT / "topic_keyword_trends.csv",
    index=False
)

# Figure 1: Annual publication counts
plt.figure(figsize=(11, 5))
plt.plot(
    counts["year"],
    counts["papers"],
    marker="o"
)
plt.title("ACL Corpus: Annual Publication Counts")
plt.xlabel("Publication year")
plt.ylabel("Papers with usable abstracts")
plt.xticks(range(START_YEAR, END_YEAR + 1), rotation=45)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUT / "publication_trends.png",
    dpi=200
)
plt.close()

# Figure 2: Topic keyword prevalence
selected = [
    "Machine translation",
    "Question answering",
    "Reasoning",
    "Language modeling",
    "Multilingual / low-resource",
    "Multimodal NLP"
]

plt.figure(figsize=(12, 6))

for topic in selected:
    part = trends[trends["topic"] == topic]
    plt.plot(
        part["year"],
        part["prevalence_pct"],
        marker="o",
        label=topic
    )

plt.title("NLP Topic Keyword Prevalence (2010–2025)")
plt.xlabel("Publication year")
plt.ylabel("Papers matching topic keywords (%)")
plt.xticks(range(START_YEAR, END_YEAR + 1), rotation=45)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(
    OUT / "topic_keyword_trends.png",
    dpi=200
)
plt.close()

summary = {
    "start_year": START_YEAR,
    "end_year": END_YEAR,
    "papers_analyzed": int(len(df)),
    "years": int(df["year"].nunique()),
    "topics_tracked": len(TOPIC_PATTERNS),
    "method": "Title and abstract keyword matching"
}

(OUT / "phase7_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== PHASE 7.1 RESULTS ===")
print(json.dumps(summary, indent=2))

print("\nPublication counts:")
print(counts.to_string(index=False))

print("\nSelected topic prevalence (%):")
pivot = trends[
    trends["topic"].isin(selected)
].pivot(
    index="year",
    columns="topic",
    values="prevalence_pct"
)

print(pivot.round(2).to_string())

print("\nSaved outputs to:", OUT.resolve())

