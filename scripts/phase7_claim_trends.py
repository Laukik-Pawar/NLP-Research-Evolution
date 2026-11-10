
from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("results/cleaned_corpus.csv")
OUT = Path("results/phase7")
OUT.mkdir(parents=True, exist_ok=True)

HEDGED = [
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bsuggest(?:s|ed)?\b",
    r"\bpotential(?:ly)?\b",
    r"\bappear(?:s|ed)?\b",
    r"\bpossibly\b",
    r"\bperhaps\b"
]

STRONG = [
    r"\bsignificantly\b",
    r"\bsubstantially\b",
    r"\bclearly\b",
    r"\bdefinitively\b",
    r"\boutperform(?:s|ed)?\b",
    r"\bstate.of.the.art\b",
    r"\bsuperior\b",
    r"\bdemonstrate(?:s|d)?\b"
]

df = pd.read_csv(
    INPUT,
    keep_default_na=False,
    low_memory=False
)

df["year"] = pd.to_numeric(
    df["year"], errors="coerce"
)

df = df[
    df["year"].between(2010, 2025)
].copy()

df["year"] = df["year"].astype(int)

# Analyze abstract text, not paper titles.
df["text"] = df["abstract"].astype(str)

def has_pattern(text, patterns):
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )

df["hedged_cue"] = df["text"].apply(
    lambda x: has_pattern(x, HEDGED)
)

df["strong_cue"] = df["text"].apply(
    lambda x: has_pattern(x, STRONG)
)

yearly = (
    df.groupby("year")
    .agg(
        papers=("text", "size"),
        hedged_papers=("hedged_cue", "sum"),
        strong_papers=("strong_cue", "sum")
    )
    .reset_index()
)

yearly["hedged_pct"] = (
    yearly["hedged_papers"] / yearly["papers"] * 100
)

yearly["strong_pct"] = (
    yearly["strong_papers"] / yearly["papers"] * 100
)

yearly.to_csv(
    OUT / "claim_cue_trends.csv",
    index=False
)

plt.figure(figsize=(11, 6))

plt.plot(
    yearly["year"],
    yearly["hedged_pct"],
    marker="o",
    label="Hedging cues"
)

plt.plot(
    yearly["year"],
    yearly["strong_pct"],
    marker="o",
    label="Strong-claim cues"
)

plt.title("Claim-Strength Cue Prevalence in NLP Abstracts")
plt.xlabel("Publication year")
plt.ylabel("Abstracts containing at least one cue (%)")
plt.xticks(range(2010, 2026), rotation=45)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

plt.savefig(
    OUT / "claim_cue_trends.png",
    dpi=200
)
plt.close()

summary = {
    "papers_analyzed": len(df),
    "years": df["year"].nunique(),
    "method": "Lexical hedging and strong-claim cues",
    "unit": "abstract",
    "limitation": (
        "Cue prevalence is not equivalent to "
        "sentence-level claim-strength classification."
    )
}

(OUT / "claim_trends_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n=== PHASE 7.2 RESULTS ===")
print(yearly.round(2).to_string(index=False))
print("\nSaved to:", OUT.resolve())
