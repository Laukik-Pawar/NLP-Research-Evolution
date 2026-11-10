
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
START_YEAR = 2010
END_YEAR = 2025
TARGET_PAPERS = 300


def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def sentence_split(abstract):
    text = re.sub(r"\s+", " ", str(abstract)).strip()

    # Lightweight sentence segmentation.
    # May require manual correction for abbreviations.
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.split()) >= 8
    ]


def load_pilot_titles(path):
    pilot = pd.read_csv(path, keep_default_na=False)

    if "title" not in pilot.columns:
        raise ValueError(
            "Pilot file must contain a title column."
        )

    return set(pilot["title"].map(normalize))


def allocate_quotas(year_counts, target):
    years = sorted(year_counts)

    if sum(year_counts.values()) < target:
        raise ValueError("Not enough eligible papers.")

    # Balanced quotas, distributing remainder by year.
    base, remainder = divmod(target, len(years))

    quotas = {
        year: min(base + (i < remainder), year_counts[year])
        for i, year in enumerate(years)
    }

    remaining = target - sum(quotas.values())

    # Redistribute if any year has insufficient papers.
    while remaining:
        available = [
            y for y in years
            if quotas[y] < year_counts[y]
        ]

        if not available:
            raise ValueError("Cannot allocate target sample.")

        for year in available:
            if remaining == 0:
                break

            quotas[year] += 1
            remaining -= 1

    return quotas


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        default="results/cleaned_corpus.csv"
    )

    parser.add_argument(
        "--pilot",
        default="results/phase3/topics_reviewed.csv"
    )

    parser.add_argument(
        "--out",
        default="results/phase6"
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_csv(
        args.corpus,
        keep_default_na=False,
        low_memory=False
    )

    required = {
        "year", "title", "abstract",
        "language", "venue", "source"
    }

    missing = required - set(corpus.columns)

    if missing:
        raise ValueError(
            f"Missing corpus columns: {missing}"
        )

    corpus["year"] = pd.to_numeric(
        corpus["year"], errors="coerce"
    )

    corpus = corpus[
        corpus["year"].between(
            START_YEAR, END_YEAR
        )
    ].copy()

    corpus["year"] = corpus["year"].astype(int)

    corpus["title_key"] = corpus["title"].map(normalize)

    pilot_titles = load_pilot_titles(args.pilot)

    corpus = corpus[
        ~corpus["title_key"].isin(pilot_titles)
    ].copy()

    # Remove duplicate titles.
    corpus = corpus.drop_duplicates(
        subset=["title_key"]
    )

    # Keep papers with at least two usable sentences.
    corpus["sentences"] = corpus["abstract"].map(
        sentence_split
    )

    corpus = corpus[
        corpus["sentences"].map(len) >= 2
    ].copy()

    year_counts = (
        corpus.groupby("year")
        .size()
        .to_dict()
    )

    expected_years = set(
        range(START_YEAR, END_YEAR + 1)
    )

    if not expected_years.issubset(year_counts):
        raise ValueError(
            "One or more years have no eligible papers."
        )

    quotas = allocate_quotas(
        year_counts, TARGET_PAPERS
    )

    rng = np.random.default_rng(SEED)

    selected = []

    for year in sorted(quotas):
        subset = corpus[
            corpus["year"] == year
        ]

        indices = rng.choice(
            subset.index.to_numpy(),
            size=quotas[year],
            replace=False
        )

        selected.append(corpus.loc[indices])

    papers = pd.concat(
        selected, ignore_index=True
    )

    papers = papers.sort_values(
        ["year", "title_key"]
    ).reset_index(drop=True)

    paper_rows = []
    sentence_rows = []

    for i, row in papers.iterrows():
        paper_id = f"P6_{i + 1:04d}"

        paper_rows.append({
            "paper_id": paper_id,
            "year": row["year"],
            "title": row["title"],
            "abstract": row["abstract"],
            "language": row["language"],
            "venue": row["venue"],
            "source": row["source"],
            "topic_labels": ""
        })

        sentences = row["sentences"]

        selected_sentence_indices = rng.choice(
            len(sentences),
            size=2,
            replace=False
        )

        for j, index in enumerate(
            sorted(selected_sentence_indices)
        ):
            sentence_rows.append({
                "sentence_id": f"{paper_id}_S{j + 1}",
                "paper_id": paper_id,
                "year": row["year"],
                "sentence": sentences[index],
                "claim_label": ""
            })

    paper_df = pd.DataFrame(paper_rows)
    sentence_df = pd.DataFrame(sentence_rows)

    paper_df.to_csv(
        out / "papers_to_annotate.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sentence_df.to_csv(
        out / "sentences_to_annotate.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "seed": SEED,
        "year_range": [
            START_YEAR, END_YEAR
        ],
        "papers": len(paper_df),
        "sentences": len(sentence_df),
        "pilot_papers_excluded": len(pilot_titles),
        "papers_per_year": {
            str(k): int(v)
            for k, v in paper_df.groupby("year").size().items()
        },
        "annotation_status": "Unlabeled",
        "sampling_method": (
            "Balanced stratified random sampling by year, "
            "excluding pilot titles, requiring at least "
            "two eligible abstract sentences."
        )
    }

    (out / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved files to:", out.resolve())


if __name__ == "__main__":
    main()
