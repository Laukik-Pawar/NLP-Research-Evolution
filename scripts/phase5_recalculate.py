
"""Recalculate multi-label topic scores on all papers.

Invalid outputs count as empty predictions rather than being excluded.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


TOPICS = [
    "Machine translation",
    "Summarization",
    "Question answering",
    "Reasoning",
    "Dialogue systems",
    "Information extraction",
    "Sentiment and opinion",
    "Syntax and parsing",
    "Language modeling",
    "Multilingual / low-resource",
    "Fairness, bias and ethics",
    "Evaluation and benchmarks",
    "Multimodal NLP",
    "Speech and spoken language",
    "Other",
]


def labels(value):
    if pd.isna(value) or not str(value).strip():
        return []

    return [
        v.strip()
        for v in str(value).split(";")
        if v.strip()
    ]


def calculate(path):
    df = pd.read_csv(path, dtype={"paper_id": str})

    required = {
        "paper_id",
        "topic_labels",
        "predicted_labels",
        "error",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    if df.paper_id.duplicated().any():
        raise ValueError("Duplicate paper IDs")

    true = df.topic_labels.map(labels).tolist()

    bad = [
        (i, label)
        for i, row in enumerate(true)
        for label in row
        if label not in TOPICS
    ]

    if bad:
        raise ValueError(
            f"Unknown reference labels: {bad[:5]}"
        )

    failed = (
        df.error.notna()
        & df.error.astype(str).str.strip().ne("")
    )

    pred = [
        [] if invalid else labels(value)
        for invalid, value in zip(
            failed, df.predicted_labels
        )
    ]

    badpred = [
        (i, label)
        for i, row in enumerate(pred)
        for label in row
        if label not in TOPICS
    ]

    if badpred:
        raise ValueError(
            f"Unknown predicted labels in valid rows: {badpred[:5]}"
        )

    mlb = MultiLabelBinarizer(classes=TOPICS)

    Y = mlb.fit_transform(true)
    P = mlb.transform(pred)

    return {
        "papers": len(df),
        "valid": int((~failed).sum()),
        "failed": int(failed.sum()),
        "invalid_output_rate": round(
            float(failed.mean()), 6
        ),
        "topic_micro_f1": float(
            f1_score(
                Y, P,
                average="micro",
                zero_division=0
            )
        ),
        "topic_macro_f1": float(
            f1_score(
                Y, P,
                average="macro",
                zero_division=0
            )
        ),
        "paper_ids": set(df.paper_id),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--zero", required=True)
    ap.add_argument("--few", required=True)

    ap.add_argument(
        "--out",
        default=(
            "results/phase5/paired_comparison/"
            "corrected_metrics.json"
        ),
    )

    args = ap.parse_args()

    zero = calculate(args.zero)
    few = calculate(args.few)

    if zero.pop("paper_ids") != few.pop("paper_ids"):
        raise ValueError(
            "Zero-shot and few-shot paper IDs do not match"
        )

    result = {
        "method": (
            "Invalid outputs count as empty label sets; "
            "all papers included."
        ),
        "zero_shot": zero,
        "few_shot": few,
    }

    dest = Path(args.out)
    dest.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    dest.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(result, indent=2))
    print("Saved:", dest)


if __name__ == "__main__":
    main()
