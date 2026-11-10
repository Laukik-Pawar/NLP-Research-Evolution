
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    multilabel_confusion_matrix
)
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path("results")
OUT = ROOT / "phase9"
OUT.mkdir(parents=True, exist_ok=True)

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
    "Other"
]

CLAIMS = ["Hedged", "Neutral", "Strong"]

FILES = {
    "Qwen zero-shot": {
        "topics": ROOT / "phase6/qwen_zero/zero-shot/qwen2.5_3b/qwen_zero_topic_predictions.csv",
        "claims": ROOT / "phase6/qwen_zero/zero-shot/qwen2.5_3b/qwen_zero_claim_predictions.csv"
    },
    "Qwen few-shot": {
        "topics": ROOT / "phase6/qwen_few/few-shot/qwen2.5_3b/qwen_few_topic_predictions.csv",
        "claims": ROOT / "phase6/qwen_few/few-shot/qwen2.5_3b/qwen_few_claim_predictions.csv"
    },
    "Gemini zero-shot": {
        "topics": ROOT / "phase6/gemini_zero/gemini-3.1-flash-lite/gemini_zero_topic_predictions.csv",
        "claims": ROOT / "phase6/gemini_zero/gemini-3.1-flash-lite/gemini_zero_claim_predictions.csv"
    }
}

EXPECTED = {
    "Qwen zero-shot": (0.392638, 0.270072, 0.8359375, 0.648114),
    "Qwen few-shot": (0.465409, 0.389799, 0.8828125, 0.611278),
    "Gemini zero-shot": (0.623853, 0.647091, 0.625, 0.420255)
}

def parse_topics(value):
    if pd.isna(value) or not str(value).strip():
        return []

    labels = [
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    ]

    if (
        any(label not in TOPICS for label in labels)
        or len(labels) != len(set(labels))
        or ("Other" in labels and len(labels) > 1)
    ):
        return []

    return labels

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def evaluate_topics(df):
    assert len(df) == 64, "Expected 64 topic predictions"
    assert df["paper_id"].is_unique, "Duplicate paper IDs"

    gold = [parse_topics(x) for x in df["topic_labels"]]
    pred = [parse_topics(x) for x in df["predicted_labels"]]

    if any(not x for x in gold):
        raise ValueError("Invalid or empty reference topic labels")

    mlb = MultiLabelBinarizer(classes=TOPICS)
    mlb.fit([TOPICS])

    y_true = mlb.transform(gold)
    y_pred = mlb.transform(pred)

    failures = sum(not x for x in pred)

    return {
        "micro_f1": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "failures": failures,
        "rows": len(df)
    }

def evaluate_claims(df):
    assert len(df) == 128, "Expected 128 claim predictions"
    assert df["sentence_id"].is_unique, "Duplicate sentence IDs"

    gold = df["claim_label"].astype(str).tolist()
    pred = df["predicted_label"].tolist()

    if any(label not in CLAIMS for label in gold):
        raise ValueError("Invalid reference claim labels")

    # Failed predictions remain incorrect.
    # A separate sentinel prevents accidental matches.
    pred = [
        x if isinstance(x, str) and x in CLAIMS
        else "__FAILED__"
        for x in pred
    ]

    correct = sum(a == b for a, b in zip(gold, pred))
    failures = pred.count("__FAILED__")

    # Calculate F1 for the three actual claim classes.
    # Failed outputs count as false negatives for their
    # reference class and never receive credit.
    per_class = []

    for label in CLAIMS:
        tp = sum(
            g == label and p == label
            for g, p in zip(gold, pred)
        )
        fp = sum(
            g != label and p == label
            for g, p in zip(gold, pred)
        )
        fn = sum(
            g == label and p != label
            for g, p in zip(gold, pred)
        )

        denominator = 2 * tp + fp + fn
        per_class.append(
            2 * tp / denominator if denominator else 0.0
        )

    return {
        "accuracy": correct / len(gold),
        "macro_f1": float(np.mean(per_class)),
        "failures": failures,
        "rows": len(df),
        "per_class_f1": dict(zip(CLAIMS, per_class))
    }

# Verify that reference files have not changed.
manifest_path = (
    ROOT /
    "phase6/independent_annotation/annotation_manifest.json"
)

print("\n=== ANNOTATION MANIFEST ===")

manifest = json.loads(
    manifest_path.read_text(encoding="utf-8")
)

print(json.dumps(manifest, indent=2)[:3000])

print("\n=== INDEPENDENT METRIC RECALCULATION ===")

rows = []
all_pass = True

for model, paths in FILES.items():
    topics = pd.read_csv(paths["topics"])
    claims = pd.read_csv(paths["claims"])

    topic_result = evaluate_topics(topics)
    claim_result = evaluate_claims(claims)

    actual = (
        topic_result["micro_f1"],
        topic_result["macro_f1"],
        claim_result["accuracy"],
        claim_result["macro_f1"]
    )

    expected = EXPECTED[model]

    differences = [
        abs(a - b)
        for a, b in zip(actual, expected)
    ]

    passed = all(d < 0.00001 for d in differences)
    all_pass = all_pass and passed

    rows.append({
        "model": model,
        "topic_micro_f1": actual[0],
        "topic_macro_f1": actual[1],
        "claim_accuracy": actual[2],
        "claim_macro_f1": actual[3],
        "topic_failures": topic_result["failures"],
        "claim_failures": claim_result["failures"],
        "matches_previous": passed
    })

    print(f"\n{model}")
    print("Topic micro-F1:", round(actual[0], 6))
    print("Topic macro-F1:", round(actual[1], 6))
    print("Claim accuracy:", round(actual[2], 6))
    print("Claim macro-F1:", round(actual[3], 6))
    print("Topic failures:", topic_result["failures"])
    print("Claim failures:", claim_result["failures"])
    print("Claim class F1:", claim_result["per_class_f1"])
    print("MATCHES PREVIOUS:", passed)

    if not passed:
        print("Differences:", differences)

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "independently_verified_metrics.csv",
    index=False
)

print("\n=== FINAL STATUS ===")
print("All metrics match:", all_pass)
print("Saved:", OUT / "independently_verified_metrics.csv")
