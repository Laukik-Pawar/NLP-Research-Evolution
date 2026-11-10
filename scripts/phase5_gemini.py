
"""Phase 5.3: Gemini API classification on the same Qwen test split.

Install:
    pip install -U google-genai pandas scikit-learn tqdm

Smoke test:
    python scripts/phase5_gemini.py --limit 1 --out results/phase5/gemini_smoke

Full run:
    python scripts/phase5_gemini.py

The script caches responses and evaluates invalid topic outputs as empty
predictions across all test papers.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Free-tier rate limiting
REQUEST_INTERVAL_SECONDS = 7.0
_last_request_time = 0.0


def wait_for_rate_limit():
    global _last_request_time

    elapsed = time.monotonic() - _last_request_time
    remaining = REQUEST_INTERVAL_SECONDS - elapsed

    if remaining > 0:
        time.sleep(remaining)

    _last_request_time = time.monotonic()

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

CLAIMS = ["Hedged", "Neutral", "Strong"]

TOPIC_INSTRUCTIONS = (
    "You are a careful research-paper topic annotator. "
    "Assign ALL applicable topics from the fixed taxonomy. "
    "Use the title and abstract. Do not invent categories. "
    "Use Other only when no substantive category applies. "
    'Return JSON: {"topics":["exact category name"]}. '
    "Allowed categories: " + "; ".join(TOPICS)
)

CLAIM_INSTRUCTIONS = (
    "Classify rhetorical claim strength, not factual truth. "
    "Hedged: uncertainty, caution, or possibility. "
    "Neutral: descriptive or factual wording without explicit "
    "confidence or superiority. "
    "Strong: explicit superiority, performance, or capability claim. "
    "Interpret words in context. "
    'Return JSON: {"label":"Hedged|Neutral|Strong"}.'
)


def parse_labels(value):
    return [
        x.strip()
        for x in str(value).split(";")
        if x.strip()
    ]


def read_data(path, required, key):
    df = pd.read_csv(
        path, dtype={key: str}, keep_default_na=False
    )
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    if df[key].duplicated().any():
        raise ValueError(f"{path}: duplicate {key}")
    return df


def read_cache(path):
    result = {}
    if path.exists():
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                item = json.loads(line)
                result[str(item["id"])] = item
    return result


def request_prediction(client, model, instructions, prompt, kind):
    for attempt in range(3):
        try:
            print(f"[Gemini] {kind} request, attempt {attempt + 1}/3",flush=True)
            wait_for_rate_limit()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    temperature=0,
                    response_mime_type="application/json",
                    max_output_tokens=256,
                ),
            )

            raw = response.text or ""
            obj = json.loads(raw)

            if not isinstance(obj, dict):
                raise ValueError("Expected JSON object")

            if kind == "topic":
                pred = obj.get("topics")
                if not isinstance(pred, list) or not pred:
                    raise ValueError("Invalid topic list")

                if any(
                    not isinstance(x, str) or x not in TOPICS
                    for x in pred
                ):
                    raise ValueError(f"Unknown topics: {pred}")

                pred = list(dict.fromkeys(pred))

                if "Other" in pred and len(pred) > 1:
                    raise ValueError(
                        "Other cannot coexist with other topics"
                    )
            else:
                pred = obj.get("label")
                if pred not in CLAIMS:
                    raise ValueError(
                        f"Invalid claim label: {pred}"
                    )

            return pred, raw, ""

        except Exception as exc:
            error = str(exc)
            print(f"[Gemini] Request failed: {error[:300]}",flush=True)
            # Stop on quota, authentication, or permission errors.
            # Retrying these usually wastes time.
            if any(
                code in error.lower()
                for code in [
                    "429",
                    "resource_exhausted",
                    "quota",
                    "401",
                    "403",
                    "permission_denied",
                    "api key not valid",
                ]
            ):
                raise RuntimeError(
                    "Gemini API quota/authentication problem: "
                    + error
                ) from exc

            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    return None, "", error


def run_predictions(client, model, df, kind, out):
    idcol = "paper_id" if kind == "topic" else "sentence_id"
    cache_path = out / f"{kind}_responses.jsonl"
    cache = read_cache(cache_path)

    rows = []

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=kind
    ):
        key = str(row[idcol])

        if key not in cache:
            if kind == "topic":
                prompt = (
                    "Classify this paper:\n"
                    f"Title: {row['title']}\n"
                    f"Abstract: {row['abstract']}\n"
                    "Return JSON only."
                )
                instructions = TOPIC_INSTRUCTIONS
            else:
                prompt = (
                    "Classify this sentence:\n"
                    f"Sentence: {row['sentence']}\n"
                    "Return JSON only."
                )
                instructions = CLAIM_INSTRUCTIONS

            prediction, raw, error = request_prediction(
                client, model, instructions, prompt, kind
            )

            item = {
                "id": key,
                "prediction": prediction,
                "raw": raw,
                "error": error,
            }

            with cache_path.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(item, ensure_ascii=False)
                    + "\n"
                )

            cache[key] = item

        item = cache[key]
        record = row.to_dict()

        if kind == "topic":
            pred = item["prediction"]
            record["predicted_labels"] = (
                "; ".join(pred) if pred else ""
            )
        else:
            record["predicted_label"] = (
                item["prediction"] or ""
            )

        record["error"] = item["error"]
        record["raw_response"] = item["raw"]
        rows.append(record)

    return pd.DataFrame(rows)


def evaluate_topics(df):
    mlb = MultiLabelBinarizer(classes=TOPICS)

    truth = df["topic_labels"].map(parse_labels).tolist()

    failed = df["error"].fillna("").astype(str).str.strip().ne("")

    predictions = [
        [] if is_failed else parse_labels(value)
        for is_failed, value in zip(
            failed, df["predicted_labels"]
        )
    ]

    y_true = mlb.fit_transform(truth)
    y_pred = mlb.transform(predictions)

    return {
        "evaluated": len(df),
        "failed": int(failed.sum()),
        "invalid_output_rate": float(failed.mean()),
        "micro_f1": float(
            f1_score(
                y_true, y_pred,
                average="micro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                average="macro", zero_division=0
            )
        ),
    }


def evaluate_claims(df):
    failed = df["error"].fillna("").astype(str).str.strip().ne("")
    valid = df.loc[~failed]

    if valid.empty:
        return {
            "evaluated": 0,
            "failed": len(df),
            "accuracy": None,
            "macro_f1": None,
        }

    truth = valid["claim_label"].astype(str).str.strip().str.title()
    pred = valid["predicted_label"].astype(str).str.strip()

    return {
        "evaluated": len(valid),
        "failed": int(failed.sum()),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(
            f1_score(
                truth, pred, labels=CLAIMS,
                average="macro", zero_division=0
            )
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--topics",
        default="results/phase5/split/test_topics.csv",
    )
    ap.add_argument(
        "--claims",
        default="results/phase5/split/test_claims.csv",
    )
    ap.add_argument(
        "--out",
        default="results/phase5/gemini",
    )
    ap.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Number of papers and sentences to test; 0 means all",
    )
    args = ap.parse_args()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError(
            "Set GEMINI_API_KEY in PowerShell first."
        )

    topics = read_data(
        args.topics,
        ["paper_id", "title", "abstract", "topic_labels"],
        "paper_id",
    )
    claims = read_data(
        args.claims,
        ["sentence_id", "paper_id", "sentence", "claim_label"],
        "sentence_id",
    )

    if args.limit < 0:
        raise ValueError("--limit cannot be negative")

    if args.limit:
        topics = topics.head(args.limit).copy()
        claims = claims.head(args.limit).copy()

    out = Path(args.out) / args.model.replace("/", "_")
    out.mkdir(parents=True, exist_ok=True)

    client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(
        timeout=30000,  # 30 seconds in milliseconds
        retry_options=types.HttpRetryOptions(
            attempts=1
        )
    )
)

    print("Model:", args.model)
    print("Topic papers:", len(topics))
    print("Claim sentences:", len(claims))
    print("Output:", out.resolve())

    topic_df = run_predictions(
        client, args.model, topics, "topic", out
    )
    topic_df.to_csv(
        out / "topic_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    claim_df = run_predictions(
        client, args.model, claims, "claim", out
    )
    claim_df.to_csv(
        out / "claim_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    metrics = {
        "model": args.model,
        "mode": "zero-shot",
        "temperature": 0,
        "topic": evaluate_topics(topic_df),
        "claim": evaluate_claims(claim_df),
        "annotation_provenance": (
            "Human-reviewed AI-generated labels; "
            "not independent human ground truth."
        ),
        "note": (
            "Topic failures count as empty predictions. "
            "Claim failures are excluded from claim F1 "
            "and separately reported."
        ),
    }

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("\n=== GEMINI RESULTS ===")
    print(json.dumps(metrics, indent=2))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
