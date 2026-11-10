
from pathlib import Path
import hashlib
import json
import pandas as pd

base = Path("results/phase6/independent_annotation")

files = {
    "papers": base / "papers_for_review_annotated.csv",
    "sentences": base / "sentences_for_review_annotated.csv"
}

papers = pd.read_csv(files["papers"], keep_default_na=False)
sentences = pd.read_csv(files["sentences"], keep_default_na=False)

assert len(papers) == 64, "Expected 64 papers"
assert len(sentences) == 128, "Expected 128 sentences"

assert not papers.paper_id.duplicated().any()
assert not sentences.sentence_id.duplicated().any()
assert set(sentences.paper_id) == set(papers.paper_id)
assert (sentences.groupby("paper_id").size() == 2).all()

topics = {
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
}

claims = {"Hedged", "Neutral", "Strong"}

errors = []

for _, row in papers.iterrows():
    labels = [
        x.strip()
        for x in str(row["topic_labels"]).split(";")
        if x.strip()
    ]

    if not labels:
        errors.append(f"{row.paper_id}: missing topic label")
    elif len(labels) != len(set(labels)):
        errors.append(f"{row.paper_id}: duplicate topic label")
    elif not set(labels).issubset(topics):
        errors.append(f"{row.paper_id}: unknown topic label")
    elif "Other" in labels and len(labels) > 1:
        errors.append(f"{row.paper_id}: Other mixed with topics")

for _, row in sentences.iterrows():
    if str(row["claim_label"]).strip() not in claims:
        errors.append(f"{row.sentence_id}: invalid claim label")

if errors:
    print("\n".join(errors))
    raise ValueError("Annotation validation failed")

manifest = {
    "papers": len(papers),
    "sentences": len(sentences),
    "claim_distribution": sentences.claim_label.value_counts().to_dict(),
    "files": {}
}

for name, path in files.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["files"][name] = {
        "path": str(path),
        "sha256": digest
    }

(base / "annotation_manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print("VALIDATION PASSED")
print(json.dumps(manifest, indent=2))
