
from pathlib import Path
import pandas as pd

SEED = 123
PAPERS_PER_YEAR = 4

base = Path("results/phase6")
out = base / "independent_annotation"
out.mkdir(parents=True, exist_ok=True)

papers = pd.read_csv(
    base / "papers_to_annotate.csv",
    keep_default_na=False
)

sentences = pd.read_csv(
    base / "sentences_to_annotate.csv",
    keep_default_na=False
)

# Four papers from each of the 16 years.
subset = (
    papers.groupby("year", group_keys=False)
    .sample(n=PAPERS_PER_YEAR, random_state=SEED)
    .sort_values(["year", "paper_id"])
    .reset_index(drop=True)
)

selected_ids = set(subset["paper_id"])

sentence_subset = sentences[
    sentences["paper_id"].isin(selected_ids)
].copy()

sentence_subset = sentence_subset.sort_values(
    ["year", "sentence_id"]
).reset_index(drop=True)

# Empty annotation fields: do not generate reference labels.
subset["topic_labels"] = ""
subset["annotation_notes"] = ""
sentence_subset["claim_label"] = ""
sentence_subset["annotation_notes"] = ""

subset.to_csv(
    out / "papers_for_review.csv",
    index=False,
    encoding="utf-8-sig"
)

sentence_subset.to_csv(
    out / "sentences_for_review.csv",
    index=False,
    encoding="utf-8-sig"
)

assert len(subset) == 64
assert len(sentence_subset) == 128
assert sentence_subset["paper_id"].nunique() == 64

print("Papers selected:", len(subset))
print("Sentences selected:", len(sentence_subset))
print("Years:", subset["year"].nunique())
print("Papers per year:",
      subset.groupby("year").size().to_dict())
print("Output:", out.resolve())
