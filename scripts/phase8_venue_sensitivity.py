
from pathlib import Path
import re
import pandas as pd

ROOT = Path("results")
OUT = ROOT / "phase8"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ROOT / "cleaned_corpus.csv",
    keep_default_na=False,
    low_memory=False
)

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(2016, 2025)].copy()
df["year"] = df["year"].astype(int)

df["period"] = df["year"].apply(
    lambda y: "2016-2022" if y <= 2022 else "2023-2025"
)

# Normalize venue families using source XML filenames.
# This is more stable than complete proceedings titles.
def venue_family(source):
    filename = str(source).replace("\\", "/").split("/")[-1].lower()
    stem = re.sub(r"\.xml$", "", filename)

    # Standard conference XML: 2025.acl.xml, 2024.emnlp.xml, etc.
    match = re.match(r"^\d{4}\.([a-z0-9_-]+)$", stem)
    if match:
        return match.group(1)

    # Historical anthology XML: P19.xml, W19.xml, etc.
    match = re.match(r"^([a-z]+)\d{2}$", stem)
    if match:
        return match.group(1)

    return stem

df["venue_family"] = df["source"].apply(venue_family)

coverage = (
    df.groupby(["venue_family", "period"])
    .size()
    .unstack(fill_value=0)
)

shared = coverage[
    (coverage["2016-2022"] >= 30)
    & (coverage["2023-2025"] >= 30)
].index

subset = df[df["venue_family"].isin(shared)].copy()

print("\n=== SHARED VENUE FAMILIES ===")
print(coverage.loc[shared].to_string())

print("\nPapers in original analysis:", len(df))
print("Papers in shared-family subset:", len(subset))

patterns = {
    "Language modeling": r"\blanguage models?\b",
    "Reasoning": r"\breasoning\b|\bchain.of.thought\b",
    "Machine translation": r"\bmachine translation\b",
    "Multimodal NLP": (
        r"\bmultimodal\b|\bvision.language\b"
        r"|\bimage captioning\b"
    ),
    "Strong-claim cues": (
        r"\bsignificantly\b|\bsubstantially\b"
        r"|\bclearly\b|\bdefinitively\b"
        r"|\boutperform(?:s|ed)?\b"
        r"|\bstate.of.the.art\b|\bsuperior\b"
        r"|\bdemonstrate(?:s|d)?\b"
    )
}

rows = []

for name, pattern in patterns.items():
    for label, data in [
        ("Full corpus", df),
        ("Shared families", subset)
    ]:
        # Topic indicators use title + abstract.
        # Claim indicators use abstract only.
        if name == "Strong-claim cues":
            text = data["abstract"].astype(str)
        else:
            text = (
                data["title"].astype(str)
                + " "
                + data["abstract"].astype(str)
            )

        matched = text.str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )

        temp = data[["period"]].copy()
        temp["matched"] = matched.to_numpy()

        rates = temp.groupby("period")["matched"].mean() * 100

        pre = rates.get("2016-2022", float("nan"))
        post = rates.get("2023-2025", float("nan"))

        rows.append({
            "indicator": name,
            "sample": label,
            "pre_pct": pre,
            "post_pct": post,
            "difference_pp": post - pre
        })

results = pd.DataFrame(rows)

results.to_csv(
    OUT / "venue_sensitivity.csv",
    index=False
)

print("\n=== SENSITIVITY RESULTS ===")
print(results.round(2).to_string(index=False))

print("\nSaved:", OUT / "venue_sensitivity.csv")
