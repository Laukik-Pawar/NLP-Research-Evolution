
from pathlib import Path
import pandas as pd

INPUT = Path("results/cleaned_corpus.csv")
OUT = Path("results/phase7")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT, keep_default_na=False, low_memory=False)
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(2010, 2025)].copy()
df["year"] = df["year"].astype(int)

print("\n=== ANNUAL COVERAGE ===")
print(df.groupby("year").size().to_string())

print("\n=== EARLY YEAR VENUE COVERAGE ===")
for year in range(2010, 2016):
    part = df[df["year"] == year]
    print(f"\nYEAR {year}: {len(part)} papers")
    print("Unique venues:", part["venue"].nunique())
    print("Top 5 venues:")
    print(part["venue"].value_counts().head(5).to_string())

print("\n=== SOURCE FIELD DIAGNOSTICS ===")
print("Unique source values:", df["source"].nunique())
print("Sample source values:")
for value in df["source"].value_counts().head(10).index:
    print(repr(str(value)[:100]))

print("\n=== EARLY YEAR SOURCE COVERAGE ===")
for year in range(2010, 2016):
    part = df[df["year"] == year]
    print(
        f"{year}: {len(part)} papers, "
        f"{part['source'].nunique()} unique source values"
    )

print("\n=== ABSTRACT LENGTH BY YEAR ===")
df["abstract_words"] = df["abstract"].astype(str).str.split().str.len()
print(
    df.groupby("year")["abstract_words"]
    .agg(["count", "median", "mean"])
    .round(1)
    .to_string()
)

print("\nAudit completed.")
