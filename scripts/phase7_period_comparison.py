
from pathlib import Path
import math
import pandas as pd
from scipy.stats import norm

ROOT = Path("results/phase7")
ROOT.mkdir(parents=True, exist_ok=True)

topics = pd.read_csv(ROOT / "topic_keyword_trends.csv")
claims = pd.read_csv(ROOT / "claim_cue_trends.csv")

PERIODS = {
    "2016-2022": (2016, 2022),
    "2023-2025": (2023, 2025),
}

def compare_periods(data, successes, label):
    records = []

    for period, (start, end) in PERIODS.items():
        part = data[data["year"].between(start, end)]

        n = int(part["papers"].sum())
        x = int(part[successes].sum())

        records.append({
            "indicator": label,
            "period": period,
            "papers": n,
            "matched": x,
            "prevalence_pct": 100 * x / n,
        })

    a, b = records
    p1 = a["matched"] / a["papers"]
    p2 = b["matched"] / b["papers"]

    difference = p2 - p1

    se = math.sqrt(
        p1 * (1 - p1) / a["papers"]
        + p2 * (1 - p2) / b["papers"]
    )

    pooled = (
        (a["matched"] + b["matched"])
        / (a["papers"] + b["papers"])
    )

    pooled_se = math.sqrt(
        pooled * (1 - pooled)
        * (1 / a["papers"] + 1 / b["papers"])
    )

    z = difference / pooled_se if pooled_se else 0
    p_value = 2 * norm.sf(abs(z)) if pooled_se else 1

    return {
        "indicator": label,
        "pre_2016_2022_pct": a["prevalence_pct"],
        "post_2023_2025_pct": b["prevalence_pct"],
        "difference_pp": 100 * difference,
        "ci_low_pp": 100 * (difference - 1.96 * se),
        "ci_high_pp": 100 * (difference + 1.96 * se),
        "p_value": p_value,
        "pre_n": a["papers"],
        "post_n": b["papers"],
        "pre_matches": a["matched"],
        "post_matches": b["matched"],
    }

results = []

for topic, group in topics.groupby("topic"):
    results.append(
        compare_periods(group, "matched_papers", topic)
    )

results.append(
    compare_periods(
        claims, "hedged_papers", "Hedging cues"
    )
)

results.append(
    compare_periods(
        claims, "strong_papers", "Strong-claim cues"
    )
)

output = pd.DataFrame(results)
output = output.sort_values(
    "difference_pp", ascending=False
)

output.to_csv(
    ROOT / "period_comparison.csv",
    index=False
)

print("\n=== PHASE 7.3 RESULTS ===")
print(
    output[
        [
            "indicator",
            "pre_2016_2022_pct",
            "post_2023_2025_pct",
            "difference_pp",
            "p_value"
        ]
    ].round(4).to_string(index=False)
)

print("\nSaved:", ROOT / "period_comparison.csv")
print(
    "\nNote: p-values are exploratory, unadjusted "
    "for multiple comparisons, and assume "
    "independent papers."
)
