"""Reproduce preliminary lexical trends from the Phase 1 cleaned ACL corpus.

Usage:
    python scripts/phase2_lexical.py --corpus results/cleaned_corpus.csv --out results/phase2
Dependencies: pandas, matplotlib

Cue definitions are explicit approximations; inspect them before interpreting
comparisons with previously reported percentages.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# A paper counts once per cue if its abstract contains any listed expression.
# Word boundaries avoid matching substrings (e.g. "may" in "maybe").
CUES = {
    'large language model / LLM': r'\b(?:large[\s-]+language[\s-]+models?|LLMs?)\b',
    'BERT': r'\bBERT\b',
    'machine translation / NMT': r'\b(?:neural[\s-]+machine[\s-]+translation|machine[\s-]+translation|NMT)\b',
    'significantly': r'\bsignificantly\b',
    'state-of-the-art / SOTA': r'\b(?:state[\s-]+of[\s-]+the[\s-]+art|SOTA)\b',
    'outperform': r'\boutperform(?:s|ed|ing)?\b',
    'may / might': r'\b(?:may|might)\b',
    'human-level / human parity': r'\b(?:human[\s-]+level|human[\s-]+parity)\b',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/phase2'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.corpus, usecols=['year', 'abstract'], keep_default_na=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df['year'].between(2016, 2025)].copy()
    df['year'] = df['year'].astype(int)
    if df.empty:
        raise SystemExit('No papers from 2016–2025 in input file.')

    years = list(range(2016, 2026))
    denominators = df.groupby('year').size().reindex(years, fill_value=0)
    counts = pd.DataFrame(index=years)
    percentages = pd.DataFrame(index=years)
    for name, pattern in CUES.items():
        hits = df['abstract'].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)
        yearly_hits = hits.groupby(df['year']).sum().reindex(years, fill_value=0)
        counts[name] = yearly_hits.astype(int)
        percentages[name] = (100 * yearly_hits / denominators.replace(0, float('nan'))).round(3)

    yearly = pd.concat([denominators.rename('abstracts'), counts.add_suffix(' [count]'), percentages.add_suffix(' [%]')], axis=1)
    yearly.index.name = 'year'
    yearly.to_csv(args.out / 'yearly_cue_frequencies.csv')

    selected = percentages.loc[[2016, 2019, 2022, 2025]].T
    selected.index.name = 'cue'
    selected.to_csv(args.out / 'table5_reproduced.csv', float_format='%.3f')

    fig, ax = plt.subplots(figsize=(11, 6))
    for name in CUES:
        ax.plot(years, percentages[name], marker='o', linewidth=1.6, markersize=3, label=name)
    for year, label in [(2018, 'BERT'), (2020, 'GPT-3'), (2022, 'ChatGPT')]:
        ax.axvline(year, linestyle=':', color='gray', alpha=0.6)
        ax.text(year + 0.06, ax.get_ylim()[1] * 0.98, label, rotation=90, va='top', fontsize=8)
    ax.set(xlabel='Publication year', ylabel='Abstracts containing cue (%)', title='Lexical cue frequencies in ACL Anthology abstracts, 2016–2025')
    ax.set_xticks(years)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out / 'figure2_reproduced.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    (args.out / 'cue_definitions.json').write_text(json.dumps(CUES, indent=2), encoding='utf-8')
    print('Papers analyzed (2016–2025):', len(df))
    print('\nReproduced Table 5 (percent):\n', selected.round(1).to_string())
    print('\nFiles saved in:', args.out.resolve())


if __name__ == '__main__':
    main()
