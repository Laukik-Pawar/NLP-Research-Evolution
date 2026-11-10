"""Create reproducible paper-disjoint development/test CSVs for Phase 5.

Run from NLP_Project root:
python scripts/phase5_prepare_split.py

All claims belonging to a paper stay in the same split. The 10 development
papers are selected deterministically to include each claim class when possible.
This is an exploratory split of a pilot already examined; not an untouched test set.
"""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

CLASSES = ['Hedged', 'Neutral', 'Strong']

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topics', default='results/phase3/topics_reviewed.csv')
    p.add_argument('--claims', default='results/phase3/claims_reviewed.csv')
    p.add_argument('--out', default='results/phase5/split')
    p.add_argument('--dev-papers', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    topics = pd.read_csv(a.topics, encoding='utf-8-sig', keep_default_na=False, dtype={'paper_id':str})
    claims = pd.read_csv(a.claims, encoding='utf-8-sig', keep_default_na=False, dtype={'paper_id':str})
    for name, df, cols in [('topics', topics, ['paper_id','title','abstract','topic_labels']),
                           ('claims', claims, ['sentence_id','paper_id','sentence','claim_label'])]:
        missing = set(cols)-set(df.columns)
        if missing: raise ValueError(f'{name} missing {sorted(missing)}')
    if topics.paper_id.duplicated().any() or claims.sentence_id.duplicated().any():
        raise ValueError('Duplicate IDs in input')
    if not set(claims.paper_id).issubset(set(topics.paper_id)):
        raise ValueError('Some claim sentences have no matching topic paper')
    if not 3 <= a.dev_papers < len(topics):
        raise ValueError('Choose at least 3 development papers and leave test papers')
    rng = np.random.default_rng(a.seed)
    ids = list(rng.permutation(topics.paper_id.to_numpy()))
    # Greedy deterministic selection for class diversity; never examines model predictions.
    selected = []
    for label in CLASSES:
        candidates = [pid for pid in ids if pid not in selected and
                      ((claims.paper_id==pid) & (claims.claim_label.str.strip().str.title()==label)).any()]
        if candidates: selected.append(candidates[0])
    selected += [pid for pid in ids if pid not in selected][:a.dev_papers-len(selected)]
    dev_ids = set(selected)
    dt, et = topics[topics.paper_id.isin(dev_ids)].copy(), topics[~topics.paper_id.isin(dev_ids)].copy()
    dc, ec = claims[claims.paper_id.isin(dev_ids)].copy(), claims[~claims.paper_id.isin(dev_ids)].copy()
    if set(dt.paper_id) & set(et.paper_id) or set(dc.paper_id) & set(ec.paper_id):
        raise RuntimeError('Split overlap')
    # Put one of each claim class first, so the existing phase5_llm.py --k 3
    # uses a diverse few-shot demonstration set.
    first=[]
    for label in CLASSES:
        match=dc[dc.claim_label.str.strip().str.title().eq(label)]
        if not match.empty: first.append(match.index[0])
    dc=pd.concat([dc.loc[first], dc.drop(index=first)], ignore_index=True)
    out=Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fname, df in [('dev_topics.csv',dt),('test_topics.csv',et),
                      ('dev_claims.csv',dc),('test_claims.csv',ec)]:
        df.to_csv(out/fname,index=False,encoding='utf-8-sig')
    metadata={'seed':a.seed,'dev_papers':len(dt),'test_papers':len(et),
              'dev_sentences':len(dc),'test_sentences':len(ec),
              'dev_claim_distribution':dc.claim_label.value_counts().to_dict(),
              'test_claim_distribution':ec.claim_label.value_counts().to_dict(),
              'warning':'Exploratory pilot; earlier full-pilot test outcomes have already been examined.'}
    (out/'split_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))
    print('Saved:',out.resolve())

if __name__=='__main__': main()
