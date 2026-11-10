import argparse
import hashlib
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

TOPICS = ["Machine translation", "Summarization", "Question answering", "Reasoning", "Dialogue systems", "Information extraction", "Sentiment and opinion", "Syntax and parsing", "Language modeling", "Multilingual / low-resource", "Fairness, bias and ethics", "Evaluation and benchmarks", "Multimodal NLP", "Speech and spoken language", "Other"]

def sentences(text):
    # spaCy sentence segmentation as specified in the midterm; no model download required.
    import spacy
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    return [s.text.strip() for s in nlp(str(text)).sents if len(s.text.strip()) >= 15]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',required=True)
    p.add_argument('--out',default='results/phase3')
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(args.corpus).fillna('')
    df['year']=pd.to_numeric(df['year'],errors='coerce')
    df=df[df.year.between(2016,2025)].copy()
    df['paper_id']=df.apply(lambda r: hashlib.sha256(f"{int(r.year)}|{str(r.title).strip().lower()}|{str(r.abstract).strip().lower()}".encode('utf-8')).hexdigest()[:16],axis=1)
    df=df.sort_values(['year','paper_id']).drop_duplicates('paper_id')
    # Deterministic sampling of papers eligible for two sentences.
    import spacy
    nlp=spacy.blank('en');nlp.add_pipe('sentencizer')
    def get_sents(t):return [s.text.strip() for s in nlp(str(t)).sents if len(s.text.strip())>=15]
    df['sentences']=df.abstract.map(get_sents)
    eligible=df[df.sentences.map(len)>=2]
    rng=np.random.default_rng(42)
    chosen=[]
    for year in range(2016,2026):
        group=eligible[eligible.year==year]
        if len(group)<5:raise ValueError(f'Only {len(group)} eligible papers in {year}')
        indices=rng.choice(len(group),size=5,replace=False)
        chosen.append(group.iloc[indices])
    papers=pd.concat(chosen).sort_values(['year','paper_id']).reset_index(drop=True)
    topic_rows=[]; claim_rows=[]
    for _,r in papers.iterrows():
        topic_rows.append({'paper_id':r.paper_id,'year':int(r.year),'title':r.title,'abstract':r.abstract,'topic_labels':'','annotator_notes':''})
        ids=rng.choice(len(r.sentences),size=2,replace=False)
        for j in sorted(ids):
            claim_rows.append({'sentence_id':f'{r.paper_id}-s{j+1}','paper_id':r.paper_id,'year':int(r.year),'title':r.title,'sentence':r.sentences[j],'claim_label':'','annotator_notes':''})
    topics=pd.DataFrame(topic_rows);claims=pd.DataFrame(claim_rows)
    for who in ('A','B'):
        topics.to_csv(out/f'topics_annotator_{who}.csv',index=False,encoding='utf-8-sig')
        claims.to_csv(out/f'claims_annotator_{who}.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'topic_label':TOPICS}).to_csv(out/'topic_taxonomy.csv',index=False)
    (out/'annotation_guidelines.txt').write_text('TOPICS: Assign all applicable labels, separated by semicolons, using exact names in topic_taxonomy.csv. Use Other only when none of the 14 named categories fits.\nCLAIM STRENGTH: Hedged = cautious or uncertain language; Neutral = method or finding without an explicit certainty signal; Strong = asserts performance, capability or impact, including superiority over a baseline. A score alone is Neutral. Consider sentence and its paper abstract.\nTwo annotators should label independently; do not share completed sheets before agreement calculation.\n',encoding='utf-8')
    (out/'sampling_metadata.json').write_text(json.dumps({'seed':42,'papers':len(topics),'sentences':len(claims),'years':list(range(2016,2026)),'eligibility':'At least two spaCy-sentencizer sentences of 15 or more characters','selection':'5 papers per year, 2 sentences per paper; deterministic RNG'},indent=2))
    print(f'Sampled {len(topics)} papers and {len(claims)} sentences.')
    print('Papers per year:',topics.groupby('year').size().to_dict())
    print('Outputs:',out.resolve())

if __name__=='__main__':main()
