"""Phase 4: Reproducible pilot baseline evaluation.
Run: python scripts/phase4_baselines.py --topics results/phase3/topics_reviewed.csv --claims results/phase3/claims_reviewed.csv --out results/phase4
Requires: pandas numpy scikit-learn matplotlib
"""
import argparse, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TOPICS = ['Machine translation','Summarization','Question answering','Reasoning','Dialogue systems',
'Information extraction','Sentiment and opinion','Syntax and parsing','Language modeling',
'Multilingual / low-resource','Fairness, bias and ethics','Evaluation and benchmarks',
'Multimodal NLP','Speech and spoken language','Other']
CLASSES = ['Hedged','Neutral','Strong']
# Conservative, reproducible lexical rules, applied to whole sentences.
HEDGES = [r'\bmay\b',r'\bmight\b',r'\bcould\b',r'\bpossibly\b',r'\bperhaps\b',
          r'\bsuggest(?:s|ed|ing)?\b',r'\bappear(?:s|ed)?\s+to\b',r'\bpotentially\b',
          r'\bseem(?:s|ed)?\b',r'\blikely\b']
BOOSTERS = [r'\bstate[ -]of[ -]the[ -]art\b',r'\bsota\b',r'\boutperform(?:s|ed|ing)?\b',
            r'\bsignificantly\b',r'\bhuman[ -]level\b',r'\bsubstantially\b',
            r'\bsurpass(?:es|ed|ing)?\b',r'\bsuperior\b']

def read_csv(path, needed, idcol):
    df=pd.read_csv(path,keep_default_na=False,encoding='utf-8-sig')
    missing=set(needed)-set(df.columns)
    if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if df[idcol].duplicated().any(): raise ValueError(f'{path}: duplicate {idcol}')
    return df

def topic_run(df,out):
    if len(df)!=50: warnings.warn(f'Expected 50 papers; got {len(df)}')
    label_sets=[]
    for _,row in df.iterrows():
        labs=[v.strip() for v in str(row.topic_labels).split(';') if v.strip()]
        unknown=set(labs)-set(TOPICS)
        if unknown: raise ValueError(f'Unknown topic labels for {row.paper_id}: {unknown}')
        if not labs: raise ValueError(f'Empty topic labels for {row.paper_id}')
        label_sets.append(labs)
    mlb=MultiLabelBinarizer(classes=TOPICS)
    Y=mlb.fit_transform(label_sets)
    X=(df.title.astype(str)+' '+df.abstract.astype(str)).tolist()
    predictions=np.zeros_like(Y)
    fold_ids=np.zeros(len(df),dtype=int)
    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    for fold,(train,test) in enumerate(kf.split(X),start=1):
        fold_ids[test]=fold
        vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=2)
        X_train=vectorizer.fit_transform([X[i] for i in train])
        X_test=vectorizer.transform([X[i] for i in test])
        for k in range(len(TOPICS)):
            ytrain=Y[train,k]
            if len(np.unique(ytrain))<2:
                predictions[test,k]=int(ytrain[0])
            else:
                clf=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)
                clf.fit(X_train,ytrain)
                predictions[test,k]=(clf.predict_proba(X_test)[:,1]>=0.5).astype(int)
    micro=f1_score(Y,predictions,average='micro',zero_division=0)
    macro=f1_score(Y,predictions,average='macro',zero_division=0)
    per=[]
    for k,label in enumerate(TOPICS):
        p,r,f,_=precision_recall_fscore_support(Y[:,k],predictions[:,k],average='binary',zero_division=0)
        per.append({'topic':label,'support':int(Y[:,k].sum()),'predicted_positive':int(predictions[:,k].sum()),'precision':float(p),'recall':float(r),'f1':float(f)})
    pd.DataFrame(per).to_csv(out/'topic_per_label_metrics.csv',index=False)
    pred=df[['paper_id','year','title']].copy()
    pred['fold']=fold_ids
    pred['reference_labels']=['; '.join(x) for x in label_sets]
    pred['predicted_labels']=['; '.join(np.array(TOPICS)[p.astype(bool)]) for p in predictions]
    pred.to_csv(out/'topic_predictions.csv',index=False)
    return {'papers':len(df),'folds':5,'micro_f1':float(micro),'macro_f1':float(macro),
            'rare_labels_support_below_5':[x['topic'] for x in per if x['support']<5],
            'zero_support_labels':[x['topic'] for x in per if x['support']==0],
            'note':'TF-IDF fitted inside each training fold. Constant predictions used for labels absent from a training fold. Macro-F1 includes all 15 prespecified labels.'}

def classify(sentence):
    s=str(sentence).lower()
    h=sum(bool(re.search(p,s)) for p in HEDGES)
    b=sum(bool(re.search(p,s)) for p in BOOSTERS)
    return ('Strong' if b>h else 'Hedged' if h>b else 'Neutral'),h,b

def claims_run(df,out):
    if len(df)!=100: warnings.warn(f'Expected 100 sentences; got {len(df)}')
    truth=df.claim_label.astype(str).str.strip().str.title()
    unknown=set(truth)-set(CLASSES)
    if unknown: raise ValueError(f'Unknown claim labels: {unknown}')
    scored=[classify(x) for x in df.sentence]
    pred=[x[0] for x in scored]
    df=df[['sentence_id','paper_id','sentence','claim_label']].copy()
    df['predicted_label']=pred
    df['hedge_matches']=[x[1] for x in scored]
    df['booster_matches']=[x[2] for x in scored]
    df['correct']=(truth.to_numpy()==np.array(pred))
    df.to_csv(out/'claim_predictions.csv',index=False)
    report=classification_report(truth,pred,labels=CLASSES,output_dict=True,zero_division=0)
    pd.DataFrame([{'label':c,**{k:report[c][k] for k in ('precision','recall','f1-score','support')}} for c in CLASSES]).to_csv(out/'claim_per_class_metrics.csv',index=False)
    cm=confusion_matrix(truth,pred,labels=CLASSES)
    fig,ax=plt.subplots(figsize=(6,5))
    ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(3),CLASSES);ax.set_yticks(range(3),CLASSES)
    ax.set_xlabel('Predicted');ax.set_ylabel('Reviewed reference');ax.set_title('Keyword baseline — claim strength')
    for i in range(3):
        for j in range(3): ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(out/'claim_confusion_matrix.png',dpi=180);plt.close(fig)
    return {'sentences':len(df),'accuracy':float(report['accuracy']),
            'macro_f1':float(report['macro avg']['f1-score']),
            'weighted_f1':float(report['weighted avg']['f1-score']),
            'reference_distribution':{c:int((truth==c).sum()) for c in CLASSES},
            'predicted_distribution':{c:int(sum(x==c for x in pred)) for c in CLASSES},
            'lexicon':{'hedges':HEDGES,'boosters':BOOSTERS},
            'note':'Counts distinct regex patterns per sentence, not word occurrences. Ties map to Neutral.'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--topics',required=True);ap.add_argument('--claims',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    t=read_csv(args.topics,['paper_id','title','abstract','topic_labels'],'paper_id')
    c=read_csv(args.claims,['sentence_id','paper_id','sentence','claim_label'],'sentence_id')
    if not set(c.paper_id).issubset(set(t.paper_id)): raise ValueError('Claims refer to papers absent from topic file')
    results={'topic_baseline':topic_run(t,out),'claim_baseline':claims_run(c,out),
             'annotation_provenance':'User-reviewed AI-generated labels; not independently human-annotated ground truth.'}
    (out/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'topic_micro_f1':results['topic_baseline']['micro_f1'],
                      'topic_macro_f1':results['topic_baseline']['macro_f1'],
                      'claim_macro_f1':results['claim_baseline']['macro_f1'],
                      'claim_accuracy':results['claim_baseline']['accuracy'],
                      'rare_topics':results['topic_baseline']['rare_labels_support_below_5']},indent=2))
    print('Outputs:',out.resolve())
if __name__=='__main__':main()
