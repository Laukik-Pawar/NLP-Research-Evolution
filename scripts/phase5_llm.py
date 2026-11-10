"""Phase 5: Open-weight LLM pilot classification via local Ollama.

Default: zero-shot evaluation on reviewed pilot data.
Few-shot: supply --dev-topics and --dev-claims containing independent,
non-overlapping labeled development examples (never evaluation examples).

pip install ollama pandas scikit-learn tqdm
python scripts/phase5_llm.py
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
from ollama import Client

TOPICS = [
    'Machine translation', 'Summarization', 'Question answering', 'Reasoning',
    'Dialogue systems', 'Information extraction', 'Sentiment and opinion',
    'Syntax and parsing', 'Language modeling', 'Multilingual / low-resource',
    'Fairness, bias and ethics', 'Evaluation and benchmarks', 'Multimodal NLP',
    'Speech and spoken language', 'Other'
]
CLAIMS = ['Hedged', 'Neutral', 'Strong']

TOPIC_SYSTEM = ("You are a careful research-paper topic annotator. Assign ALL applicable "
                "topics from the fixed taxonomy. Use the paper title and abstract. "
                "Do not invent categories. Use Other only if none of the 14 substantive "
                "categories applies. Respond with JSON only: {\"topics\":[\"exact label\"]}.\n"
                "Allowed labels: " + '; '.join(TOPICS))
CLAIM_SYSTEM = ("You classify the rhetorical strength of a research claim, NOT whether "
                "the claim is true. Hedged: uncertainty or cautious possibility. "
                "Neutral: descriptive, factual, or no explicit confidence/superiority "
                "signal. Strong: explicit performance, superiority, or capability "
                "assertion. Consider context: a numerical score alone is not necessarily "
                "Strong; words such as significantly need contextual interpretation. "
                "Respond with JSON only: {\"label\":\"Hedged|Neutral|Strong\"}.")


def load_csv(path, columns, key):
    df = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    if df[key].astype(str).duplicated().any():
        raise ValueError(f'{path}: duplicate {key}')
    return df


def labels(value):
    vals = [v.strip() for v in str(value).split(';') if v.strip()]
    bad = set(vals) - set(TOPICS)
    if bad or not vals:
        raise ValueError(f'Invalid topic labels {bad or vals}')
    return vals


def parse_json(raw):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, flags=re.S)
        if match:
            return json.loads(match.group())
        raise


def predict(client, model, system, prompt, kind, retries=3):
    error = ''
    for attempt in range(retries):
        try:
            response = client.chat(
                model=model,
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0, 'num_ctx': 4096, 'num_predict': 220},
            )
            raw = response.message.content
            obj = parse_json(raw)
            if not isinstance(obj, dict):
                raise ValueError('JSON response is not an object')
            if kind == 'topic':
                pred = obj.get('topics')
                if not isinstance(pred, list) or not pred:
                    raise ValueError('Missing/non-list/empty topics')
                if any(not isinstance(x, str) or x not in TOPICS for x in pred):
                    raise ValueError(f'Unknown topics: {pred}')
                pred = list(dict.fromkeys(pred))
                if 'Other' in pred and len(pred) > 1:
                    raise ValueError('Other must not coexist with substantive topics')
            else:
                pred = obj.get('label')
                if pred not in CLAIMS:
                    raise ValueError(f'Invalid claim label: {pred}')
            return pred, raw, ''
        except Exception as exc:
            error = str(exc)
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    return None, '', error


def topic_prompt(row, examples):
    parts = []
    if examples:
        parts.append('Labeled development examples (follow the labeling conventions):')
        for e in examples:
            parts.append('Title: ' + str(e['title']) + '\nAbstract: ' + str(e['abstract']) +
                         '\nAnswer: ' + json.dumps({'topics': labels(e['topic_labels'])}))
    parts.append('Classify this NEW paper:\nTitle: ' + str(row['title']) +
                 '\nAbstract: ' + str(row['abstract']) + '\nJSON answer:')
    return '\n\n'.join(parts)


def claim_prompt(row, examples):
    parts = []
    if examples:
        parts.append('Labeled development examples:')
        for e in examples:
            parts.append('Sentence: ' + str(e['sentence']) + '\nAnswer: ' +
                         json.dumps({'label': str(e['claim_label']).strip().title()}))
    parts.append('Classify this NEW sentence:\nSentence: ' + str(row['sentence']) +
                 '\nJSON answer:')
    return '\n\n'.join(parts)


def read_cache(path):
    if not path.exists():
        return {}
    return {str(x['id']): x for x in (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip())}


def run_items(client, model, df, kind, examples, cache_path):
    cache = read_cache(cache_path)
    idcol = 'paper_id' if kind == 'topic' else 'sentence_id'
    system = TOPIC_SYSTEM if kind == 'topic' else CLAIM_SYSTEM
    for _, row in tqdm(df.iterrows(), total=len(df), desc=kind):
        key = str(row[idcol])
        if key in cache and not cache[key].get('error'):
            continue
        prompt = topic_prompt(row, examples) if kind == 'topic' else claim_prompt(row, examples)
        result, raw, error = predict(client, model, system, prompt, kind)
        item = {'id': key, 'prediction': result, 'raw': raw, 'error': error}
        with cache_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
        cache[key] = item
    rows = []
    for _, row in df.iterrows():
        key = str(row[idcol])
        item = cache[key]
        rows.append({**row.to_dict(), 'predicted_labels':
                     '; '.join(item['prediction']) if kind == 'topic' and item['prediction'] else '',
                     'predicted_label': item['prediction'] if kind == 'claim' and item['prediction'] else '',
                     'error': item['error'], 'raw_response': item['raw']})
    return pd.DataFrame(rows)


def evaluate_topics(df):
    valid = df[df.error.eq('')].copy()
    if valid.empty:
        return {'evaluated': 0, 'failed': len(df)}
    mlb = MultiLabelBinarizer(classes=TOPICS)
    y_true = mlb.fit_transform(valid.topic_labels.map(labels))
    y_pred = mlb.transform(valid.predicted_labels.map(labels))
    return {'evaluated': len(valid), 'failed': len(df) - len(valid),
            'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'per_topic': {topic: {'support': int(y_true[:, i].sum()),
                                  'f1': float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0))}
                          for i, topic in enumerate(TOPICS)}}


def evaluate_claims(df):
    valid = df[df.error.eq('')].copy()
    if valid.empty:
        return {'evaluated': 0, 'failed': len(df)}
    truth = valid.claim_label.astype(str).str.strip().str.title()
    pred = valid.predicted_label.astype(str)
    return {'evaluated': len(valid), 'failed': len(df) - len(valid),
            'accuracy': float(accuracy_score(truth, pred)),
            'macro_f1': float(f1_score(truth, pred, labels=CLAIMS, average='macro', zero_division=0)),
            'per_class': classification_report(truth, pred, labels=CLAIMS,
                                               output_dict=True, zero_division=0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--topics', default='results/phase3/topics_reviewed.csv')
    ap.add_argument('--claims', default='results/phase3/claims_reviewed.csv')
    ap.add_argument('--out', default='results/phase5')
    ap.add_argument('--model', default='qwen2.5:3b')
    ap.add_argument('--host', default='http://localhost:11434')
    ap.add_argument('--mode', choices=['zero-shot', 'few-shot'], default='zero-shot')
    ap.add_argument('--dev-topics', help='Independent development CSV, required for few-shot')
    ap.add_argument('--dev-claims', help='Independent development CSV, required for few-shot')
    ap.add_argument('--k', type=int, default=3)
    args = ap.parse_args()
    topics = load_csv(args.topics, ['paper_id', 'title', 'abstract', 'topic_labels'], 'paper_id')
    claims = load_csv(args.claims, ['sentence_id', 'paper_id', 'sentence', 'claim_label'], 'sentence_id')
    for val in topics.topic_labels:
        labels(val)
    if not set(claims.claim_label.astype(str).str.strip().str.title()).issubset(CLAIMS):
        raise ValueError('Unknown claim-strength labels')
    if args.k < 1:
        raise ValueError('--k must be positive')
    t_examples, c_examples = [], []
    if args.mode == 'few-shot':
        if not args.dev_topics or not args.dev_claims:
            ap.error('Few-shot requires BOTH --dev-topics and --dev-claims from independent development data')
        dt = load_csv(args.dev_topics, ['paper_id', 'title', 'abstract', 'topic_labels'], 'paper_id')
        dc = load_csv(args.dev_claims, ['sentence_id', 'paper_id', 'sentence', 'claim_label'], 'sentence_id')
        if set(dt.paper_id.astype(str)) & set(topics.paper_id.astype(str)):
            raise ValueError('DATA LEAKAGE: development papers overlap evaluation papers')
        if set(dc.sentence_id.astype(str)) & set(claims.sentence_id.astype(str)):
            raise ValueError('DATA LEAKAGE: development sentences overlap evaluation sentences')
        if set(dc.paper_id.astype(str)) & set(topics.paper_id.astype(str)):
            raise ValueError('DATA LEAKAGE: development claim sentences from evaluation papers')
        if set(dt.paper_id.astype(str)) & set(claims.paper_id.astype(str)):
            raise ValueError('DATA LEAKAGE: development topic papers overlap evaluation claim papers')
        if len(dt) < args.k or len(dc) < args.k:
            raise ValueError('Not enough independent development examples')
        t_examples = dt.head(args.k).to_dict('records')
        c_examples = dc.head(args.k).to_dict('records')
    out = Path(args.out) / args.mode / args.model.replace(':', '_').replace('/', '_')
    out.mkdir(parents=True, exist_ok=True)
    client = Client(host=args.host, timeout=120)
    print('Model:', args.model, '| Mode:', args.mode)
    print('Topic papers:', len(topics), '| Claim sentences:', len(claims))
    print('Output:', out.resolve())
    topic_df = run_items(client, args.model, topics, 'topic', t_examples, out / 'topic_responses.jsonl')
    topic_df.to_csv(out / 'topic_predictions.csv', index=False, encoding='utf-8-sig')
    claim_df = run_items(client, args.model, claims, 'claim', c_examples, out / 'claim_responses.jsonl')
    claim_df.to_csv(out / 'claim_predictions.csv', index=False, encoding='utf-8-sig')
    metrics = {'model': args.model, 'mode': args.mode, 'temperature': 0,
               'topic': evaluate_topics(topic_df), 'claim': evaluate_claims(claim_df),
               'annotation_provenance': 'Human-reviewed AI-generated labels; not independent human ground truth.',
               'note': 'Failed JSON responses excluded from F1; failure counts reported. Predictions cached for restart.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== PHASE 5 RESULTS ===')
    print(json.dumps({'topic_micro_f1': metrics['topic'].get('micro_f1'),
                      'topic_macro_f1': metrics['topic'].get('macro_f1'),
                      'topic_failed': metrics['topic']['failed'],
                      'claim_macro_f1': metrics['claim'].get('macro_f1'),
                      'claim_accuracy': metrics['claim'].get('accuracy'),
                      'claim_failed': metrics['claim']['failed']}, indent=2))
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
