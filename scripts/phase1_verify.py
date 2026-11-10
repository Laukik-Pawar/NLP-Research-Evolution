"""Verify ACL Anthology XML corpus counts; no hard-coded results.
Usage: python phase1_verify.py --xml-dir path/to/anthology/data/xml --out results
Requires: pip install langdetect
"""
import argparse, csv, json, re, unicodedata
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


def plain(elem):
    return ' '.join(''.join(elem.itertext()).split()) if elem is not None else ''


def normalized_title(s):
    s=unicodedata.normalize('NFKC',s).casefold()
    return ''.join(ch for ch in s if ch.isalnum())


def run(root,out):
    try:
        from langdetect import detect, DetectorFactory, LangDetectException
    except ImportError:
        raise SystemExit('Install dependency first: pip install langdetect')
    DetectorFactory.seed=42
    out.mkdir(parents=True,exist_ok=True)
    raw=[]
    for path in sorted(root.rglob('*.xml')):
        try: tree=ET.parse(path)
        except ET.ParseError as e:
            print(f'SKIP invalid XML: {path}: {e}');continue
        for volume in tree.getroot().iter('volume'):
            year_text=plain(volume.find('meta/year')) or plain(volume.find('year'))
            try: year=int(year_text[:4])
            except (ValueError,TypeError):continue
            if not 2010<=year<=2025:continue
            venue=plain(volume.find('meta/booktitle'))
            for paper in volume.findall('paper'):
                if paper.attrib.get('id') == '0':continue # volume front matter
                title=plain(paper.find('title'))
                abstract=plain(paper.find('abstract'))
                lang=plain(paper.find('language')).lower()
                raw.append(dict(year=year,title=title,abstract=abstract,language=lang,venue=venue,source=str(path)))
    raw_by_year=Counter(x['year'] for x in raw)
    # The steps are sequential: a removed record cannot appear in later categories.
    seen=set(); unique=[];duplicates=0
    for row in sorted(raw,key=lambda r:(r['year'],r['source'],r['title'])):
        key=normalized_title(row['title'])
        if key in seen:duplicates+=1;continue
        seen.add(key);unique.append(row)
    complete=[];missing=0
    for row in unique:
        if not row['title'] or len(row['abstract'])<20:
            missing+=1;continue
        complete.append(row)
    cleaned=[];nonenglish=0;unknown=0
    for row in complete:
        lang=row['language']
        if lang and lang not in ('en','eng','english'):
            nonenglish+=1;continue
        if not lang:
            try: lang=detect(row['abstract'])
            except LangDetectException: unknown+=1;continue
            if lang!='en':nonenglish+=1;continue
        cleaned.append(row)
    clean_by_year=Counter(x['year'] for x in cleaned)
    with (out/'yearly_coverage.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['year','raw_papers','cleaned_papers','retention_pct'])
        for y in range(2010,2026):
            w.writerow([y,raw_by_year[y],clean_by_year[y],round(100*clean_by_year[y]/raw_by_year[y],2) if raw_by_year[y] else ''])
    with (out/'cleaned_corpus.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['year','title','abstract','language','venue','source']);w.writeheader();w.writerows(cleaned)
    counts={'raw':len(raw),'duplicates_removed':duplicates,'missing_or_short_abstract_removed':missing,'non_english_removed':nonenglish,'language_detection_failed_removed':unknown,'final':len(cleaned)}
    counts['arithmetic_check']=counts['raw']-sum(counts[k] for k in ('duplicates_removed','missing_or_short_abstract_removed','non_english_removed','language_detection_failed_removed'))==counts['final']
    (out/'counts.json').write_text(json.dumps(counts,indent=2),encoding='utf8')
    print(json.dumps(counts,indent=2))
    print('Outputs:',out.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--xml-dir',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results'))
    a=p.parse_args();run(a.xml_dir,a.out)
