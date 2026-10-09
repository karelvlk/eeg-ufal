"""Audit the full pinned index/cache, optionally load representative public files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re

import pandas as pd
from src.dataset_cache import blob_sha, index, resolve, safe_path

PHASES = ['Read', 'Translate', 'See', 'Update']

def check_metadata(manifest, root):
    commit, files = index(str(root))
    frame = pd.read_csv(manifest, keep_default_na=False)
    errors = []
    if not {'participant','sentence','condition','order','dataset_commit'} <= set(frame):
        raise ValueError('Missing trial identity or pinned-commit fields')
    if frame.duplicated(['participant','sentence','condition','order']).any():
        errors.append('Duplicate exact trial identity')
    if not frame.dataset_commit.eq(commit).all():
        errors.append('Manifest commit does not match dataset index')
    linked = 0
    for column in [name for name in frame if name.endswith('_path')]:
        for relative in frame[column]:
            if not relative:
                continue
            linked += 1
            try:
                safe_path(root, relative)
                if relative not in files:
                    errors.append(f'{column}: not in pinned tree: {relative}')
            except ValueError as exc:
                errors.append(f'{column}: {exc}')
    cached, empty, cached_bytes, corrupt = 0, 0, 0, []
    for relative, entry in files.items():
        path = safe_path(root, relative)
        if not path.is_file():
            continue
        content = path.read_bytes()
        cached += 1
        cached_bytes += len(content)
        empty += not content
        if len(content) != entry['size'] or blob_sha(content) != entry['sha']:
            corrupt.append(relative)
    eeg = [name for name in files if name.startswith('preprocessed-data/eeg/') and name.endswith('.csv')]
    gaze = [name for name in files if name.startswith('preprocessed-data/gaze/') and name.endswith('.csv')]
    eeg_people = sorted({re.search(r'P\d+',Path(name).name)[0] for name in eeg})
    report = {'dataset_commit':commit, 'manifest_sha256':hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              'trials':len(frame), 'participants':frame.participant.nunique(),
              'indexed_files':len(files), 'indexed_bytes':sum(x['size'] for x in files.values()),
              'phase_eeg_csv':len(eeg), 'phase_gaze_csv':len(gaze),
              'eeg_participants':eeg_people,
              'linked_paths_checked':linked, 'metadata_errors':errors,
              'cached_files_checked':cached, 'cached_bytes':cached_bytes,
              'published_empty_cached_files':empty, 'corrupt_cached_files':corrupt,
              'coverage_note':'Complete metadata/cache audit; sample loading is not verification of every remote file or scientific data quality.'}
    return frame, files, report

def choose_samples(frame, files):
    """One deterministic phase per participant; EEG when available, plus media."""
    selected = {}
    for number, (participant, group) in enumerate(frame.groupby('participant',sort=True)):
        phases = PHASES[number % 4:] + PHASES[:number % 4]
        for phase in phases:
            gaze_col, eeg_col = f'gaze_{phase}_path', f'eeg_{phase}_path'
            has_gaze = group[gaze_col].map(lambda value: bool(value) and files.get(value,{}).get('size',0)>0)
            candidates = group[has_gaze].copy()
            if candidates.empty:
                continue
            has_eeg = candidates[eeg_col].map(lambda value: bool(value) and files.get(value,{}).get('size',0)>0)
            if has_eeg.any():
                candidates = candidates[has_eeg]
            row = candidates.sort_values(['order','sentence']).iloc[0]
            for kind in ['gaze','eeg']:
                path = row[f'{kind}_{phase}_path']
                if path and files.get(path,{}).get('size',0)>0:
                    selected[path]={'kind':kind,'participant':participant,'phase':phase}
            break
    for condition in ['N','R','U']:
        group = frame[frame.image_condition.eq(condition) & frame.raw_events_path.ne('') & frame.image_path.ne('') & frame.raw_mp3_path.ne('')]
        if group.empty:
            continue
        row = group.sort_values(['participant','order']).iloc[0]
        for kind, field in [('image','image_path'),('events','raw_events_path'),('audio','raw_mp3_path')]:
            selected[row[field]]={'kind':kind,'participant':row.participant,'condition':condition}
    return selected

def load_sample(root, relative, info):
    result = {'path':relative, **info}
    try:
        path = resolve(root,relative)
        result['bytes'] = path.stat().st_size
        if info['kind'] in ['eeg','gaze']:
            frame = pd.read_csv(path)
            required = {'TimeStamp','X','Y','Event'} if info['kind']=='gaze' else {'TimeStamp','RAW_TP9','RAW_AF7','RAW_AF8','RAW_TP10'}
            if not required <= set(frame):
                raise ValueError('Missing expected columns')
            result['rows'] = len(frame)
            result['columns'] = list(frame.columns)
            if frame.empty:
                result.update(status='published_no_samples', note='Verified published CSV has headers but no recorded samples; not zero activity.')
                return result
        elif info['kind']=='image':
            from PIL import Image
            with Image.open(path) as image:
                result['image_size'] = list(image.size)
                image.verify()
        elif info['kind']=='events':
            text = path.read_text()
            if 'present image' not in text or 'start response recording' not in text:
                raise ValueError('Missing expected trial event markers')
        elif info['kind']=='audio':
            from pydub import AudioSegment
            sound = AudioSegment.from_file(path,format='mp3')
            if len(sound)==0:
                raise ValueError('Empty decoded audio')
            result['duration_seconds'] = len(sound)/1000
        result['status']='loaded'
    except Exception as exc:
        result.update(status='failed',error=str(exc))
    return result

def main():
    project = Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=project/'.cache/full_dataset/data')
    parser.add_argument('--manifest',type=Path,default=project/'.cache/full_dataset/manifest.csv')
    parser.add_argument('--output',type=Path,default=project/'.cache/dataset_audit.json')
    parser.add_argument('--fetch-samples',action='store_true')
    args=parser.parse_args()
    os.environ['EMMT_LAZY_DOWNLOAD']='1'
    frame,files,report=check_metadata(args.manifest,args.root)
    report['sample_results']=[]
    if args.fetch_samples:
        samples=choose_samples(frame,files)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(load_sample,args.root,path,info) for path,info in samples.items()]
            for future in as_completed(futures):
                result=future.result()
                report['sample_results'].append(result)
                print(result['status'],result['path'],result.get('error',''),flush=True)
        report['sample_results'].sort(key=lambda r:r['path'])
        loaded=[r for r in report['sample_results'] if r['status']=='loaded']
        report['loaded_by_kind']=dict(Counter(r['kind'] for r in loaded))
        report['loaded_csv_participants']=sorted({r['participant'] for r in loaded if r['kind'] in ['eeg','gaze']})
        report['loaded_csv_phases']=sorted({r['phase'] for r in loaded if 'phase' in r})
    report['published_data_gaps']=[r['path'] for r in report['sample_results'] if r['status']=='published_no_samples']
    report['passed']=not(report['metadata_errors'] or report['corrupt_cached_files'] or any(r['status']=='failed' for r in report['sample_results']))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='sample_results'},indent=2),flush=True)
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':
    main()
