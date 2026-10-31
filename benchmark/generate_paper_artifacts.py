"""Regenerate the paper from saved observations, without running any benchmark.

From the repository root: python benchmark/generate_paper_artifacts.py
Requires Python, NumPy and Matplotlib. Outputs are repository-relative.
"""
from pathlib import Path
import argparse
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paper_artifacts_io import EvidenceReads, normalize_exports, write_json, utf8_sanity

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only',choices=['campaign','support','sensitivity'])
    args=parser.parse_args()
    utf8_sanity(ROOT)
    reads=EvidenceReads(ROOT).start()
    if args.only in (None,'campaign'):
        from paper_artifacts_campaign import generate_campaign
        with plt.rc_context(plt.rcParamsDefault):
            generate_campaign(ROOT)
    if args.only in (None,'support'):
        from paper_artifacts_support import generate_support
        with plt.rc_context(plt.rcParamsDefault):
            generate_support(ROOT)
    if args.only in (None,'sensitivity'):
        from paper_artifacts_sensitivity import generate
        generate(ROOT)
    if args.only is None:
        from paper_artifacts_charts import write_atlas
        write_atlas(ROOT)
        from paper_artifacts_reference_audit import generate as generate_reference_audit
        generate_reference_audit(ROOT)
        from paper_artifacts_claims import generate as generate_claims
        generate_claims(ROOT)
        reads.save()
    normalize_exports(ROOT)
    utf8_sanity(ROOT, outputs=True)
    files=[]
    for folder in ['latex/images/current','latex/generated','benchmark/paper-artifacts']:
        for path in sorted((ROOT/folder).glob('*')):
            if path.is_file() and path.name!='artifact_index.json':
                files.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),
                              'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    atlas=ROOT/'latex/supplementary_figures.tex'
    if atlas.is_file():
        files.append({'path':atlas.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(atlas.read_bytes()).hexdigest(),'bytes':atlas.stat().st_size})
    write_json(ROOT/'benchmark/paper-artifacts/artifact_index.json',{'generator':'benchmark/generate_paper_artifacts.py','artifacts':files})
    print('Generated and indexed',len(files),'artifacts.')

if __name__=='__main__':main()
