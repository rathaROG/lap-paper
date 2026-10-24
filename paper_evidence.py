"""Audit the revised paper against frozen evidence without executing solvers."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import re
import zipfile

PAPER = Path(__file__).resolve().parent
COLLECTION = PAPER / 'collections/python-3.11.17-20261024'
EXTENSION = PAPER / 'extensions/python-3.14.9-20261024'
COMPARISON = PAPER / 'comparisons/python-3.11.17-vs-3.14.9-20261024'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def preserved():
    manifest=read(PAPER/'publication/evidence_manifest.json')
    for name, expected in manifest.items():
        assert digest(PAPER/name)==expected, f'Frozen evidence changed: {name}'
    return len(manifest)

def verify():
    import check_artifact as helpers
    import narrative
    import update_manuscript_tables as tables
    count=preserved()
    output=PAPER/'output/arxiv'
    output.mkdir(parents=True,exist_ok=True)
    recomputed=output/'evidence-check'
    recomputed.mkdir(exist_ok=True)
    spec=importlib.util.spec_from_file_location('frozen_comparison',COMPARISON/'compare.py')
    analysis=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    # The old guard deliberately froze the previous manuscript too. Publication
    # now has its own manifest covering every measurement and comparison file.
    analysis.protect=preserved
    analysis.OUT=recomputed
    _,_,_,_,result=analysis.analyze()
    historical=read(COMPARISON/'comparison.json')
    for key in ('environments','suites','settings','cases','primary_comparisons','pairwise_comparisons',
                'pairwise_reversals','winner_changes','reversal_margin_sensitivity','correctness',
                'edge_checks_identical','input_hashes_identical','memory_conditions','max_absolute_memory_median_difference_mib'):
        assert result[key]==historical[key], key
    for name in ('timing_comparison.csv','primary_orderings.csv','pairwise_reversals.csv',
                 'case_winners.csv','memory_comparison.csv','key_metrics.csv'):
        assert rows(recomputed/name)==rows(COMPARISON/name), name

    manuscript=(PAPER/'manuscript.md').read_text(encoding='utf-8')
    assert not re.search(r'0\.11\.0|3\.11\.16|0\.11\.0rc1|archive/python',manuscript)
    assert not re.search(r'program\.,|1,271|23 Rust unit tests|20 for LAPX',manuscript)
    checked_tables=0
    for section,(expected,places) in tables.all_table_values().items():
        block=re.search(rf'### {re.escape(section)} .*?\n(.*?)(?=\n### |\n## |\Z)',manuscript,re.S).group(1)
        actual=[v for _,v in helpers.numeric_rows(block)]
        assert actual==[[float(f'{v:.{places}f}') for v in row] for row in expected],section
        checked_tables+=1
    for section,sentence in narrative.expected_sentences(*narrative.load()):
        if section=='Abstract':
            continue  # The revised abstract summarizes both environments.
        block=re.search(rf'### {re.escape(section)} .*?\n(.*?)(?=\n### |\n## |\Z)',manuscript,re.S).group(1)
        # A complete sentence must not match a truncated decimal such as 65.97.43.
        boundary = r'(?=\s|$)' if sentence.endswith('.') else ''
        assert re.search(re.escape(sentence) + boundary, block), (section,sentence)

    # Independent formatting of Table 7 directly from each extended summary.
    expected=[]
    summaries=[{(r['case'],r['solver']):float(r['median_ns'])/1e6 for r in rows(p/'summary.csv')}
               for p in (EXTENSION,COLLECTION/'extended')]
    for d in read(EXTENSION/'plan.json')['timing']:
        if (d['family']=='large_square' and d['n']==5000) or (d['family']=='large_rectangle' and (d['n'],d['m'])==(4000,5000)) or d['family']=='signed':
            ref='scipy' if d['regime']=='required' else 'scipy_capped'
            expected.append([float(f'{s[d["case"],m]:.3f}') for s in summaries for m in ('jvrust',ref)])
    block=re.search(r'### 6\.7 .*?\n(.*?)(?=\n### )',manuscript,re.S).group(1)
    assert [v for _,v in helpers.numeric_rows(block)]==expected
    checked_tables+=1
    assert result['settings']==994 and result['cases']==204
    assert sum(s['timing_samples_per_run'] for s in result['suites'].values())==8946
    assert sum(s['records'] for s in result['correctness'].values())==3520
    assert result['max_absolute_memory_median_difference_mib']<.09
    assert result['edge_checks_identical']==129
    for method in ('jvrust','jvcpp'):
        assert sum(s['cases'] for s in result['primary_comparisons'][method].values())==201
        assert all(s['ordering_changes']==0 for s in result['primary_comparisons'][method].values())

    # Check extended qualitative claims and quantitative ranges in both runs.
    ranges=[]
    for folder,summary in zip((EXTENSION,COLLECTION/'extended'),summaries):
        plan=read(folder/'plan.json')['timing']
        large=[summary[d['case'],'jvcpp']/summary[d['case'],'jvrust'] for d in plan if d['family'] in ('large_square','large_rectangle')]
        tall=[summary[d['case'],'jvcpp']/summary[d['case'],'jvrust'] for d in plan if d['family']=='extreme_rectangle' and d['n']>d['m']]
        ranges.append({'large_cpp_over_rust':[min(large),max(large)],'extreme_tall_cpp_over_rust':[min(tall),max(tall)]})
        for d in plan:
            ref='scipy' if d['regime']=='required' else 'scipy_capped'
            if d['family'] in ('large_square','large_rectangle','extreme_rectangle','forbidden','ties','signed'):
                assert all(summary[d['case'],m]<summary[d['case'],ref] for m in ('jvrust','jvcpp'))
        for r in (read(folder/'run.json'),):
            assert r['status']=='completed' and r['excluded_method_settings']==0 and not r['failures']
    for version,info in zip(('3.14','3.11'),ranges):
        values=info['large_cpp_over_rust']
        assert f'{values[0]:.3f}–{values[1]:.3f} in Python {version}' in manuscript
    assert f"{result['max_absolute_memory_median_difference_mib']:.5f} MiB" in manuscript

    for base in (PAPER/'results',COLLECTION/'results'):
        assert all(s['exit_code']==0 for s in read(base/'test_validation.json')['suites'].values())
        assert read(base/'batch_validation.json')['checked_outputs']==11040
        assert read(base/'validation.json')['maximum_absolute_objective_error']==0
        assert read(base/'formulations/validation.json')['maximum_absolute_grid_objective_error']==0

    figure_sources=read(PAPER/'publication/figure_sources.json')
    image_stems=[Path(p).stem for p in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',manuscript)]
    assert set(image_stems)=={Path(n).stem for n in figure_sources}
    for name,record in figure_sources.items():
        assert digest(PAPER/record['source'])==record['sha256']==digest(PAPER/'latex/images'/name)
        assert record['preparation_script_sha256']==digest(PAPER/'paper_figures.py')
    build=read(output/'build_report.json'); html=read(output/'html_report.json')
    assert build['manuscript_sha256']==digest(PAPER/'manuscript.md')
    assert build['pdf_sha256']==digest(output/'native_rectangular_assignment.pdf')
    assert build['archive_sha256']==html['archive_sha256']==digest(output/'native_rectangular_assignment_arxiv.zip')
    assert html['preview_sha256']==digest(output/'html/index.html')
    assert not build['log_diagnostics'] and not html['diagnostics']
    assert build['verified_from_extracted_archive'] and not build['shell_escape']
    with zipfile.ZipFile(output/'native_rectangular_assignment_arxiv.zip') as archive:
        assert set(archive.namelist())=={'main.tex','main.bbl','references.bib'}|{'images/'+n for n in figure_sources}
        assert all(hashlib.sha256(archive.read(n)).hexdigest()==sha for n,sha in build['files'].items())
    assert preserved()==count
    report={'status':'passed','frozen_evidence_files_unchanged':count,'timing_samples_per_environment':8946,
            'recomputed_comparison_matches':True,'numeric_tables_checked':checked_tables,'publication_figures':len(figure_sources),
            'extended_implementation_ranges':ranges,'manuscript_sha256':digest(PAPER/'manuscript.md'),
            'archive_sha256':build['archive_sha256'],'pdf_sha256':build['pdf_sha256'],
            'pdf_pages':build['pages'],'html_counts':html['counts'],'solver_measurements_rerun':False}
    (output/'evidence_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    verify()
