from paper_artifacts_common import REPO, DATA, save
"""Read saved summaries only. No repository imports, input generation or solvers."""
from pathlib import Path
import collections
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
import numpy as np

ROOT = REPO.parent
OUT = DATA
OUT.mkdir(parents=True, exist_ok=True)
RUNS = [
    ('L311', 'Linux · Python 3.11', 'linux-py3.11.18-20261031-0455'),
    ('L314', 'Linux · Python 3.14', 'linux-py3.14.10-20261031-0630'),
    ('W311', 'Windows · Python 3.11', 'windows-py3.11.18-20261030-2322'),
    ('W314', 'Windows · Python 3.14', 'windows-py3.14.10-20261031-0322'),
]
_shared = json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))
STYLE = {m: (_shared[m]['color'], _shared[m]['marker'], _shared[m]['label'])
         for m in ['jvrust', 'jvhybrid', 'jvcpp']}
FAMILIES = [('base', 'Base'), ('formulations', 'Formulations'),
            ('scaling', 'Uniform / blocks'), ('large_square', 'Large square'),
            ('large_rectangle', 'Large rectangle'), ('extreme_rectangle', 'Extreme rectangle'),
            ('geometry', 'Tracking geometry'), ('congestion', 'Congestion'),
            ('forbidden', 'Forbidden entries'), ('ties', 'Tied costs'),
            ('signed', 'Signed costs'), ('rejection', 'Rejection sweep')]
plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False,
                     'savefig.dpi': 210, 'font.family': 'DejaVu Sans'})
loaded = []
for alias, title, run in RUNS:
    p = REPO / 'benchmark/runs' / run / 'paper-run-light'
    summary = json.loads((p/'campaign-summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((p/'campaign-session/manifest.json').read_text(encoding='utf-8'))
    cases = {c['id']: c for c in manifest['cases']}
    assert summary['profile'] == 'light'
    assert summary['coverage']['complete_attempts'] == 3286
    assert not summary['unobserved_planned_conditions']
    pairs = [x for x in summary['paired'] if x['status'] == 'complete'
             and x['method'] in STYLE and cases[x['case_id']]['kind'] == 'single']
    assert len(pairs) == 638
    assert all(x['reference'] == cases[x['case_id']]['metadata']['primary_comparator'] for x in pairs)
    assert all(x['attempt_id'] == x['reference_attempt_id'] for x in pairs)
    loaded.append((alias, title, cases, pairs))

rows = [(family, label, objective)
        for family, label in FAMILIES for objective in ['required', 'optional']
        if any(c['family'] == family and c['objective'] == objective
               for c in loaded[0][2].values() if c['kind'] == 'single')]
fig, axes = plt.subplots(1, 4, figsize=(10.7, 10.1), sharex=True, sharey=True)
offset = {'jvrust': -.22, 'jvhybrid': 0, 'jvcpp': .22}
audit = []
for ax, (alias, title, cases, pairs) in zip(axes, loaded):
    ax.set_xscale('log')
    ax.set_xlim(.78, 20)
    ax.set_ylim(len(rows)-.48, -.55)
    ax.axvspan(.97, 1.03, color='#d9d9d9', zorder=0)
    ax.axvline(1, color='#555555', linewidth=.8, zorder=1)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 4, 8, 16]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}×'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', color='#dddddd', linewidth=.45)
    for yi, (family, label, objective) in enumerate(rows):
        if yi % 2 == 0:
            ax.axhspan(yi-.45, yi+.45, color='#f5f5f5', zorder=-1)
        for method, (color, marker, name) in STYLE.items():
            points = sorted([x for x in pairs if x['method'] == method
                             and cases[x['case_id']]['family'] == family
                             and cases[x['case_id']]['objective'] == objective],
                            key=lambda x:x['case_id'])
            if not points:
                continue
            # Case-ID-based deterministic displacement, never sorted by performance.
            yy = yi + offset[method] + np.array([((i*7)%11-5)*.011 for i in range(len(points))])
            ax.scatter([x['ratio'] for x in points], yy, marker=marker, s=8,
                       color=color, alpha=.75, edgecolors='white', linewidths=.18, zorder=3)
            for x in points:
                audit.append({'environment': alias, 'family':family, 'objective':objective,
                              'case_id':x['case_id'], 'method':method, 'reference':x['reference'],
                              'ratio':x['ratio'], 'seconds':x['seconds'],
                              'reference_seconds':x['reference_seconds'],
                              'classification': 'win' if x['ratio']>1.03 else 'loss' if x['ratio']<.97 else 'tie'})
            # Directly label the hardest native case, including near ties.
            if family == 'congestion' and objective == 'optional' and method in ['jvrust', 'jvcpp']:
                worst = min(points, key=lambda x:x['ratio'])
                ax.annotate(f"{worst['ratio']:.3f}×", (worst['ratio'],yi+offset[method]),
                            xytext=(8,-3 if method=='jvcpp' else 4),textcoords='offset points',
                            fontsize=10.3,color=color,weight='bold',zorder=5,
                            bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=.3))
    ax.set_title(alias, fontsize=12, weight='bold', pad=12)
axes[0].set_yticks(range(len(rows)))
labels = []
for family, label, objective in rows:
    n = sum(c['family']==family and c['objective']==objective and c['kind']=='single'
            for c in loaded[0][2].values())
    labels.append(f"{label} · {'R' if objective=='required' else 'F'}  ({n})")
axes[0].set_yticklabels(labels)


handles = [Line2D([],[],linestyle='',marker=marker,color=color,label=name,markersize=5)
           for color,marker,name in STYLE.values()]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.62,.99),ncol=3,frameon=False,fontsize=9)
fig.supxlabel('Matching SciPy / native time (×; logarithmic)',y=.04,fontsize=10)
fig.text(.08,.012,'R: required. F: finite limit. Each point is one case; gray band: ±3%.',fontsize=8)
fig.subplots_adjust(left=.255,right=.99,bottom=.09,top=.92,wspace=.10)
save(fig, 'F2_native_overview')
plt.close(fig)
with (OUT/'F2_native_overview_data.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
print(f'Generated F2_native_overview; {len(audit)} plotted ratios.')
