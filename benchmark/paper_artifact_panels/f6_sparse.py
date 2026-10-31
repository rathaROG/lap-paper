from paper_artifacts_common import REPO, DATA, save
"""Read saved benchmark summaries only; no benchmark imports, generation or solves."""

import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT=REPO/'benchmark/runs'
OUT=DATA
RUNS=[('L311','Linux · Python 3.11','linux-py3.11.18-20261031-0455'),
      ('L314','Linux · Python 3.14','linux-py3.14.10-20261031-0630'),
      ('W311','Windows · Python 3.11','windows-py3.11.18-20261030-2322'),
      ('W314','Windows · Python 3.14','windows-py3.14.10-20261031-0322')]
STYLES=json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))

plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':11,
 'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':10,'figure.dpi':140,
 'savefig.dpi':190,'axes.spines.top':False,'axes.spines.right':False,
 'axes.titleweight':'bold','axes.titlelocation':'center','axes.spines.left':True,'axes.edgecolor':'#888888','axes.axisbelow':True,'font.family':'DejaVu Sans'})
TIE=(.97,1.03)
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def csvload(path):return list(csv.DictReader(path.open(encoding='utf-8-sig',newline='')))
def writecsv(path,rows):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def axes_style(ax,xlim,ylim,xticks,yticks):
 ax.set_xscale('log');ax.set_yscale('log');ax.set_xlim(*xlim);ax.set_ylim(*ylim)
 ax.axvspan(*TIE,color='#e7e5df',zorder=0);ax.axhspan(*TIE,color='#e7e5df',zorder=0)
 ax.axvline(1,color='#777777',lw=.9);ax.axhline(1,color='#777777',lw=.9)
 ax.grid(True,which='major',color='#ddddda',linewidth=.55)
 for axis,ticks in ((ax.xaxis,xticks),(ax.yaxis,yticks)):
  axis.set_major_locator(FixedLocator(ticks));axis.set_minor_locator(NullLocator())
  axis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}×'))
def handles(keys):
 return [Line2D([],[],marker=STYLES[k]['marker'],ls='none',color=STYLES[k]['color'],
                markersize=7,label=STYLES[k]['label']) for k in keys]
def classify(r):return 'win' if r>1.03 else 'loss' if r<.97 else 'tie'

def sparse():
 rows=[];audit=[]
 for env,title,run in RUNS:
  p=ROOT/run/'paper-run-light';s=load(p/'campaign-summary.json');m=load(p/'campaign-session/manifest.json')
  cases={c['id']:c for c in m['cases']}
  planned={c['id'] for c in m['cases'] if 'sparse:csgraph' in c['methods']}
  assert len(planned)==166
  estimates={(e['case_id'],e['method']):e for e in s['estimates'] if e['status']=='complete'}
  paired={(e['case_id'],e['method']):e for e in s['paired'] if e['status']=='complete'}
  for case in sorted(planned):
   c=cases[case];rust=estimates[(case,'jvrust')];dense=estimates[(case,'scipy:capped')]
   for method in ('sparse:csgraph','sparse:lapmod'):
    e=estimates[(case,method)];q=paired[(case,method)]
    assert q['reference']=='scipy:capped'
    for key in ('attempt_id','session_id','input_hashes'):
     assert e[key]==rust[key]==dense[key],(env,case,key)
    x=q['ratio'];y=rust['seconds']/e['seconds']
    assert math.isclose(x,dense['seconds']/e['seconds'],rel_tol=1e-12)
    assert x>0 and y>0
    rows.append(dict(environment=env,case_id=case,family=c['family'],rows=c['rows'],cols=c['cols'],
     method=method,capped_scipy_over_sparse=x,jvrust_over_sparse=y,
     versus_capped_class=classify(x),versus_jvrust_class=classify(y),
     sparse_seconds=e['seconds'],capped_scipy_seconds=dense['seconds'],jvrust_seconds=rust['seconds'],
     session_id=e['session_id'],attempt_id=e['attempt_id'],source_summary=str(p/'campaign-summary.json'),
     source_manifest=str(p/'campaign-session/manifest.json')))
  assert len([r for r in rows if r['environment']==env])==332
  audit.append(dict(environment=env,planned_cases=166,observed_method_case_points=332,
   all_methods_complete=True,same_session_attempt_input_hashes=True))
 writecsv(OUT/'sparse_bivariate_data.csv',rows)
 fig,axes=plt.subplots(1,4,figsize=(13.0,6.3),sharex=True,sharey=True)
 for ax,(env,title,run) in zip(axes,RUNS):
  axes_style(ax,(.009,9),(.00065,3.2),[.01,.1,1,5],[.001,.01,.1,1,3])
  for method in ('sparse:csgraph','sparse:lapmod'):
   r=[r for r in rows if r['environment']==env and r['method']==method]
   st=STYLES[method]
   ax.scatter([e['capped_scipy_over_sparse'] for e in r],[e['jvrust_over_sparse'] for e in r],
     marker=st['marker'],c=st['color'],s=29,linewidth=.35,edgecolors='white',alpha=.83,zorder=3)
  # Short case codes avoid overlapping dimension labels; the key gives exact dimensions.
  # Every tall case/solver exceeding JVRUST by >3% is labelled.
  winners=[r for r in rows if r['environment']==env and r['rows']>r['cols'] and r['jvrust_over_sparse']>1.03]
  codes={(2000,1000):'A',(3000,2000):'B',(5000,4000):'C'}
  for e in winners:
   is_cs=e['method']=='sparse:csgraph'
   code=codes[(e['rows'],e['cols'])]
   offsets_cs={'B':(-10,13),'C':(-35,25)}
   offsets_lap={'A':(-31,20),'B':(-35,-5),'C':(-2,-30)} if env.startswith('W') else {'B':(-11,13),'C':(-2,-30)}
   xytext=(offsets_cs if is_cs else offsets_lap)[code]
   ax.annotate(code,(e['capped_scipy_over_sparse'],e['jvrust_over_sparse']),
    xytext=xytext,textcoords='offset points',fontsize=9.5,fontweight='bold',color=STYLES[e['method']]['color'],
    arrowprops=dict(arrowstyle='-',color=STYLES[e['method']]['color'],lw=.65),
    bbox=dict(boxstyle='round,pad=.12',fc='white',ec='none',alpha=.82),zorder=5)
  ax.set_title(env,pad=12)

 axes[0].set_ylabel('JVRUST / sparse time (×)')
 fig.supxlabel('Capped SciPy / sparse time (×; logarithmic)',y=.115,fontsize=11)

 fig.legend(handles=handles(['sparse:csgraph','sparse:lapmod']),loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,frameon=False)
 fig.text(.09,.030,'Tall cases: A = 2000×1000; B = 3000×2000; C = 5000×4000.',fontsize=9,color='#555555')

 fig.subplots_adjust(left=.10,right=.98,top=.82,bottom=.23,wspace=.13)
 save(fig, 'F6_sparse');plt.close(fig)
 return audit

sparse()
