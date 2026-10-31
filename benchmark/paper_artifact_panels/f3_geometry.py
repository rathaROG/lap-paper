from paper_artifacts_common import REPO, DATA, save
"""Connected geometry curves from saved light results only.

Uses the agreed connected-curve design. No benchmark or solver imports.
Main plot: squares; the exact 5:4 companion data use the same selector.
"""
from pathlib import Path
import csv,json,math,hashlib,collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator,LogLocator

OUT=DATA
RUNS=REPO/'benchmark/runs'
ENVS=[('L311','Linux 3.11.18','linux-py3.11.18-20261031-0455','-'),
      ('L314','Linux 3.14.10','linux-py3.14.10-20261031-0630',(0,(5,2))),
      ('W311','Windows 3.11.18','windows-py3.11.18-20261030-2322',(0,(1,1.7))),
      ('W314','Windows 3.14.10','windows-py3.14.10-20261031-0322',(0,(5,1.6,1,1.6)))]
STYLE=json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))
STYLE.update({
 'REF':dict(label='SciPy reference',color='#000000',marker='*'),
 'scipy:square':dict(label='SciPy square',color='#444444',marker='8'),
})
COLUMNS=[('required',(.9,.25),'Required\nq=.9, motion=.25δ'),
         ('required',(.6,1.),'Required\nq=.6, motion=δ'),
         ('optional',(.9,.25),'Optional L=1.5δ\nq=.9, motion=.25δ'),
         ('optional',(.6,1.),'Optional L=1.5δ\nq=.6, motion=δ')]
TOP={'required':['jvrust','jvcpp','jvhybrid','jvx','jvs','REF'],
     'optional':['jvrust','jvcpp','jvx','REF','scipy:square','sparse:lapmod','sparse:csgraph']}
BOTTOM={'required':['jvrust','jvcpp','jvhybrid'],
        'optional':['jvrust','jvcpp','sparse:lapmod','sparse:csgraph']}
SIZES=[2**i for i in range(3,13)]
rows=[];audit={'description':'Saved light data only; no environment or setting averaging.',
 'selection':'geometry single; q/motion in {(0.9,0.25),(0.6,1)}; required OR optional L/delta=1.5,gate/delta=None. Do not exclude geometric_sensitivity=true before the 5:4 selection.',
 'estimand':'seconds is saved median of three equal-input round means. Ratio is saved matching-reference estimate / method estimate, checked against paired[].ratio.',
 'environment_line_styles':{k:str(ls) for k,_,_,ls in ENVS},'environments':{}}
canonical_ids=None
for en,label,folder,linestyle in ENVS:
 root=RUNS/folder/'paper-run-light'
 manifest=json.loads((root/'campaign-session/manifest.json').read_text(encoding='utf-8'))
 summary=json.loads((root/'campaign-summary.json').read_text(encoding='utf-8'))
 estimates={(e['case_id'],e['method']):e for e in summary['estimates'] if e['status']=='complete'}
 paired={(p['case_id'],p['method']):p for p in summary['paired'] if p['status']=='complete'}
 selected=[]
 for c in manifest['cases']:
  md=c['metadata']
  if c['kind']!='single' or c['family']!='geometry':continue
  if (md['q'],md['motion']) not in [(.9,.25),(.6,1.)]:continue
  if c['objective']=='optional' and not (md.get('limit_over_delta')==1.5 and md.get('gate_over_delta') is None):continue
  selected.append(c)
 assert len(selected)==80
 assert len([c for c in selected if c['rows']==c['cols']])==40
 assert len([c for c in selected if c['cols']==5*c['rows']//4 and c['cols']!=c['rows']])==40
 # The 256x320 primary optional cases are flagged as sensitivity cases.
 sensitivity=[c['id'] for c in selected if c['metadata']['geometric_sensitivity']]
 assert len(sensitivity)==2 and all('256x320' in c for c in sensitivity)
 ids=sorted(c['id'] for c in selected)
 if canonical_ids is None:canonical_ids=ids
 else:assert canonical_ids==ids
 local=0
 for c in selected:
  md=c['metadata'];ref=md['primary_comparator'];re=estimates[c['id'],ref]
  assert ref==('scipy' if c['objective']=='required' else 'scipy:capped')
  for token in TOP[c['objective']]:
   method=ref if token=='REF' else token
   planned=method in c['methods']
   if not planned:
    assert method in {'jvx','jvs','scipy:square'} and c['rows']>1024
    continue
   e=estimates[c['id'],method]
   assert e['input_hashes']==re['input_hashes'] and e['attempt_id']==re['attempt_id']
   ratio=re['seconds']/e['seconds']
   if method!=ref:
    p=paired[c['id'],method]
    assert p['reference']==ref and math.isclose(p['ratio'],ratio,rel_tol=1e-12)
   rows.append(dict(environment=en,environment_label=label,run=folder,
     shape='square' if c['rows']==c['cols'] else '5:4',objective=c['objective'],q=md['q'],motion=md['motion'],
     rows=c['rows'],cols=c['cols'],case_id=c['id'],geometric_sensitivity=md['geometric_sensitivity'],
     method_token=token,method=method,seconds=e['seconds'],milliseconds=e['seconds']*1000,
     reference=ref,reference_seconds=re['seconds'],ratio_reference_over_method=ratio,
     in_ratio_row=token in BOTTOM[c['objective']],attempt_id=e['attempt_id'],
     reference_attempt_id=re['attempt_id'],session_id=e['session_id'],environment_id=e['environment_id'],
     input_hashes=json.dumps(e['input_hashes'],sort_keys=True)))
   local+=1
 assert local==488
 audit['environments'][en]=dict(selected_cases=80,square_cases=40,rectangle_cases=40,
  selected_sensitivity_cases=sensitivity,all_shape_method_points=local,
  manifest_sha256=hashlib.sha256((root/'campaign-session/manifest.json').read_bytes()).hexdigest())

for shape,filename in [('square','F3_geometry.csv'),('5:4','F3_geometry_5to4_companion.csv')]:
 rr=[r for r in rows if r['shape']==shape]
 assert len(rr)==976
 assert sum(r['in_ratio_row'] for r in rr)==560
 for en,_,_,_ in ENVS:
  for obj,setting,_ in COLUMNS:
   for token in TOP[obj]:
    seq=[r for r in rr if r['environment']==en and r['objective']==obj and (r['q'],r['motion'])==setting and r['method_token']==token]
    expected=SIZES[:8] if token in {'jvx','jvs','scipy:square'} else SIZES
    assert sorted(r['rows'] for r in seq)==expected
 with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)

square=[r for r in rows if r['shape']=='square']
audit.update(square_top_method_points=len(square),square_ratio_method_points=sum(r['in_ratio_row'] for r in square),
             per_environment_top_method_points=244,per_environment_ratio_method_points=140,
             per_series_sizes=SIZES,cutoff_last_observed_rows=1024,cutoff_first_not_planned_rows=2048,
             main_shape='square',companion_shape='5:4',original_user_script_preserved=True)
audit['ratio_range']=[min(r['ratio_reference_over_method'] for r in square if r['in_ratio_row']),max(r['ratio_reference_over_method'] for r in square if r['in_ratio_row'])]
(OUT/'F3_connected_geometry_assertions.json').write_text(json.dumps(audit,indent=2)+'\n')

for shape in ['square','5:4']:
 square=[r for r in rows if r['shape']==shape]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
  'axes.grid':True,'grid.color':'#e5e5e5','grid.linewidth':.55,'axes.edgecolor':'#aaaaaa',
  'axes.titleweight':'bold','axes.titlelocation':'left','legend.frameon':False})
 fig,axes=plt.subplots(2,4,figsize=(15.8,8.7),sharex=True,sharey='row',gridspec_kw={'height_ratios':[1.65,1]})
 for col,(obj,setting,title) in enumerate(COLUMNS):
  top,bottom=axes[0,col],axes[1,col]
  for token in TOP[obj]:
   st=STYLE[token]
   for env_index,(en,label,folder,linestyle) in enumerate(ENVS):
    seq=sorted([r for r in square if r['environment']==en and r['objective']==obj and (r['q'],r['motion'])==setting and r['method_token']==token],key=lambda r:r['rows'])
    # Shift marker positions by environment along the exact common size grid;
    # no jitter or averaging alters the measured values.
    kwargs=dict(color=st['color'],lw=1.0,alpha=.88,ls=linestyle,marker=st['marker'],
       ms=3.6,mec='white',mew=.25,markevery=(env_index,4))
    top.plot([r['rows'] for r in seq],[r['milliseconds'] for r in seq],**kwargs)
    if token in BOTTOM[obj]:
     bottom.plot([r['rows'] for r in seq],[r['ratio_reference_over_method'] for r in seq],**kwargs)
  top.set_title(title,fontsize=10,pad=11)
  for ax in (top,bottom):
   ax.set_xscale('log',base=2);ax.set_xlim(7,6000)
   ax.xaxis.set_major_locator(FixedLocator([8,32,128,512,4096]));ax.xaxis.set_minor_locator(NullLocator())
   ax.xaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
   # Boundary at last planned point, not a continuation of excluded baselines.
   ax.axvline(1024,color='#999999',ls=(0,(2,2)),lw=.75,zorder=0)
  top.set_yscale('log');top.set_ylim(.001,3000)
  top.yaxis.set_major_locator(LogLocator(base=10,numticks=8));top.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
  bottom.set_yscale('log');bottom.set_ylim(.03,12)
  bottom.yaxis.set_major_locator(FixedLocator([.05,.1,.2,.5,1,2,5,10]));bottom.yaxis.set_minor_locator(NullLocator())
  bottom.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}×'))
  bottom.axhspan(.97,1.03,color='#dedede',lw=0,zorder=0);bottom.axhline(1,color='#555555',lw=.8)
  bottom.set_xlabel('Rows (' + shape + ')')

 axes[0,0].set_ylabel('Complete call (ms)')
 axes[1,0].set_ylabel('SciPy / method (×)')
 assert min(r['ratio_reference_over_method'] for r in square if r['in_ratio_row'])>.03
 assert max(r['ratio_reference_over_method'] for r in square if r['in_ratio_row'])<12
 assert min(r['milliseconds'] for r in square)>.001 and max(r['milliseconds'] for r in square)<3000

 mh=[Line2D([],[],color=STYLE[t]['color'],marker=STYLE[t]['marker'],lw=1.1,ms=4,label=STYLE[t]['label'])
     for t in ['jvrust','jvcpp','jvhybrid','jvx','jvs','REF','scipy:square','sparse:lapmod','sparse:csgraph']]
 fig.legend(handles=mh,loc='upper center',bbox_to_anchor=(.52,.995),ncol=5,fontsize=8.8,columnspacing=1.8,handletextpad=.4)
 eh=[Line2D([],[],color='#555555',lw=1.25,ls=ls,label=en) for en,label,folder,ls in ENVS]
 fig.legend(handles=eh,loc='upper center',bbox_to_anchor=(.52,.90),ncol=4,fontsize=8.6,handlelength=3.4,columnspacing=2)

 fig.text(.10,.025,'Vertical rule: legacy/expanded baselines stop at 1024 rows. Optional inputs have no gate.',fontsize=8.5)


 fig.subplots_adjust(left=.085,right=.975,bottom=.15,top=.78,hspace=.18,wspace=.19)
 save(fig, 'F3_geometry' if shape == 'square' else 'A1_geometry_5to4');plt.close(fig)
