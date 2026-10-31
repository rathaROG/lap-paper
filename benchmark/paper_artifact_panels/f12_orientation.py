from paper_artifacts_common import REPO, DATA, save
"""Draw only saved summary ratios; no benchmark or solver imports."""
from pathlib import Path
import csv,json,collections,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator

base=DATA
records=list(csv.DictReader((DATA/'F12_orientation.csv').open()))
assert len(records)==296
assert collections.Counter(r['environment'] for r in records)==dict.fromkeys(['L311','L314','W311','W314'],74)
assert len({(r['environment'],r['objective'],r['shape'],r['method']) for r in records})==296
styles=json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))
styles.update({
 'scipy':dict(label='SciPy reference',color='#000000',marker='*'),
 'scipy:capped':dict(label='SciPy capped',color='#000000',marker='*'),
 'scipy:dummies':dict(label='Dummy columns',color='#666666',marker='>'),
 'scipy:diagonal_dummies':dict(label='Diagonal dummies',color='#999999',marker='<'),
 'scipy:square':dict(label='SciPy square',color='#444444',marker='8'),
})
methods={
 'required':['jvrust','jvhybrid','jvcpp','jvx','jvs','scipy'],
 'optional':['jvrust','jvcpp','jvx','scipy:capped','scipy:dummies','scipy:diagonal_dummies','scipy:square','sparse:csgraph','sparse:lapmod']}
shapes=['1000x2000','2000x3000','4000x5000','64x4096','64x8192','64x16384']
rowkeys=[(obj,shape) for obj in ['required','optional'] for shape in shapes]
lookup={(r['environment'],r['objective'],r['shape'],r['method']):r for r in records}
fig,axes=plt.subplots(1,4,figsize=(16.6,10.0),sharex=True,sharey=True)
plt.rcParams.update({'font.family':'DejaVu Sans'})
titles={'L311':'Linux · Python 3.11.18','L314':'Linux · Python 3.14.10','W311':'Windows · Python 3.11.18','W314':'Windows · Python 3.14.10'}
for ax,en in zip(axes,titles):
 ax.set_xscale('log');ax.set_xlim(.73,95);ax.set_ylim(11.65,-.65)
 ax.axvspan(.97,1.03,color='#dddddd',zorder=0);ax.axvline(1,color='#777777',lw=.8)
 for y in range(12):ax.axhline(y+.5,color='#e6e6e6',lw=.6,zorder=0)
 ax.axhline(5.5,color='#888888',lw=1)
 for i,(obj,shape) in enumerate(rowkeys):
  mm=methods[obj]
  for k,method in enumerate(mm):
   yy=i+(k-(len(mm)-1)/2)*(.64/(len(mm)-1))
   st=styles[method];r=lookup.get((en,obj,shape,method))
   if r:
    ax.scatter(float(r['tall_over_wide']),yy,s=24,c=st['color'],marker=st['marker'],edgecolors='#222222',linewidths=.28,zorder=3)
   else:
    # Coverage is a separate lane outside the numeric ratio axis.
    ax.add_patch(Rectangle((-.068,yy-.042),.043,.084,transform=ax.get_yaxis_transform(),facecolor='#f5f5f5',edgecolor='#999999',hatch='////',lw=.35,clip_on=False))
    ax.plot([-.046],[yy],transform=ax.get_yaxis_transform(),marker=st['marker'],markersize=3.4,color=st['color'],markeredgecolor='#333333',markeredgewidth=.25,clip_on=False)
 ax.text(-.046,1.018,'*',transform=ax.transAxes,ha='center',va='bottom',fontsize=10,color='#777777')
 ax.set_title(en,fontsize=16,pad=22)
 ax.xaxis.set_major_locator(FixedLocator([1,3,10,30]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}×'))
 ax.xaxis.set_minor_locator(NullLocator());ax.tick_params(axis='x',labelsize=9);ax.tick_params(axis='y',length=0,labelsize=9,pad=24)
 ax.grid(axis='x',color='#eeeeee',lw=.6)
 for s in ['top','right','left']:ax.spines[s].set_visible(False)
 ax.spines['bottom'].set_color('#aaaaaa')
 axes[0].set_yticks(range(12),[f"{shape.replace('x',' × ')} ↔ {' × '.join(reversed(shape.split('x')))}  {'R' if obj=='required' else 'F'}" for obj,shape in rowkeys])

legend_groups=[('Native',['jvrust','jvhybrid','jvcpp'],.28),
               ('Legacy dense',['jvx','jvs'],.43),
               ('SciPy formulations',['scipy','scipy:dummies','scipy:diagonal_dummies','scipy:square'],.65),
               ('Sparse',['sparse:csgraph','sparse:lapmod'],.88)]
for title,legend_methods,anchor in legend_groups:
 handles=[Line2D([],[],linestyle='',marker=styles[m]['marker'],color=styles[m]['color'],markeredgecolor='#222222',markeredgewidth=.3,markersize=5,label=styles[m]['label']) for m in legend_methods]
 leg=fig.legend(handles=handles,title=title,loc='upper center',bbox_to_anchor=(anchor,.995),ncol=1,frameon=False,fontsize=8.3,title_fontsize=8.8,handletextpad=.5,labelspacing=.4)
 leg.get_title().set_weight('bold')

fig.text(.55,.084,'Tall / wide complete-call time (×; logarithmic)',ha='center',fontsize=10)
fig.text(.055,.043,'R: required. F: finite limit. Hatched symbols: not planned. Gray band: ±3%.',fontsize=8.8)

fig.subplots_adjust(left=.25,right=.99,bottom=.14,top=.80,wspace=.22)
save(fig, 'F12_orientation')
plt.close(fig)
(DATA/'F12_orientation_assertions.json').write_text(json.dumps({
 'total_plotted_estimates':len(records),'method_pairs_per_environment':dict(collections.Counter(r['environment'] for r in records)),
 'case_pairs_per_environment':12,'source_csv':str(DATA/'F12_orientation.csv'),
 'source_csv_sha256':hashlib.sha256((DATA/'F12_orientation.csv').read_bytes()).hexdigest(),
 'legend_groups':[g[0] for g in legend_groups],
 'absence_encoding':'Hatched method glyph in * lane outside numeric axis: not timed by protocol, not a measurement failure.',
 'no_estimates_removed':True,'estimator':'tall case seconds / wide case seconds; separate attempts; no environment pooling'
 },indent=2)+'\n')
print('Generated F12_orientation')
