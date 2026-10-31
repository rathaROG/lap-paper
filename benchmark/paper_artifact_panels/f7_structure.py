from paper_artifacts_common import REPO, DATA, save

import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
BASE = DATA
RUNS = REPO / 'benchmark/runs'
ENV = [
    ('L311', 'Linux 3.11', 'linux-py3.11.18-20261031-0455', '-'),
    ('L314', 'Linux 3.14', 'linux-py3.14.10-20261031-0630', (0,(5,2))),
    ('W311', 'Windows 3.11', 'windows-py3.11.18-20261030-2322', (0,(1,1.5))),
    ('W314', 'Windows 3.14', 'windows-py3.14.10-20261031-0322', (0,(5,1.5,1,1.5))),
]
FAMILIES = [('congestion','Congestion','row_fraction',[.25,.5,.625,.75,.9]),
            ('forbidden','Forbidden entries','forbidden',[0,.5,.9,.99]),
            ('ties','Quantized ties','levels',[1,2,8,256])]
STYLE = json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))
records=[]; limits={}; omitted=[]
for env,label,run,ls in ENV:
    root=RUNS/run/'paper-run-light'
    manifest=json.loads((root/'campaign-session/manifest.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'campaign-summary.json').read_text(encoding='utf-8'))
    assert manifest['profile']==summary['profile']=='light'
    estimates={(e['case_id'],e['method']):e for e in summary['estimates']}
    paired={(e['case_id'],e['method']):e for e in summary['paired']}
    cases=[c for c in manifest['cases'] if c['kind']=='single' and c['family'] in {f[0] for f in FAMILIES}]
    assert len(cases)==26
    omitted=[c['id'] for c in manifest['cases'] if c['family']=='signed']
    assert len(omitted)==4
    for family,title,parameter,levels in FAMILIES:
        for objective in ['required','optional']:
            group=[c for c in cases if c['family']==family and c['objective']==objective]
            assert sorted(c['input_group']['parameters'][parameter] for c in group)==levels
            fixed_limits={c['limit'] for c in group};assert len(fixed_limits)==1
            limits[f'{family}:{objective}']=next(iter(fixed_limits))
            for case in group:
                reference=case['metadata']['primary_comparator']
                assert reference==('scipy' if objective=='required' else 'scipy:capped')
                ref=estimates[case['id'],reference]
                natives=['jvrust','jvhybrid','jvcpp'] if objective=='required' else ['jvrust','jvcpp']
                for identity in natives:
                    method=identity
                    e=estimates[case['id'],method]
                    assert e['status']==ref['status']=='complete'
                    for key in ['session_id','attempt_id','input_hashes']:
                        assert e[key]==ref[key]
                    ratio=ref['seconds']/e['seconds']
                    assert math.isfinite(ratio) and ratio>0
                    r=paired[case['id'],method]
                    assert r['status']=='complete' and r['reference']==reference
                    assert math.isclose(ratio,r['ratio'],rel_tol=1e-14)
                    records.append(dict(environment=env,environment_label=label,run=run,family=family,objective=objective,
                                        parameter=parameter,parameter_value=case['input_group']['parameters'][parameter],
                                        case_id=case['id'],rows=case['rows'],cols=case['cols'],limit=case['limit'],gate=case['gate'],
                                        identity=identity,method=method,reference=reference,seconds=e['seconds'],
                                        reference_seconds=ref['seconds'],scipy_over_method=ratio,
                                        session_id=e['session_id'],attempt_id=e['attempt_id'],
                                        input_hashes_json=json.dumps(e['input_hashes'],sort_keys=True),
                                        source=str(root/'campaign-summary.json')))
assert len(records)==260
assert all(r['identity'] in ['jvrust','jvhybrid','jvcpp'] for r in records)
for env,*_ in ENV:
    for family,title,parameter,levels in FAMILIES:
        for objective in ['required','optional']:
            for method in (['jvrust','jvhybrid','jvcpp'] if objective=='required' else ['jvrust','jvcpp']):
                rs=[r for r in records if r['environment']==env and r['family']==family and r['objective']==objective and r['identity']==method]
                assert sorted(r['parameter_value'] for r in rs)==levels
with (BASE/'F7_structure_lines_data.csv').open('w',encoding='utf-8',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelsize':9.5,'savefig.facecolor':'white'})
fig,axes=plt.subplots(2,3,figsize=(13.3,8.35),sharey=True)
fig.subplots_adjust(left=.075,right=.985,bottom=.20,top=.80,wspace=.17,hspace=.60)
for row,objective in enumerate(['required','optional']):
    for col,(family,title,parameter,levels) in enumerate(FAMILIES):
        ax=axes[row,col]
        ax.axhspan(.97,1.03,color='#d9d9d9',alpha=.65,lw=0,zorder=0)
        ax.axhline(1,color='#777777',lw=.65,zorder=0)
        natives=['jvrust','jvhybrid','jvcpp'] if objective=='required' else ['jvrust','jvcpp']
        for identity in natives:
            style=STYLE[identity]
            for env,label,run,ls in ENV:
                rs=sorted([r for r in records if r['environment']==env and r['family']==family and r['objective']==objective and r['identity']==identity],key=lambda r:r['parameter_value'])
                ax.plot([r['parameter_value'] for r in rs],[r['scipy_over_method'] for r in rs],
                        color=style['color'],marker=style['marker'],linestyle=ls,lw=.95,alpha=.62,
                        markersize=4.0,markeredgewidth=.25,markeredgecolor='white',zorder=3)
        # Matching SciPy is the reference numerator: retain only the neutral 1x rule and tie band.
        ax.set_yscale('log');ax.set_ylim(.75,20)
        ax.yaxis.set_major_locator(FixedLocator([1,2,4,8,16]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}x'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y',which='major',color='#dedede',lw=.55,zorder=0)
        ax.set_xticks(levels)
        if family=='ties':
            ax.set_xscale('log',base=2);ax.set_xlim(.8,330)
            ax.set_xticks(levels,[str(x) for x in levels]);ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Number of cost levels (log base 2)',labelpad=7)
        else:
            ax.set_xlim((.20,.94) if family=='congestion' else (-.04,1.04))
            ax.set_xticks(levels,[f'{x*100:g}%' for x in levels])
            ax.set_xlabel('Congested row fraction' if family=='congestion' else 'Forbidden-entry probability',labelpad=7)
        if row == 0:
            ax.set_xlabel('')
        if family == 'forbidden':
            for tick in ax.get_xticklabels():
                tick.set_rotation(65); tick.set_ha('right')
        fixed=limits[f'{family}:{objective}']
        subtitle='Required assignment' if objective=='required' else f'Optional assignment, L={fixed:g}'
        ax.set_title(f'{title}\n{subtitle}',fontsize=10.8,pad=10)
        if col==0:
            ax.set_ylabel('SciPy / native time (×)',labelpad=9)
# Explicitly preserve the near-one congestion exception without favorably selecting the other levels.
axes[1,0].annotate('Rust at 90%:\nW311 0.916×\nW314 0.926×',xy=(.9,.9155544836649819),xycoords='data',
                   xytext=(.42,2.4),textcoords='data',ha='left',va='bottom',fontsize=8.3,
                   bbox=dict(boxstyle='round,pad=.22',facecolor='white',edgecolor='#cccccc',alpha=.96),
                   arrowprops=dict(arrowstyle='-',lw=.7,color='#555555'),zorder=8)

method_handles=[Line2D([0],[0],color=STYLE[m]['color'],marker=STYLE[m]['marker'],lw=1.2,markersize=5,
                       label=STYLE[m]['label']) for m in ['jvrust','jvhybrid','jvcpp']]
fig.legend(handles=method_handles,loc='upper center',bbox_to_anchor=(.55,.99),ncol=3,frameon=False,columnspacing=2.3)
env_handles=[Line2D([0],[0],color='#444444',linestyle=ls,lw=1.4,label=env) for env,label,_,ls in ENV]
fig.legend(handles=env_handles,loc='upper center',bbox_to_anchor=(.55,.93),ncol=4,frameon=False,columnspacing=2.0)
fig.text(.075,.078,'One curve per method/environment. Gray band: ±3%.',fontsize=8.9)

save(fig, 'F7_structure')
plt.close(fig)
audit={'sources':'Only saved four light campaign summaries and manifests; no solver import or calls.',
       'case_count':26,'native_ratio_points':260,'reference_points':0,'all_csv_rows':len(records),
       'scipy_reference_glyphs_or_curve_drawn':False,'neutral_one_rule_and_tie_band_only':True,
       'no_cross_environment_pooling':True,'all_levels_present_for_each_applicable_method_and_environment':True,
       'same_session_attempt_and_input_hashes_for_each_ratio':True,'fixed_limit_within_each_curve':True,
       'limits_by_family_objective':limits,'required_favored_column_fraction':.25,
       'signed_cases_not_connected_as_a_sweep':omitted,
       'optional_congestion_case041_jvrust_ratios':{env:next(r['scipy_over_method'] for r in records if r['environment']==env and r['identity']=='jvrust' and r['case_id']=='extended:041_congestion_optional_256x320') for env,*_ in ENV},
       'matching_reference':'SciPy seconds remain as the explicit reference numerator in every native CSV record; no synthetic reference point or curve is plotted.'}
(BASE/'F7_structure_lines_assertions.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Generated F7_structure')
