from paper_artifacts_common import REPO, DATA, save

import csv, json, math, statistics
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
    ('L314', 'Linux 3.14', 'linux-py3.14.10-20261031-0630', (0, (5, 2))),
    ('W311', 'Windows 3.11', 'windows-py3.11.18-20261030-2322', (0, (1, 1.5))),
    ('W314', 'Windows 3.14', 'windows-py3.14.10-20261031-0322', (0, (5, 1.5, 1, 1.5))),
]
ARMS = [('no_reductions', 'No reductions\n(initializer bypass)'), ('budget1', '1'), ('budget2', '2'), ('reference', '4\n(reference)'), ('budget8', '8')]
STYLE = json.loads((REPO/'benchmark/paper_artifact_panels/style_mapping.json').read_text(encoding='utf-8'))
records = []
nominal_records = []
NOMINAL_ARMS = [('ties_off', 'Free ties off'), ('cutoff_off', 'Cutoff off'), ('ties_cutoff_off', 'Ties + cutoff off')]
case_ids_by_env = {}
for env, env_label, run, ls in ENV:
    root = RUNS / run / 'paper-run-light'
    manifest = json.loads((root / 'campaign-session/manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((root / 'campaign-summary.json').read_text(encoding='utf-8'))
    assert manifest['profile'] == summary['profile'] == 'light'
    assert summary['coverage']['unobserved_planned_case_methods'] == 0
    estimates = {(r['case_id'], r['method']): r for r in summary['estimates']}
    for lang, native in [('rust', 'jvrust'), ('cpp', 'jvcpp')]:
        cases = [c for c in manifest['cases'] if f'policy:{lang}:budget1' in c['methods']]
        assert len(cases) == 31
        case_ids_by_env[f'{env}:{lang}'] = [c['id'] for c in cases]
        for case in cases:
            control = estimates[case['id'], f'policy:{lang}:reference']
            assert control['status'] == 'complete' and control['seconds'] > 0
            for x, (arm, arm_label) in enumerate(ARMS):
                method = f'policy:{lang}:{arm}'
                estimate = estimates[case['id'], method]
                assert estimate['status'] == 'complete' and estimate['seconds'] > 0
                for key in ['session_id', 'attempt_id', 'input_hashes']:
                    assert estimate[key] == control[key], (env, case['id'], method, key)
                ratio = estimate['seconds'] / control['seconds']
                assert math.isfinite(ratio) and ratio > 0
                if arm == 'reference':
                    assert ratio == 1
                records.append(dict(environment=env, environment_label=env_label, run=run, language=lang,
                                    native_identity=native, case_id=case['id'], family=case['family'],
                                    objective=case['objective'], rows=case['rows'], cols=case['cols'],
                                    arm=arm, ordinal_x=x, arm_label=arm_label.replace('\n', ' '),
                                    ablation_seconds=estimate['seconds'], budget4_seconds=control['seconds'],
                                    time_over_budget4=ratio, session_id=estimate['session_id'],
                                    attempt_id=estimate['attempt_id'],
                                    input_hashes_json=json.dumps(estimate['input_hashes'], sort_keys=True),
                                    source=str(root / 'campaign-summary.json')))
            for nominal_index, (arm, arm_label) in enumerate(NOMINAL_ARMS):
                method = f'policy:{lang}:{arm}'
                if method not in case['methods']:
                    continue
                estimate = estimates[case['id'], method]
                assert estimate['status'] == 'complete' and estimate['seconds'] > 0
                for key in ['session_id', 'attempt_id', 'input_hashes']:
                    assert estimate[key] == control[key], (env, case['id'], method, key)
                ratio = estimate['seconds'] / control['seconds']
                assert math.isfinite(ratio) and ratio > 0
                nominal_records.append(dict(environment=env, environment_label=env_label, run=run,
                    language=lang, native_identity=native, case_id=case['id'], family=case['family'],
                    objective=case['objective'], rows=case['rows'], cols=case['cols'], arm=arm,
                    nominal_row=2-nominal_index, arm_label=arm_label,
                    ablation_seconds=estimate['seconds'], budget4_seconds=control['seconds'],
                    time_over_budget4=ratio, session_id=estimate['session_id'], attempt_id=estimate['attempt_id'],
                    input_hashes_json=json.dumps(estimate['input_hashes'], sort_keys=True),
                    source=str(root / 'campaign-summary.json')))
assert len(nominal_records) == 456
for env, *_ in ENV:
    for lang in ['rust', 'cpp']:
        for arm, _ in NOMINAL_ARMS:
            observed = [r for r in nominal_records if r['environment']==env and r['language']==lang and r['arm']==arm]
            assert len(observed) == (31 if arm=='ties_off' else 13)
with (BASE / 'F5_nominal_policy_data.csv').open('w', encoding='utf-8', newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(nominal_records[0])); writer.writeheader(); writer.writerows(nominal_records)
assert len(records) == 31 * 2 * 4 * 5 == 1240
for lang in ['rust', 'cpp']:
    assert all(case_ids_by_env[f'{env}:{lang}'] == case_ids_by_env[f'L311:{lang}'] for env, *_ in ENV)
with (BASE / 'F5_budget_data.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
medians = []
for env, label, run, ls in ENV:
    for lang in ['rust', 'cpp']:
        for x, (arm, _) in enumerate(ARMS):
            values = [r['time_over_budget4'] for r in records if r['environment'] == env and r['language'] == lang and r['arm'] == arm]
            assert len(values) == 31
            medians.append(dict(environment=env, language=lang, arm=arm, ordinal_x=x, cases=31,
                                within_environment_case_median=statistics.median(values)))
with (BASE / 'F5_budget_environment_medians.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(medians[0])); writer.writeheader(); writer.writerows(medians)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.2,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','axes.labelsize':9.8,
                     'savefig.facecolor':'white'})
fig, axes = plt.subplots(2,2,figsize=(12.5,10.7),gridspec_kw={'height_ratios':[2.3,1.35]})
fig.subplots_adjust(left=.14,right=.985,bottom=.17,top=.82,wspace=.18,hspace=.70)
budget_connectors=[]
# Categorical x offsets keep the four environment observations recoverable.
ENV_DX=[-.06,-.02,.02,.06]
ENV_NOMINAL_DY=[.18,.06,-.06,-.18]
for col,(lang,native) in enumerate([('rust','jvrust'),('cpp','jvcpp')]):
    ax=axes[0,col]; dotax=axes[1,col]; style=STYLE[native]
    ax.axhspan(.97,1.03,color='#d9d9d9',alpha=.65,lw=0,zorder=0)
    ax.axhline(1,color='#8a8a8a',lw=.8,zorder=0)
    ax.axvline(.5,color='#bcbcbc',lw=.8,ls=':')
    for env_index,(env,label,run,ls) in enumerate(ENV):
        dx=ENV_DX[env_index]
        for case_id in case_ids_by_env[f'{env}:{lang}']:
            case_rows=[r for r in records if r['environment']==env and r['language']==lang and r['case_id']==case_id]
            bypass=next(r for r in case_rows if r['arm']=='no_reductions')
            # No plot call contains both bypass x=0 and any budget x>=1.
            ax.plot([dx],[bypass['time_over_budget4']],linestyle='None',color=style['color'],
                    marker=style['marker'],markersize=3.0,alpha=.20,markeredgewidth=0,zorder=2)
            values=sorted([r for r in case_rows if r['arm']!='no_reductions'],key=lambda r:r['ordinal_x'])
            assert [r['ordinal_x'] for r in values]==[1,2,3,4]
            xs=[r['ordinal_x']+dx for r in values]
            budget_connectors.append(xs)
            ax.plot(xs,[r['time_over_budget4'] for r in values],color=style['color'],linestyle=ls,
                    lw=.64,alpha=.15,marker=style['marker'],markersize=2.3,markeredgewidth=0,zorder=1)
        env_medians=[r for r in medians if r['environment']==env and r['language']==lang]
        bypass=next(r for r in env_medians if r['arm']=='no_reductions')
        ax.plot([dx],[bypass['within_environment_case_median']],linestyle='None',color=style['color'],
                marker=style['marker'],markersize=6.0,markeredgecolor='white',markeredgewidth=.55,zorder=5)
        values=sorted([r for r in env_medians if r['arm']!='no_reductions'],key=lambda r:r['ordinal_x'])
        assert [r['ordinal_x'] for r in values]==[1,2,3,4]
        xs=[r['ordinal_x']+dx for r in values]
        budget_connectors.append(xs)
        ax.plot(xs,[r['within_environment_case_median'] for r in values],color=style['color'],
                linestyle=ls,lw=2.0,alpha=.98,marker=style['marker'],markersize=5.2,
                markeredgecolor='white',markeredgewidth=.45,zorder=4)
    ax.set_title(f"{style['label']} controlled policies",pad=12)
    ax.set_yscale('log');ax.set_ylim(.8,12)
    ax.yaxis.set_major_locator(FixedLocator([.8,1,1.5,2,4,8,12]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}x'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis='y',which='major',color='#dedede',lw=.55,zorder=0)
    ax.set_xlim(-.25,4.22);ax.set_xticks(range(5),[a[1] for a in ARMS])
    ax.set_xlabel('Bypass intervention | displacement budget',labelpad=8)
    ax.tick_params(axis='x',length=0,pad=7)
    if col==0:ax.set_ylabel('Time / same-case budget-4 time (log)')

    # Nominal interventions have dots only. Four tiny vertical lanes identify environments.
    dotax.axvspan(.97,1.03,color='#d9d9d9',alpha=.65,lw=0,zorder=0)
    dotax.axvline(1,color='#8a8a8a',lw=.8,zorder=0)
    for nominal_index,(arm,arm_label) in enumerate(NOMINAL_ARMS):
        center=2-nominal_index
        dotax.axhline(center,color='#eeeeee',lw=.6,zorder=0)
        for env_index,(env,label,run,ls) in enumerate(ENV):
            rows=sorted([r for r in nominal_records if r['environment']==env and r['language']==lang and r['arm']==arm],key=lambda r:r['case_id'])
            for case_index,r in enumerate(rows):
                case_offset=(case_index-(len(rows)-1)/2)*(.055/max(len(rows)-1,1))
                y=center+ENV_NOMINAL_DY[env_index]+case_offset
                value=r['time_over_budget4']
                dotax.vlines(value,y-.018,y+.018,color=style['color'],lw=.55,alpha=.45,zorder=2)
                dotax.plot([value],[y],linestyle='None',color=style['color'],marker=style['marker'],
                           markersize=2.6,markeredgewidth=0,alpha=.48,zorder=3)
    dotax.set_title('Nominal interventions (unconnected dots)',fontsize=10,pad=10)
    dotax.set_xscale('log');dotax.set_xlim(.8,4.3)
    dotax.set_xticks([.8,1,1.5,2,3,4])
    dotax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}x'))
    dotax.xaxis.set_minor_formatter(NullFormatter())
    dotax.grid(axis='x',which='major',color='#dedede',lw=.55,zorder=0)
    dotax.set_yticks([2,1,0],[a[1] for a in NOMINAL_ARMS]);dotax.set_ylim(-.4,2.4)
    dotax.tick_params(axis='y',length=0,pad=6)
    if col == 1:
        dotax.tick_params(axis='y',labelleft=False)
    dotax.set_xlabel('Time / same-language controlled reference (log)',labelpad=8)
assert len(budget_connectors)==248+8
assert min(min(xs) for xs in budget_connectors)>.9

fig.text(.14,.97,'Thin budget curves: every case. Thick curves/large bypass dots: case median within each environment separately.',fontsize=9.5)
handles=[Line2D([0],[0],color='#404040',ls=ls,lw=1.6,label=env) for env,label,_,ls in ENV]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.92),ncol=4,frameon=False,columnspacing=2.3)
fig.text(.115,.080,'Bypass and nominal switches are unconnected. Dot lanes: L311, L314, W311, W314.',fontsize=8.5)
fig.text(.115,.050,'1,696 points; medians remain within environment. Gray bands: ±3%.',fontsize=8.7)
save(fig, 'F5_policy')
plt.close(fig)
audit={'sources':'Only saved four light campaign summaries and manifests; no solver import or calls.',
       'bypass_budget_points':len(records),'nominal_policy_points':len(nominal_records),
       'all_plotted_source_points':len(records)+len(nominal_records),
       'eligible_cases_per_language_per_environment':31,'within_environment_case_medians':len(medians),
       'no_cross_environment_pooling':True,'same_session_attempt_input_hashes_for_each_policy_control_pair':True,
       'ordinal_categories':['no reductions',1,2,4,8],'connected_budgets_only':[1,2,4,8],
       'bypass_to_budget_connector_count':0,'budget_only_case_traces':248,'budget_only_median_traces':8,
       'nominal_interventions_unconnected':True,'no_reductions_is_bypass_not_budget_zero':True,
       'nominal_counts_per_environment_language':{'ties_off':31,'cutoff_off':13,'ties_cutoff_off':13},
       'min_time_over_budget4':min(r['time_over_budget4'] for r in records+nominal_records),
       'max_time_over_budget4':max(r['time_over_budget4'] for r in records+nominal_records),'case_ids':case_ids_by_env}
(BASE/'F5_budget_assertions.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Generated F5_policy')
