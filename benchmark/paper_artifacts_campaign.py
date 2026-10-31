"""Publication panels and complete indexes from the four frozen light sessions.

No benchmark modules, matrix constructors, native extensions or solver calls are
imported. Diagnostic records and validation outcomes are read, never regenerated.
"""
import collections
import hashlib
import json
import math
import runpy
import statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, FixedLocator, NullLocator
from paper_artifacts_common import *

CANONICAL = ['jvrust','jvcpp','jvhybrid','jvx','jvs','scipy','scipy:capped',
             'scipy:dummies','scipy:diagonal_dummies','scipy:square','sparse:csgraph','sparse:lapmod']

def ratio_axis(ax, direction='x'):
    getattr(ax, 'set_'+direction+'scale')('log')
    if direction=='x':
        ax.axvspan(.97,1.03,color='.88',zorder=0);ax.axvline(1,color='.45',lw=.7)
    else:
        ax.axhspan(.97,1.03,color='.88',zorder=0);ax.axhline(1,color='.45',lw=.7)
    ax.grid(axis=direction,color='.9',lw=.5)

def context(d,c):
    return dict(environment=d['env'],run=d['run'],case_id=c['id'],family=c['family'],kind=c['kind'],
                objective=c['objective'],rows=c['rows'],cols=c['cols'],limit=c['limit'],gate=c['gate'],
                source=str((d['root']/'campaign-summary.json').relative_to(REPO)),
                manifest=str((d['root']/'campaign-session/manifest.json').relative_to(REPO)))

def export_primary(runs):
    records=[]; excluded=[]; build_index=[]
    for d in runs:
        pairs={(x['case_id'],x['method']):x for x in d['summary']['paired']}
        for e in d['summary']['estimates']:
            c=d['cases'][e['case_id']];p=pairs.get((e['case_id'],e['method']),{})
            row={**context(d,c),**e,'input_group':c['input_group'],'case_metadata':c['metadata'],
                 'reference':p.get('reference'), 'reference_seconds':p.get('reference_seconds'),
                 'ratio':p.get('ratio'),'reference_attempt_id':p.get('reference_attempt_id'),
                 'pairing':p.get('pairing'),'classification':classification(p['ratio']) if p.get('ratio') else ''}
            records.append(row)
        for c in d['cases'].values():
            if c['kind']!='single':continue
            desired=['jvrust','jvcpp','jvhybrid','jvx','jvs','scipy'] if c['objective']=='required' else ['jvrust','jvcpp','jvx','scipy:capped','scipy:dummies','scipy:diagonal_dummies','scipy:square','sparse:csgraph','sparse:lapmod']
            for method in desired:
                if method not in c['methods']:
                    reason='light geometry baseline row cutoff' if c['family']=='geometry' and c['rows']>1024 and method in d['manifest']['light_coverage']['capped_geometry_methods'] else 'saved protocol representation admission / not planned'
                    excluded.append({**context(d,c),'method':method,'status':'not_planned','reason':reason})
        for name in ['environment.json','configuration.json','build-index.json','manifest.json']:
            p=d['root']/'campaign-session'/name
            build_index.append(dict(environment=d['env'],path=str(p.relative_to(REPO)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    writecsv('S2_primary_conditions.csv',records)
    writecsv('S2_planned_exclusions.csv',excluded)
    writecsv('S2_build_source_index.csv',build_index)
    writecsv('S3_policy_timing.csv',[r for r in records if r['method'].startswith('policy:')])
    writecsv('S3_layout_timing.csv',[r for r in records if r['method'].startswith(('cost:','shape:'))])
    writecsv('S5_batch_timing.csv',[{**r,'milliseconds_per_batch':r['seconds']*1000} for r in records if r['kind']=='batch'])
    return records

def representations(runs):
    rows=[];full=[];panel_ids=[]
    shapes=['smallest_fill_zero','smallest_fill_inf','sum_fill_inf','sum_row_inf','sum_col_inf','sum_split_inf']
    costs=['fill_row_pad_zero','fill_row_pad_inf','fill_col_pad_zero','fill_col_pad_inf','fill_split',
           'diag_row_pad_zero','diag_row_pad_inf','diag_col_pad_0','diag_col_pad_inf','diag_split','diag_split_sparse']
    canonical=['jvrust','jvcpp','jvx','scipy:capped','scipy:dummies','scipy:diagonal_dummies','scipy:square']
    groups=[('Required square controls',['jvrust','jvcpp','jvhybrid','jvx','jvs','scipy']),
            ('Required shape layouts',shapes),('Optional formulations',canonical),('Optional cost layouts',costs)]
    keys=[(title,key) for title,keys_ in groups for key in keys_]
    positions={};headings={};y=0
    for title,keys_ in groups:
        headings[title]=y;y+=1
        for key in keys_:positions[title,key]=y;y+=1
    for d in runs:
        panel=[c for c in d['cases'].values() if any(m.startswith(('cost:','shape:')) for m in c['methods'])]
        controls=[c for c in d['cases'].values() if c['kind']=='single' and c['objective']=='required' and c['rows']==c['cols'] and any(m.startswith('policy:') for m in c['methods'])]
        assert len(panel)==23 and len(controls)==2
        panel_ids.append([c['id'] for c in panel+controls])
        pairs={(p['case_id'],p['method']):p for p in d['summary']['paired']}
        layout_count=0
        for c in panel+controls:
            for method in c['methods']:
                if method.startswith(('cost:','shape:')):
                    block='Required shape layouts' if method.startswith('shape:') else 'Optional cost layouts'
                    key=method.split(':',2)[2];layout_count+=1
                elif c in controls and method in CANONICAL:
                    block='Required square controls';key=method
                elif c['objective']=='optional' and method in canonical:
                    block='Optional formulations';key=method
                else:continue
                e=d['single'][c['id'],method];ref=c['metadata']['primary_comparator'];re=d['single'][c['id'],ref]
                assert e['attempt_id']==re['attempt_id'] and e['input_hashes']==re['input_hashes']
                ratio=re['seconds']/e['seconds']
                if method!=ref:assert math.isclose(pairs[c['id'],method]['ratio'],ratio,rel_tol=1e-12)
                rows.append({**context(d,c),'block':block,'row':key,'method':method,'ratio':ratio,'seconds':e['seconds'],
                             'reference':ref,'reference_seconds':re['seconds'],'attempt_id':e['attempt_id'],'input_hashes':e['input_hashes']})
            for category,layouts in [('shape',shapes)] if c['objective']=='required' else [('cost',costs)]:
                for backend in ['scipy','jvx','jvs']:
                    for layout in layouts:
                        method=f'{category}:{backend}:{layout}'
                        full.append({**context(d,c),'method':method,'status':'complete' if method in c['methods'] else 'not_planned','display':'value' if method in c['methods'] else '—'})
        assert layout_count==624
    assert all(p==panel_ids[0] for p in panel_ids)
    writecsv('F4_representations.csv',rows);writecsv('S3_layout_coverage.csv',full)
    fig,ax=plt.subplots(figsize=(11.8,10.2))
    ratio_axis(ax);ax.set_xlim(min(r['ratio'] for r in rows)*.8,max(r['ratio'] for r in rows)*1.3)
    for block,key in keys:
        i=positions[block,key]
        if i%2==0:ax.axhspan(i-.45,i+.45,color='.97',zorder=-1)
        for env_index,(env,*_) in enumerate(ENVS):
            rr=[r for r in rows if r['environment']==env and r['block']==block and r['row']==key]
            for method in sorted({r['method'] for r in rr}):
                sr=sorted([r for r in rr if r['method']==method],key=lambda r:r['case_id']);st=style(method)
                backend_offset=({'scipy':-.10,'jvx':0,'jvs':.10}.get(method.split(':')[1],0) if ':' in method and method.startswith(('cost:','shape:')) else 0)
                yy=[i+(env_index-1.5)*.12+backend_offset+((j*7)%5-2)*.012 for j in range(len(sr))]
                ax.scatter([r['ratio'] for r in sr],yy,color=st['color'],marker=st['marker'],s=13,alpha=.65,edgecolors='white',linewidths=.2)
    labels=[(STYLE[k]['label'] if k in STYLE else k.replace('_',' ')) for b,k in keys]
    ax.set_yticks([positions[k] for k in keys],labels,fontsize=8);ax.set_ylim(y-.4,-.6)
    for title,start in headings.items():
        ax.axhline(start-.45,color='.55',lw=.6)
        ax.text(.02,start,title,transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8,weight='bold')
    legends(fig,['jvrust','jvcpp','jvhybrid','jvx','jvs','scipy','scipy:dummies','scipy:diagonal_dummies','scipy:square'],top=.99,environment_top=.90)
    ax.set_xlabel('Matching SciPy / representation time (×; logarithmic)')
    fig.text(.27,.035,'Within-row lanes: L311, L314, W311, W314. Gray band: ±3%.',fontsize=8)
    fig.subplots_adjust(left=.27,right=.98,top=.83,bottom=.11)
    save(fig,'F4_representations');plt.close(fig)
    return dict(cases_per_environment=25,layout_conditions_per_environment=624,plotted_points=len(rows))

def batch(runs):
    rows=[]
    for d in runs:
        assert len(d['summary']['batch_contrasts'])==267
        for r in d['summary']['batch_contrasts']:
            assert r['status']=='complete'
            c=d['cases'][r['case_id']]
            inverse=r['contrast'] in ('solver_state_lifetime','orientation_buffer_lifetime','total_worker_lifetime')
            ratio=1/r['ratio'] if inverse else r['ratio']
            method=r['method'];native=('jvrust' if ':rust:' in method else 'jvcpp') if method.startswith('workspace:') else method.split(':')[1]
            label={'batch_overhead':'Loop / batch','worker_count':'Batch 1 / 4 workers',
                   'workspace_worker_count':'Workspace 1 / 4','solver_state_lifetime':'Scratch W1 / W0',
                   'orientation_buffer_lifetime':'Orientation W2 / W1','total_worker_lifetime':'Worker W2 / W0'}[r['contrast']]
            arm=method.split(':')[2] if method.startswith('workspace:') else ''
            row_label=f'{label} · {STYLE[native]["label"]}'
            if r['contrast']=='workspace_worker_count':row_label+=f' {arm.upper()}'
            elif r['contrast'] in ('solver_state_lifetime','orientation_buffer_lifetime'):row_label+=f' {r["workers"]}w'
            rows.append({**context(d,c),**r,'native_identity':native,'row_label':row_label,'display_ratio':ratio,
                         'display_definition':'method_seconds / reference_seconds' if inverse else r['ratio_definition'],
                         'display_classification':classification(ratio),'plotted':r['contrast']!='total_worker_lifetime'})
    writecsv('S5_batch_contrasts.csv',rows)
    printed=[r for r in rows if r['plotted']];writecsv('F8_batch.csv',printed)
    order=['batch_overhead','worker_count','workspace_worker_count','solver_state_lifetime','orientation_buffer_lifetime']
    labels=[]
    for con in order:
        labels.extend(sorted({r['row_label'] for r in printed if r['contrast']==con}))
    fig,axes=plt.subplots(1,4,figsize=(14.4,10.2),sharex=True,sharey=True)
    low=min(r['display_ratio'] for r in printed)*.85;high=max(r['display_ratio'] for r in printed)*1.18
    for ax,d in zip(axes,runs):
        ratio_axis(ax);ax.set_xlim(low,high)
        for i,label in enumerate(labels):
            rr=sorted([r for r in printed if r['environment']==d['env'] and r['row_label']==label],key=lambda r:r['case_id'])
            if i%2==0:ax.axhspan(i-.43,i+.43,color='.96',zorder=-1)
            for j,r in enumerate(rr):
                st=STYLE[r['native_identity']];ax.scatter(r['display_ratio'],i+(j-(len(rr)-1)/2)*.035,s=18,marker=st['marker'],color=st['color'],edgecolors='white',lw=.3,zorder=3)
        ax.set_title(d['env'],fontsize=14);ax.set_ylim(len(labels)-.5,-.5)
        ax.xaxis.set_major_locator(FixedLocator([.5,1,2,4,8]));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}×'))
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].set_yticks(range(len(labels)),[f'{lab}  ({sum(r["row_label"]==lab and r["environment"]=="L311" for r in printed)})' for lab in labels],fontsize=7.5)
    fig.supxlabel('Whole-batch latency ratio (×; >1 favors the treatment)',y=.05,fontsize=10)
    fig.text(.30,.015,'Parentheses: case count. w: workers. Gray band: ±3%.',fontsize=8)
    fig.subplots_adjust(left=.37,right=.99,top=.94,bottom=.11,wspace=.12)
    save(fig,'F8_batch');plt.close(fig)
    return rows

def orientation(runs):
    records=[];pairs=[];source_hashes=[]
    for d in runs:
        cs=[c for c in d['cases'].values() if c['kind']=='single' and c['family'] in {'large_rectangle','extreme_rectangle'}]
        groups=collections.defaultdict(list)
        for c in cs:groups[(json.dumps(c['input_group'],sort_keys=True),c['objective'],c['limit'],c['gate'])].append(c)
        assert len(cs)==24 and len(groups)==12
        for key,group in groups.items():
            assert len(group)==2
            w=next(c for c in group if not c['transpose']);t=next(c for c in group if c['transpose'])
            assert w['rows']==t['cols'] and w['cols']==t['rows'] and w['methods']==t['methods']
            pairs.append({**context(d,w),'tall_case_id':t['id'],'input_group':w['input_group']})
            for method in w['methods']:
                a=d['single'][w['id'],method];b=d['single'][t['id'],method]
                assert a['session_id']==b['session_id'] and a['environment_id']==b['environment_id']
                assert set(a['input_hashes'])==set(b['input_hashes'])==set('01234')
                records.append({**context(d,w),'method':method,'shape':f'{w["rows"]}x{w["cols"]}',
                                'tall_case':t['id'],'wide_ms':a['seconds']*1000,'tall_ms':b['seconds']*1000,
                                'tall_over_wide':b['seconds']/a['seconds'],'wide_attempt':a['attempt_id'],'tall_attempt':b['attempt_id'],
                                'session_id':a['session_id'],'environment_id':a['environment_id'],'wide_input_hashes':a['input_hashes'],'tall_input_hashes':b['input_hashes'],
                                'input_group':w['input_group'],'pairing':'different_case_attempts_shared_input_group'})
        src=d['root']/'campaign-session/sources/benchmark/campaign/datasets.py'
        source_hashes.append(hashlib.sha256(src.read_bytes()).hexdigest())
    assert len(set(source_hashes))==1 and len(records)==296
    writecsv('F12_orientation.csv',records);writecsv('F12_orientation_pairs.csv',pairs)
    return records

def rejection(runs):
    records=[];outcomes=[]
    for d in runs:
        cases=[c for c in d['cases'].values() if c['family']=='rejection'];assert len(cases)==22
        distribution_hashes={}
        for c in cases:
            dist=c['input_group']['parameters']['distribution']
            e=d['single'][c['id'],'jvrust']
            f=d['root']/'campaign-session/attempts'/e['attempt_id']/'measurements.jsonl'
            raw=[json.loads(line) for line in f.read_text().splitlines()]
            vv=[r for r in raw if r.get('record_type')=='validation' and r.get('case_id')==c['id']]
            values=collections.defaultdict(set)
            for r in vv:values[r['input_index']].add(r['result']['cardinality'])
            assert len(values)==5 and all(len(v)==1 for v in values.values())
            vals=[r for r in vv if r['method']=='jvrust'];assert len(vals)==5
            current={str(r['input_index']):r['input_sha256'] for r in vals};assert current==e['input_hashes']
            if dist in distribution_hashes:assert distribution_hashes[dist]==current
            else:distribution_hashes[dist]=current
            rejected=[100*(1-r['result']['cardinality']/min(c['rows'],c['cols'])) for r in vals]
            for r in vals:
                outcomes.append({**context(d,c),'distribution':dist,'input_index':r['input_index'],'input_sha256':r['input_sha256'],
                                 **r['result'],'validation_source':str(f.relative_to(REPO))})
            for method in ['jvrust','jvcpp','scipy:capped']:
                e=d['single'][c['id'],method]
                records.append({**context(d,c),'distribution':dist,'method':method,'microseconds':e['seconds']*1e6,
                                'mean_rejected_rows_percent':statistics.mean(rejected),'min_rejected_rows_percent':min(rejected),
                                'max_rejected_rows_percent':max(rejected),'cardinalities':[r['result']['cardinality'] for r in vals],
                                'attempt_id':e['attempt_id'],'input_hashes':e['input_hashes']})
    assert len(records)==264
    writecsv('F11_rejection.csv',records);writecsv('S2_rejection_validation.csv',outcomes)
    fig,axes=plt.subplots(2,2,figsize=(12.2,8),sharex='col',sharey='row',gridspec_kw={'height_ratios':[2,1]})
    for j,dist in enumerate(['uniform','blocks']):
        for env,label,_,ls in ENVS:
            for method in ['jvrust','jvcpp','scipy:capped']:
                rr=sorted([r for r in records if r['environment']==env and r['distribution']==dist and r['method']==method],key=lambda r:r['limit'])
                assert len(rr)==11
                st=STYLE[method]
                axes[0,j].plot([r['limit'] for r in rr],[r['microseconds'] for r in rr],color=st['color'],marker=st['marker'],ls=ls,lw=.9,ms=3)
                if method=='jvrust':axes[1,j].plot([r['limit'] for r in rr],[r['mean_rejected_rows_percent'] for r in rr],color='.35',ls=ls,lw=1,marker='o',ms=2)
        axes[0,j].set_title(dist.title());axes[0,j].set_yscale('log')
        for ax in axes[:,j]:
            ax.set_xscale('symlog',linthresh=.0005,linscale=.6);ax.set_xlim(0,4.5);ax.grid(color='.90',lw=.5)
            ax.set_xticks([0,.0005,.001,.002,.005,.01,.02,.05,.1,.5,4])
            ax.set_xticklabels(['0','.0005','.001','.002','.005','.01','.02','.05','.1','.5','4'],rotation=45,ha='right',fontsize=7)
            if dist=='blocks':ax.axvspan(.05,.5,color='#eee5d4',alpha=.6,zorder=-1)
        axes[1,j].set_xlabel('Limit L (linear 0–0.0005, logarithmic above)')
    axes[0,0].set_ylabel('Complete call (µs; logarithmic)');axes[1,0].set_ylabel('Unmatched smaller-side rows (%)')
    axes[0,1].annotate('K = 144 throughout shaded interval\nJVRUST latency rises >10× in each environment',xy=(.15,650),xytext=(.03,.94),textcoords='axes fraction',va='top',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.7),bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
    axes[1,1].text(.05,49,'43.75% rejected rows',fontsize=8)
    legends(fig,['jvrust','jvcpp','scipy:capped'],top=.99,environment_top=.945)
    fig.subplots_adjust(left=.09,right=.985,bottom=.14,top=.855,hspace=.15,wspace=.12)
    save(fig,'F11_rejection');plt.close(fig)
    return records

def diagnostic_exports(runs):
    counters=[];layouts=[];workspace=[];builds={};vectors={};fields=['initialization_row_scans','search_row_scans','initialization_displacements','initialization_chain_starts','augmentations','rejection_endings']
    for d in runs:
        diag=load(d['root']/'campaign-diagnostics/summary.json')
        local={}
        for job in diag['jobs']:
            assert job['status']=='complete'
            base=job['directory'].replace('\\','/').rsplit('/',1)[-1]
            path=d['root']/'campaign-diagnostics/attempts'/base/'diagnostics.jsonl'
            rr=[json.loads(line) for line in path.read_text().splitlines()]
            for r in rr:
                if r['study']=='native_counters' and r['method'].startswith('policy:'):
                    assert r['status']=='complete' and r['counted_calls']==1 and r['counters']['problems']==1
                    c=d['cases'][r['case_id']];e=d['single'][r['case_id'],r['method']]
                    assert e['input_hashes'][str(r['input_index'])]==r['input_sha256']
                    key=(r['case_id'],r['method'],r['input_index'],r['input_sha256'])
                    local[key]=tuple(r['counters'][f] for f in fields)
                    build=r['build'];build_key=d['env']+':'+r['method'];builds[build_key]={**build,'source_file':path.relative_to(REPO).as_posix(),'source_field':'build'}
                    arm=r['method'].split(':')[2]
                    expected=dict(budget=int(arm[-1]) if arm.startswith('budget') else 4,
                                  free_ties=arm not in ('ties_off','ties_cutoff_off'),
                                  reduction_passes=0 if arm=='no_reductions' else 2,
                                  required_cutoff=arm not in ('cutoff_off','ties_cutoff_off'),workspace=0)
                    assert build['policy']==expected
                    compact={k:v for k,v in r.items() if k!='build'}
                    counters.append({**context(d,c),'diagnostic_source':str(path.relative_to(REPO)),
                                     'definition_sha256':c['input_group']['definition_sha256'],
                                     'build_key':build_key,'artifact_sha256':build['artifact_sha256'],'policy':build['policy'],**compact})
                elif r['study']=='native_counters':
                    build=r['build'];build_key=d['env']+':'+r['method'];builds[build_key]={**build,'source_file':path.relative_to(REPO).as_posix(),'source_field':'build'}
                    workspace.append(dict(environment=d['env'],run=d['run'],diagnostic_source=str(path.relative_to(REPO)),
                                          build_key=build_key,**{k:v for k,v in r.items() if k!='build'}))
                elif r['study']=='layout_construction':
                    layouts.append(dict(environment=d['env'],run=d['run'],diagnostic_source=str(path.relative_to(REPO)),**r))
        assert len(local)==2140
        for (case,method,index,sha),vector in local.items():
            other=method.replace(':rust:',':cpp:') if ':rust:' in method else method.replace(':cpp:',':rust:')
            assert local[case,other,index,sha]==vector
        vectors[d['env']]=local
    assert all(v==vectors['L311'] for v in vectors.values())
    writecsv('S3_native_counters.csv',counters);writecsv('S3_layout_construction.csv',layouts)
    writecsv('S3_workspace_counters.csv',workspace,compress=True)
    (DATA/'S3_diagnostic_builds.json').write_text(json.dumps(builds,indent=2)+'\n',encoding='utf-8')
    anchors=[('base:optional256x320','no_reductions','Optional uniform; bypass'),('base:near256x320','budget1','Near square; budget 1'),
             ('base:ties128x160','ties_off','Tied base; free ties off'),('geometry:256x320:q0.9:motion0.25:required','cutoff_off','Geometry required; cutoff off'),
             ('formulations:optional_blocks256x320','budget1','Optional blocks; budget 1')]
    table=[]
    for case,arm,label in anchors:
        row={'case_id':case,'arm':arm,'label':label}
        for a in ['reference',arm]:
            rs=[r for r in counters if r['environment']=='L311' and r['case_id']==case and r['method']=='policy:rust:'+a]
            assert len(rs)==5
            for f in fields:row[a+'_'+f]=statistics.mean(r['counters'][f] for r in rs)
        table.append(row)
    writecsv('T3_counter_anchors.csv',table)
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Case; intervention & Init. scans & Search scans & Augmentations & Rejections \\',r'\midrule']
    for r in table:
        vals=[f'{r["reference_"+f]:g} $\\to$ {r[r["arm"]+"_"+f]:g}' for f in ['initialization_row_scans','search_row_scans','augmentations','rejection_endings']]
        lines.append(r['label']+' & '+' & '.join(vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    (TABLES/'T3_counters.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return dict(event_vectors_per_environment=2140,language_pairs=1070,all_six_events_match_across_environments_and_languages=True,anchors=table)

def scaling(runs):
    groups=collections.defaultdict(list)
    for c in runs[0]['cases'].values():
        if c['family']=='scaling':
            name=c['id'].split(':')[1].rsplit('_',1)[0];groups[name].append(c['id'])
        elif c['family'] in ['large_square','large_rectangle','extreme_rectangle']:
            name=c['family']+'_'+c['objective']+('_tall' if c['rows']>c['cols'] else '_wide' if c['rows']<c['cols'] else '')
            groups[name].append(c['id'])
    assert sum(map(len,groups.values()))==79
    exports=[];names=[];tex=[]
    # Each page has up to four workload facets, with linked runtime and ratio rows.
    for page,start in enumerate(range(0,len(groups),4),1):
        subset=list(groups.items())[start:start+4]
        fig,axes=plt.subplots(2,len(subset),figsize=(15.8,8.5),squeeze=False,sharey='row')
        for col,(group,ids) in enumerate(subset):
            for d in runs:
                cs=sorted([d['cases'][cid] for cid in ids],key=lambda c:max(c['rows'],c['cols']))
                assert len({c['objective'] for c in cs})==1
                for method in CANONICAL:
                    eligible=[c for c in cs if method in c['methods']]
                    if not eligible:continue
                    xx=[];tt=[];rr=[]
                    for c in eligible:
                        e=d['single'][c['id'],method];ref=c['metadata']['primary_comparator'];re=d['single'][c['id'],ref]
                        x=c['cols'] if 'fixed' in group or ('extreme_rectangle' in group and c['rows']<c['cols']) else c['rows']
                        xx.append(x);tt.append(1000*e['seconds']);rr.append(re['seconds']/e['seconds'])
                        exports.append({**context(d,c),'series':group,'x':x,'method':method,'milliseconds':tt[-1],'ratio':rr[-1],
                                        'attempt_id':e['attempt_id'],'reference':ref,'input_hashes':e['input_hashes']})
                    st=style(method)
                    for ax,yy in [(axes[0,col],tt),(axes[1,col],rr)]:ax.plot(xx,yy,color=st['color'],marker=st['marker'],ls=d['ls'],lw=.8,ms=3,alpha=.75)
            title=group.replace('large_rectangle_','Large rectangle\n').replace('extreme_rectangle_','Extreme rectangle\n').replace('large_square_','Large square\n').replace('_',' ')
            axes[0,col].set_title(title.capitalize(),fontsize=9)
            for ax in axes[:,col]:
                ax.set_xscale('log');ax.grid(color='.9',lw=.5)
                cs=[runs[0]['cases'][cid] for cid in ids]
                xs=sorted({c['cols'] if 'fixed' in group or ('extreme_rectangle' in group and c['rows']<c['cols']) else c['rows'] for c in cs})
                ax.xaxis.set_major_locator(FixedLocator(xs));ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
                ax.tick_params(axis='x',labelsize=7,rotation=30)
            axes[0,col].set_yscale('log');ratio_axis(axes[1,col],'y')
            axes[1,col].set_xlabel('Columns' if 'fixed' in group or ('extreme_rectangle' in group and 'wide' in group) else 'Rows')
        axes[0,0].set_ylabel('Complete call (ms)');axes[1,0].set_ylabel('SciPy / method (×)')
        legends(fig,CANONICAL,top=.995,environment_top=.855)
        fig.text(.10,.025,'One curve per method/environment. Protocol exclusions stop the affected curves.',fontsize=8)
        fig.subplots_adjust(left=.09,right=.975,top=.74,bottom=.15,hspace=.24,wspace=.23)
        name=f'A1_scaling_{page}';save(fig,name);plt.close(fig);names.append(name)
        descriptions='; '.join(g.replace('_',' ') for g,_ in subset)
        tex += [r'\begin{figure}[!htbp]',r'\centering',r'\includegraphics[width=\textwidth]{images/current/'+name+'.pdf}',
                r'\caption{Connected non-geometric scaling atlas: '+descriptions+r'. Top: complete-call milliseconds; bottom: matching SciPy time divided by method time. Each curve is one environment, method, objective and generator; the horizontal dimension is rows, except fixed-row and wide extreme cases, which vary columns. Optional small scaling uses $L=0.5$; large and extreme cases use $L=2/\max(n,m)$. All planned methods remain; protocol exclusions stop the corresponding curves. The gray band is $[0.97,1.03]$.}',
                r'\label{fig:atlas'+str(page)+'}',r'\end{figure}']
    writecsv('A1_scaling.csv',exports)
    (TABLES/'A1_scaling_figures.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    return names

def geometry_sensitivity(runs):
    rows=[];methods=['jvrust','jvcpp','jvx','scipy:capped','scipy:dummies','scipy:diagonal_dummies','scipy:square','sparse:csgraph','sparse:lapmod']
    settings=[(.9,.25),(.9,1),(.6,.25),(.6,1)]
    fig,axes=plt.subplots(3,4,figsize=(14,10.2),sharex=True,sharey=True)
    for d in runs:
        cs=[c for c in d['cases'].values() if c['family']=='geometry' and c['rows']==256 and c['cols']==320 and c['objective']=='optional']
        assert len(cs)==36
        for col,setting in enumerate(settings):
            for i,gate in enumerate([.75,1.5,None]):
                selected=sorted([c for c in cs if (c['metadata']['q'],c['metadata']['motion'])==setting and c['metadata']['gate_over_delta']==gate],key=lambda c:c['metadata']['limit_over_delta'])
                assert len(selected)==3
                for method in methods:
                    yy=[]
                    for c in selected:
                        e=d['single'][c['id'],method];ref=d['single'][c['id'],'scipy:capped'];ratio=ref['seconds']/e['seconds'];yy.append(ratio)
                        rows.append({**context(d,c),'method':method,'ratio':ratio,'seconds':e['seconds'],'metadata':c['metadata'],'attempt_id':e['attempt_id'],'input_hashes':e['input_hashes']})
                    st=style(method);axes[i,col].plot([.5,1.5,3],yy,color=st['color'],marker=st['marker'],ls=d['ls'],ms=3,lw=.8,alpha=.7)
    for col,setting in enumerate(settings):
        for i,gate in enumerate([.75,1.5,None]):
            ax=axes[i,col];ratio_axis(ax,'y');ax.set_title(f'q={setting[0]}, motion={setting[1]}δ\ngate={gate if gate else "none"}'+('δ' if gate else ''),fontsize=8)
            ax.set_xticks([.5,1.5,3]);ax.set_ylim(min(r['ratio'] for r in rows)*.8,max(r['ratio'] for r in rows)*1.2)
            if i==2:ax.set_xlabel('L / δ')
            if col==0:ax.set_ylabel('SciPy / method (×)')
            redundant=[x for x in [.5,1.5,3] if gate is not None and gate>=x]
            for x in redundant:ax.text(x,.015,'R',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color='.35')
    legends(fig,methods,top=.99,environment_top=.90)
    fig.text(.09,.025,'256×320 inputs. R: gate at or above L, redundant for optional admissibility.',fontsize=8)
    fig.subplots_adjust(left=.09,right=.99,top=.80,bottom=.10,hspace=.45,wspace=.22)
    save(fig,'A2_geometry_sensitivity');plt.close(fig);writecsv('A2_geometry_sensitivity.csv',rows)
    # The four extra q/motion settings are compared at fixed size, never connected as a size sweep.
    fig,axes=plt.subplots(1,2,figsize=(12,4.9),sharex=True,sharey=True);panel=[]
    for j,obj in enumerate(['required','optional']):
        ax=axes[j];ratio_axis(ax)
        for d_idx,d in enumerate(runs):
            cs=[c for c in d['cases'].values() if c['family']=='geometry' and c['rows']==1024 and c['cols']==1280 and c['objective']==obj]
            assert len(cs)==4
            for i,setting in enumerate(settings):
                c=next(c for c in cs if (c['metadata']['q'],c['metadata']['motion'])==setting)
                ms=[m for m in CANONICAL if m in c['methods']]
                for k,m in enumerate(ms):
                    e=d['single'][c['id'],m];re=d['single'][c['id'],c['metadata']['primary_comparator']];v=re['seconds']/e['seconds'];st=style(m)
                    ax.scatter(v,i+(k-(len(ms)-1)/2)*.055+(d_idx-1.5)*.01,s=16,color=st['color'],marker=st['marker'],edgecolors='white',lw=.2)
                    panel.append({**context(d,c),'method':m,'ratio':v,'metadata':c['metadata']})
        ax.set_title('1024×1280 '+obj);ax.set_yticks(range(4),[f'q={q}, motion={mo}δ' for q,mo in settings]);ax.set_xlabel('Matching SciPy / method (×)')
    legends(fig,CANONICAL,top=.995,environment_top=.76)
    fig.subplots_adjust(left=.22,right=.99,top=.61,bottom=.16,wspace=.16)
    save(fig,'A2_geometry_panel');plt.close(fig);writecsv('A2_geometry_panel.csv',panel)

def summaries(runs,primary,batch_rows,rejection_rows,counters):
    summary={'source_contract':'Four saved light sessions only; each environment summarized separately; ratios classified with [0.97,1.03] tie band.',
             'environments':{},'diagnostics':counters,'sources':{d['env']:str(d['root'].relative_to(REPO)) for d in runs}}
    for d in runs:
        env=d['env'];out={}
        pp=[r for r in primary if r['environment']==env and r['kind']=='single' and r['method'] in ['jvrust','jvcpp','jvhybrid']]
        out['native']={}
        for obj in ['required','optional']:
            for method in ['jvrust','jvcpp','jvhybrid']:
                rr=[r for r in pp if r['objective']==obj and r['method']==method]
                if rr:out['native'][obj+':'+method]={'n':len(rr),'counts':dict(collections.Counter(r['classification'] for r in rr)),
                    'minimum':min(({'case_id':r['case_id'],'ratio':r['ratio']} for r in rr),key=lambda r:r['ratio']),
                    'maximum':max(({'case_id':r['case_id'],'ratio':r['ratio']} for r in rr),key=lambda r:r['ratio'])}
        out['sparse']={}
        for method in ['sparse:csgraph','sparse:lapmod']:
            rr=[r for r in primary if r['environment']==env and r['method']==method]
            derived=[dict(case_id=r['case_id'],scipy_ratio=r['ratio'],rust_ratio=d['single'][r['case_id'],'jvrust']['seconds']/r['seconds']) for r in rr]
            out['sparse'][method]={'n':len(rr),'dense_counts':dict(collections.Counter(classification(r['scipy_ratio']) for r in derived)),
                                  'rust_counts':dict(collections.Counter(classification(r['rust_ratio']) for r in derived)),
                                  'rust_wins':[r for r in derived if r['rust_ratio']>1.03]}
        out['batch']={}
        for contrast in sorted({r['contrast'] for r in batch_rows}):
            rr=[r for r in batch_rows if r['environment']==env and r['contrast']==contrast]
            out['batch'][contrast]={'n':len(rr),'counts':dict(collections.Counter(r['display_classification'] for r in rr)),
                'minimum':min(r['display_ratio'] for r in rr),'maximum':max(r['display_ratio'] for r in rr),
                'losses':[dict(case_id=r['case_id'],method=r['method'],ratio=r['display_ratio']) for r in rr if r['display_classification']=='loss']}
        out['policy']={}
        for language in ['rust','cpp']:
            for arm in ['no_reductions','budget1','budget2','budget8','ties_off','cutoff_off','ties_cutoff_off']:
                rr=[r for r in primary if r['environment']==env and r['method']==f'policy:{language}:{arm}']
                out['policy'][language+':'+arm]={'n':len(rr),'treatment_over_reference_range':[min(1/r['ratio'] for r in rr),max(1/r['ratio'] for r in rr)],
                                              'treatment_over_reference_median':statistics.median(1/r['ratio'] for r in rr)}
        plateau=[r for r in rejection_rows if r['environment']==env and r['distribution']=='blocks' and r['method']=='jvrust' and r['limit'] in [.05,.1,.5]]
        out['rejection_plateau']=plateau
        out['geometry_sparse_crossovers']={}
        for q,motion in [(.9,.25),(.6,1.)]:
            for method in ['sparse:csgraph','sparse:lapmod']:
                rr=[r for r in primary if r['environment']==env and r['family']=='geometry' and r['rows']==r['cols'] and r['method']==method and r['case_metadata']['q']==q and r['case_metadata']['motion']==motion]
                out['geometry_sparse_crossovers'][f'q{q}:motion{motion}:{method}']=[dict(rows=r['rows'],ratio=r['ratio'],case_id=r['case_id']) for r in sorted(rr,key=lambda r:r['rows'])]
        summary['environments'][env]=out
    (DATA/'campaign_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def generate_campaign(repo=REPO):
    assert Path(repo).resolve()==REPO
    import paper_artifacts_common as common
    base_rc={'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False}
    runs=read_runs();primary=export_primary(runs)
    # The original projections still export exact indexed observations. Their
    # old drawing calls are suppressed; only current question-specific figures
    # are published below.
    common.SUPPRESS_FIGURES=True
    try:
        for name in ['f2_overview.py','f3_geometry.py','f5_policy.py','f6_sparse.py','f7_structure.py']:
            with plt.rc_context(base_rc):
                runpy.run_path(str(REPO/'benchmark/paper_artifact_panels'/name),run_name='__main__')
            plt.close('all')
        with plt.rc_context(base_rc):rep=representations(runs)
        with plt.rc_context(base_rc):batch_rows=batch(runs)
        orientation(runs)
        with plt.rc_context(base_rc):
            runpy.run_path(str(REPO/'benchmark/paper_artifact_panels/f12_orientation.py'),run_name='__main__')
        with plt.rc_context(base_rc):rejection_rows=rejection(runs)
        counter_info=diagnostic_exports(runs)
        with plt.rc_context(base_rc):scaling(runs)
        with plt.rc_context(base_rc):geometry_sensitivity(runs)
    finally:
        common.SUPPRESS_FIGURES=False
        plt.close('all')
    result=summaries(runs,primary,batch_rows,rejection_rows,counter_info)
    result['representations']=rep
    from paper_artifacts_charts import generate_charts
    result['charts']=generate_charts(runs)
    (DATA/'campaign_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
