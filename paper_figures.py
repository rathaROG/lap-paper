"""Prepare publication figures from frozen measurements; never run solvers."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAPER = Path(__file__).resolve().parent
COMPARISON = PAPER / 'comparisons/python-3.11.17-vs-3.14.9-20261024'
REPLICATION = PAPER / 'collections/python-3.11.17-20261024'
EXTENSION = PAPER / 'extensions/python-3.14.9-20261024'
OUT = PAPER / 'figures/paper'

def rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    images = PAPER / 'latex/images'
    images.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for stem in ('single_runtime', 'batch_runtime', 'memory_peak', 'scaling_required', 'scaling_optional', 'scaling_blocks'):
        source = PAPER / 'figures' / (stem + '.pdf')
        shutil.copy2(source, images / source.name)
        manifest[source.name] = {'source': source.relative_to(PAPER).as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    # Reuse the measured chart definitions, but size their labels for the paper.
    # The frozen collection's figures and all measurement files stay untouched.
    import plot_extended_results as extension
    selected = {'large_square':3.4, 'large_rectangles':5.4, 'extreme_rectangles':5.4,
                'structure_required':5.2, 'structure_optional':5.2,
                'rejection_sweep':8.2, 'memory_sweeps':5.4}
    old_finish = extension.legend_and_save

    def publish_extended(fig, axes, destination, stem, title, note, ignored):
        if stem not in selected:
            plt.close(fig)
            return
        fig.set_size_inches(7.0, selected[stem])
        if stem == 'rejection_sweep':
            grid = fig.add_gridspec(4,2)
            for col in range(4):
                axes[0,col].set_subplotspec(grid[col,0])
                axes[1,col].set_subplotspec(grid[col,1])
        handles, names = [], []
        for axis_index, ax in enumerate(np.asarray(axes).ravel()):
            ax.title.set_fontsize(9)
            ax.xaxis.label.set_fontsize(8)
            ax.yaxis.label.set_fontsize(8)
            ax.tick_params(axis='both',labelsize=7.5)
            if stem.startswith('structure_'):
                ax.set_xlabel(('Competing row fraction', 'Forbidden probability', 'Quantization levels')[axis_index % 3])
            if stem=='memory_sweeps':
                ax.set_xlabel('Larger dimension (smaller = 256)')
                ax.set_ylabel('Additional peak (MiB)')
            if stem=='rejection_sweep' and axis_index >= 4:
                ax.set_ylabel('Unmatched (%)')
            h,l=ax.get_legend_handles_labels()
            for handle,name in zip(h,l):
                if name not in names: handles.append(handle); names.append(name)
        fig.legend(handles,names,ncol=3,loc='upper center',bbox_to_anchor=(.5,1),fontsize=8,frameon=False)
        top=.90 if selected[stem]>7 else .86
        fig.tight_layout(rect=(0,0,1,top),pad=.8,w_pad=1.2,h_pad=1.4)
        for suffix in ('png','svg','pdf'):
            fig.savefig(OUT/f'{stem}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
        source=OUT/f'{stem}.pdf'
        shutil.copy2(source,images/source.name)
        manifest[source.name]={'source':source.relative_to(PAPER).as_posix(),
            'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'measurement_collection':EXTENSION.relative_to(PAPER).as_posix(),
            'plot_definition':'plot_extended_results.py','note':note}

    extension.legend_and_save=publish_extended
    try:
        extension.new_timing_charts(EXTENSION, [])
        extension.rejection_chart(EXTENSION, [])
        extension.memory_chart(EXTENSION, [])
    finally:
        extension.legend_and_save=old_finish
    colors = ('#176ca4', '#d17c26')
    labels = ('Python 3.14.9', 'Python 3.11.17')
    summaries = {}
    for suite in ('scaling', 'batch'):
        rel = 'scaling/summary.csv' if suite=='scaling' else 'batch_summary.csv'
        summaries[suite] = [{(r['case'], r['solver']):r for r in rows(base / 'results' / rel)} for base in (PAPER, REPLICATION)]

    def finish(fig, stem):
        fig.tight_layout(pad=1.0, w_pad=2.0)
        for suffix in ('png','svg','pdf'):
            fig.savefig(OUT / f'{stem}.{suffix}', dpi=200, bbox_inches='tight')
        plt.close(fig)
        shutil.copy2(OUT / f'{stem}.pdf', images / f'{stem}.pdf')
        manifest[f'{stem}.pdf'] = {'source':f'figures/paper/{stem}.pdf', 'sha256':hashlib.sha256((OUT/f'{stem}.pdf').read_bytes()).hexdigest()}

    fig, axes = plt.subplots(1,2, figsize=(7.0,3.5))
    primary = rows(COMPARISON / 'primary_orderings.csv')
    palette = {'single':'#78519a','formulations':'#d17c26','scaling':'#32905d','extended':'#176ca4'}
    for suite, color in palette.items():
        points=[r for r in primary if r['suite']==suite and r['solver']=='jvrust']
        axes[0].scatter([float(r['reference_over_native_314']) for r in points], [float(r['reference_over_native_311']) for r in points],
                        label=f'{suite} ({len(points)})', color=color, s=13, alpha=.7)
    axes[0].set(xscale='log',yscale='log',xlim=(.05,200),ylim=(.05,200),
                xlabel='SciPy / JVRUST time, Python 3.14.9',ylabel='SciPy / JVRUST time, Python 3.11.17',title='201 / 201 orderings preserved')
    axes[0].plot([.05,200],[.05,200],color='#777777',ls=':',lw=.8)
    axes[0].axhline(1,color='#555555',lw=.6); axes[0].axvline(1,color='#555555',lw=.6)
    axes[0].legend(fontsize=6.5,frameon=False,loc='upper left')
    sizes=[16,32,64,128,256,512,1024]
    for table, color, label, style in zip(summaries['scaling'], colors, labels, ('-','--')):
        values=[float(table[f'optional_blocks_{n}x{5*n//4}','scipy_capped']['median_ns'])/float(table[f'optional_blocks_{n}x{5*n//4}','jvrust']['median_ns']) for n in sizes]
        axes[1].plot(sizes,values,color=color,label=label,ls=style,marker='o',ms=3)
    axes[1].set(xscale='log',yscale='log',xlabel='Smaller dimension N; M = 5N/4',ylabel='Capped SciPy / JVRUST time',title='Same sampled congestion crossover')
    axes[1].set_xticks(sizes,[str(n) for n in sizes],fontsize=6.5)
    axes[1].axhline(1,color='#555555',ls=':',lw=.8); axes[1].legend(fontsize=7,frameon=False)
    for ax in axes: ax.grid(alpha=.15); ax.set_axisbelow(True)
    finish(fig,'environment_consistency')

    fig, axes = plt.subplots(1,2,figsize=(7.0,3.4))
    for ax, suite, case, methods, divisor, ylabel, title in (
        (axes[0], 'scaling', 'required_square_32x32', ('jvc','scipy'), 1000, 'Microseconds per matrix', 'Required 32 x 32'),
        (axes[1], 'batch', 'batch64x64x80', ('jvrust_batch_t4','jvcpp_batch_t4'), 1e6, 'Milliseconds per whole batch', '64 problems, 64 x 80, four workers')):
        for index,(table,color,label) in enumerate(zip(summaries[suite],colors,labels)):
            med,lo,hi=[np.array([float(table[case,m][key])/divisor for m in methods]) for key in ('median_ns','q25_ns','q75_ns')]
            bars=ax.bar(np.arange(2)+(index-.5)*.34,med,width=.32,label=label,color=color)
            ax.errorbar(np.arange(2)+(index-.5)*.34,med,yerr=(med-lo,hi-med),fmt='none',color='#333333',capsize=2,lw=.8)
            ax.bar_label(bars,fmt='%.3f' if suite=='batch' else '%.2f',padding=4,fontsize=7)
        ax.set_xticks(range(2),('JVC','SciPy') if suite=='scaling' else ('JVRUST','JVCPP'))
        ax.set(ylabel=ylabel,title=title); ax.set_ylim(0,ax.get_ylim()[1]*1.23)
        ax.legend(fontsize=6.5,frameon=False,loc='upper left'); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    finish(fig,'ranking_exceptions')
    for record in manifest.values():
        record['preparation_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (PAPER/'publication/figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(f'Prepared {len(manifest)} publication vector figures from frozen evidence.')

if __name__ == '__main__':
    prepare()
