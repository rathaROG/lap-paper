"""Question-specific publication charts from the frozen light sessions only."""
import collections
import csv
import json
import math
import statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
from paper_artifacts_common import (REPO, DATA, FIGURES, TABLES, STYLE, ENVS,
                                    ENV_LINESTYLES, environment_marker,
                                    save, writecsv, classification)
WIDTH=6.46
INK,MUTED,GRID,TIE='#222222','#626262','#e6e6e6','#e9e8e4'
RC={'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,
    'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.edgecolor':'#444444','axes.linewidth':.6,
    'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
    'savefig.facecolor':'white','pdf.fonttype':42}


def data_from_runs(runs):
    return {d['env']:dict(root=d['root'],cases=d['cases'],seconds={key:e['seconds'] for key,e in d['single'].items()},batch=d['summary']['batch_contrasts'],raw=d) for d in runs}


def method_legend(ax, methods, marker_only=False, **kwargs):
    handles=[Line2D([],[],color=STYLE[m]['color'],marker=STYLE[m]['marker'],ms=3,lw=.9,
                    ls='none' if marker_only else '-',label=STYLE[m]['label']) for m in methods]
    return ax.legend(handles=handles,fontsize=6.3,handlelength=1.5,**kwargs)
def env_line(env):
    return dict(ls=ENV_LINESTYLES[env], lw=.8)

def env_marker(env, colour, size=3.4, marker="o"):
    return dict(**environment_marker(env, colour, size, marker), ls="none")

def ratio_axis(axis, lo, hi, which="x"):
    lo=min(lo,.9);hi=max(hi,1.1)
    ticks = [t for t in (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100)
             if lo <= t <= hi]
    fmt = FuncFormatter(lambda v, _: f"{v:g}×")
    if which == "x":
        axis.set_xscale("log"); axis.set_xlim(lo, hi)
        axis.xaxis.set_major_locator(FixedLocator(ticks)); axis.xaxis.set_major_formatter(fmt)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.axvspan(.97, 1.03, color=TIE, lw=0, zorder=0); axis.axvline(1, color="#9a9a9a", lw=.6, zorder=1)
        axis.grid(axis="x", color=GRID, lw=.4, zorder=0)
    else:
        axis.set_yscale("log"); axis.set_ylim(lo, hi)
        axis.yaxis.set_major_locator(FixedLocator(ticks)); axis.yaxis.set_major_formatter(fmt)
        axis.yaxis.set_minor_locator(NullLocator())
        axis.axhspan(.97, 1.03, color=TIE, lw=0, zorder=0); axis.axhline(1, color="#9a9a9a", lw=.6, zorder=1)
        axis.grid(axis="y", color=GRID, lw=.4, zorder=0)

def env_key(fig, x, y, kind="line"):
    if kind == "line":
        handles = [Line2D([], [], color=MUTED, **env_line(env), label=env)
                   for env, *_ in ENVS]
    else:
        handles = [Line2D([], [], **env_marker(env, MUTED), label=env)
                   for env, *_ in ENVS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(x, min(y,.99)), ncol=4, fontsize=7,
               handlelength=2.4, columnspacing=1.0)

def ratio(d, case_id, method, reference=None):
    reference = reference or d["cases"][case_id]["metadata"]["primary_comparator"]
    t, r = d["seconds"].get((case_id, method)), d["seconds"].get((case_id, reference))
    return None if t is None or r is None else r / t

def overview(data):
    """Complementary ECDF of SciPy/method over every planned single case."""
    panels = [("required", ["jvrust", "jvhybrid", "jvcpp", "jvs", "jvx"], "(a) Required assignment · 102 cases",
               "SciPy time / method time"),
              ("optional", ["jvrust", "jvcpp", "sparse:lapmod", "sparse:csgraph", "jvx"],
               "(b) Optional assignment · 166 cases", "Capped-SciPy time / method time")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), gridspec_kw=dict(wspace=.22))
    for ax, (objective, methods, title, xlabel) in zip(axes, panels):
        handles = []
        for method in methods:
            st = STYLE[method]
            for env, d in data.items():
                vals = sorted(r for cid, c in d["cases"].items()
                              if c["kind"] == "single" and c["objective"] == objective
                              and (r := ratio(d, cid, method)) is not None)
                n = len(vals)
                ax.step([min(vals)*.8]+vals+[max(vals)*1.01], [1]+[1-i/n for i in range(n)]+[0], where="pre", color=st["color"], alpha=.9,
                        **env_line(env))
            full = 102 if objective == "required" else 166
            handles.append(Line2D([], [], color=st["color"], lw=1.4,
                                  label=st["label"] + ("" if n == full else f" (n={n})")))
        ratio_axis(ax, .0002 if objective == "optional" else .005, 30)
        ax.xaxis.set_major_locator(FixedLocator([t for t in (.001, .01, .1, 1, 10) if t >= ax.get_xlim()[0]]))
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
        ax.set_xlabel(xlabel + " (log)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.legend(handles=handles, loc="lower left", fontsize=6.2, handlelength=1.6, borderaxespad=.2)
    axes[0].set_ylabel("Share of cases at least this fast")
    env_key(fig, .5, 1.02)
    fig.subplots_adjust(left=.09, right=.98, top=.82, bottom=.16)
    save(fig, "F1_overview_ccdf")

def geometry(data, q=.9, motion=.25, name="F2_geometry_lines", rectangle=False):
    groups = [("required", ["jvrust", "jvhybrid", "jvcpp", "jvs", "jvx", "scipy"]),
              ("optional", ["jvrust", "jvcpp", "sparse:lapmod", "sparse:csgraph", "jvx", "scipy:square",
                            "scipy:capped"])]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.2), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.35, 1], hspace=.12, wspace=.32))
    for col, (objective, methods) in enumerate(groups):
        top, bottom = axes[0, col], axes[1, col]
        for method in methods:
            st = STYLE[method]
            for env, d in data.items():
                cs = sorted((c for c in d["cases"].values() if c["family"] == "geometry"
                             and (c["cols"] == c["rows"]*5//4 and c["rows"] != c["cols"] if rectangle else c["rows"] == c["cols"]) and c["objective"] == objective
                             and c["metadata"]["q"] == q and c["metadata"]["motion"] == motion
                             and (objective == "required" or (c["metadata"].get("limit_over_delta") == 1.5
                                                               and c["gate"] is None))),
                            key=lambda c: c["rows"])
                pts = [(c["rows"], d["seconds"][(c["id"], method)], ratio(d, c["id"], method))
                       for c in cs if (c["id"], method) in d["seconds"]]
                xs = [p[0] for p in pts]
                mark = environment_marker(env, st["color"], 2.7, st["marker"])
                top.plot(xs, [p[1] * 1e3 for p in pts], color=st["color"], **mark, **env_line(env))
                if not method.startswith("scipy:capped") and method != "scipy":
                    bottom.plot(xs, [p[2] for p in pts], color=st["color"], **mark, **env_line(env))
        handles = [Line2D([], [], color=STYLE[m]["color"], marker=STYLE[m]["marker"], ms=3.5, lw=1.2,
                          label=STYLE[m]["label"] + (" (reference)" if m in ("scipy", "scipy:capped") else ""))
                   for m in methods]
        top.legend(handles=handles, loc="upper left", fontsize=6.3, ncol=2, handlelength=1.8,
                   columnspacing=.9, borderaxespad=.1)
        top.set_xscale("log", base=2)
        top.set_yscale("log")
        top.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:g}"));top.yaxis.set_minor_formatter(NullFormatter())
        top.set_ylim(top.get_ylim()[0], top.get_ylim()[1] * 60)
        top.set_xlim(6.5, 4096 * 1.5)
        top.grid(color=GRID, lw=.4)
        top.set_ylabel("Complete call (ms)")
        top.set_title(f"({'ab'[col]}) {objective.capitalize()} assignment" +
                      ("" if objective == "required" else ", L = 1.5δ, no gate"), loc="left", fontweight="bold")
        top.axvline(1024 * 1.41, color="#bdbdbd", lw=.6, ls=":")
        top.text(1024 * 1.5, top.get_ylim()[0] * 2, "JVX/JVS\nnot planned" if objective=="required" else "JVX/square\nnot planned", fontsize=6.3,
                 color=MUTED, va="bottom")
        ratio_axis(bottom, .01 if objective == "optional" else .1, 10, "y")
        bottom.set_ylabel(("SciPy" if objective == "required" else "Capped SciPy") + " / method")
        bottom.set_xlabel("Rows (5:4 columns/rows)" if rectangle else "Square size (rows = columns)")
        bottom.set_xticks([8, 32, 128, 512, 2048], ["8", "32", "128", "512", "2048"])
        bottom.xaxis.set_minor_locator(NullLocator())
    env_key(fig, .5, 1.0)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.09)
    save(fig, name)

def representations(data):
    methods = ["jvrust", "jvcpp", "scipy:dummies", "scipy:square", "jvx"]
    label = {"jvrust": "JVRUST (native rejection)", "jvcpp": "JVCPP (native rejection)",
             "scipy:dummies": "SciPy + r dummy columns", "scipy:square": "SciPy, (r+c)² square",
             "jvx": "JVX, square augmentation"}
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    d0 = data["L311"]
    panel = [cid for cid, c in d0["cases"].items() if c["kind"] == "single" and c["objective"] == "optional"
             and any(m.startswith("cost:") for m in c["methods"])]
    for row, method in enumerate(methods):
        st = STYLE[method]
        for k, (env, d) in enumerate(data.items()):
            vals = [ratio(d, cid, method) for cid in panel if (cid, method) in d["seconds"]]
            offset = (k - 1.5) * .11
            ax.plot(vals, [row + offset] * len(vals), **env_marker(env, st["color"], 3.0, st["marker"]), alpha=.85)
        blocks = ratio(d0, "formulations:optional_blocks256x320", method)
        if method in ("jvrust", "jvcpp"):
            ax.annotate("congested blocks\n256×320", (blocks, row), xytext=(-8, 0), textcoords="offset points",
                        fontsize=6.3, color=MUTED, ha="right", va="center")
    ax.set_yticks(range(len(methods)), [label[m] for m in methods])
    ax.set_ylim(len(methods) - .45, -.55)
    ratio_axis(ax, .02, 20)
    ax.set_xlabel("Capped-SciPy time / representation time (log); each marker is one of 14 optional workloads")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    env_key(fig, .55, 1.04, "marker")
    fig.subplots_adjust(left=.27, right=.98, top=.86, bottom=.2)
    save(fig, "F3_representations_strip")

def policy(data):
    arms = [("no_reductions", "Bypass reductions"), ("budget1", "Budget 1"), ("budget2", "Budget 2"),
            ("budget8", "Budget 8"), ("ties_off", "Free-column ties off"), ("cutoff_off", "Stall cutoff off"),
            ("ties_cutoff_off", "Ties and cutoff off")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.3), sharey=True, gridspec_kw=dict(wspace=.08))
    for ax, (lang, colour, title) in zip(axes, (("rust", STYLE["jvrust"]["color"], "(a) Rust"),
                                                ("cpp", STYLE["jvcpp"]["color"], "(b) C++"))):
        for row, (arm, _) in enumerate(arms):
            for k, (env, d) in enumerate(data.items()):
                vals = sorted(1 / r for cid in {c for (c, m) in d["seconds"] if m == f"policy:{lang}:{arm}"}
                              if (r := ratio(d, cid, f"policy:{lang}:{arm}", f"policy:{lang}:reference")))
                y = row + (k - 1.5) * .15
                q1, med, q3 = statistics.quantiles(vals, n=4, method="inclusive")
                ax.plot([vals[0], vals[-1]], [y, y], color=colour, lw=.7, ls=env_line(env)["ls"], alpha=.65, solid_capstyle="butt")
                ax.plot([q1, q3], [y, y], color=colour, lw=2.4, alpha=.45, solid_capstyle="butt")
                ax.plot([med], [y], **env_marker(env, colour, 3.4))
        ratio_axis(ax, .7, 12)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(arms)), [a[1]+(" (13)" if "cutoff" in a[0] else " (31)") for a in arms])
    axes[0].set_ylim(len(arms) - .4, -.6)
    env_key(fig, .55, 1.03, "marker")
    fig.text(.58, .045, "Intervention time / release-policy time (log; >1 is slower)\n"
             "marker: median over eligible cases · thick bar: interquartile range · thin line: extremes",
             ha="center", fontsize=6.5, linespacing=1.5)
    fig.subplots_adjust(left=.235, right=.99, top=.83, bottom=.2)
    save(fig, "F4_policy_forest")

def structure(data):
    sweeps = [("congestion", "row_fraction", "Congested row fraction", lambda v: f"{v:g}"),
              ("forbidden", "forbidden", "Forbidden-entry probability", lambda v: f"{v:g}"),
              ("ties", "levels", "Distinct cost levels", lambda v: f"{v:g}")]
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 4.4), sharey="row", gridspec_kw=dict(wspace=.12, hspace=.55))
    for col, (family, key, xlabel, fmt) in enumerate(sweeps):
        for rowi, objective in enumerate(("required", "optional")):
            ax = axes[rowi, col]
            methods = ["jvrust", "jvhybrid", "jvcpp"] if objective == "required" else ["jvrust", "jvcpp"]
            for m in methods:
                for env, d in data.items():
                    cs = sorted((c for c in d["cases"].values() if c["family"] == family and c["objective"] == objective),
                                key=lambda c: c["input_group"]["parameters"][key])
                    xs = [c["input_group"]["parameters"][key] for c in cs]
                    ax.plot(xs, [ratio(d, c["id"], m) for c in cs], color=STYLE[m]["color"],
                            **environment_marker(env, STYLE[m]["color"], 2.8, STYLE[m]["marker"]), **env_line(env))
            if family=="ties":ax.set_xscale("log",base=2);ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(xs, [fmt(x) for x in xs])
            if family=="forbidden":ax.tick_params(axis="x",labelrotation=45)
            method_legend(ax,methods,loc="upper right")
            ratio_axis(ax, .8, 40, "y")
            ax.set_title(f"{family.capitalize()} · {objective}", loc="left", fontweight="bold")
            if rowi == 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(("SciPy" if objective == "required" else "Capped SciPy") + " / native")
    axes[1, 0].annotate("case 041: Windows Rust\nand Linux C++ lose", xy=(.9, .93), xytext=(.59, 5.5),
                        fontsize=6.3, color=MUTED, ha="center",
                        arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=.5))
    env_key(fig,.5,.995)
    fig.subplots_adjust(left=.09, right=.99, top=.86, bottom=.12)
    save(fig, "F_structure_lines");plt.close(fig)

def sensitivity_heatmap(data):
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("speed", ["#f2f0eb", "#c6d9ef", "#6ea4d8", "#2c6fb0", "#0d3d73"])
    settings = [(.9, .25), (.9, 1.0), (.6, .25), (.6, 1.0)]
    limits, gates = [.5, 1.5, 3.0], [.75, 1.5, None]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.8), gridspec_kw=dict(wspace=.08))
    for ax, method in zip(axes, ("jvrust", "sparse:csgraph")):
        grid, text = [], []
        for q, motion in settings:
            for env, d in data.items():
                row, trow = [], []
                for gate in gates:
                    for lim in limits:
                        c = next(c for c in d["cases"].values() if c["family"] == "geometry" and c["rows"] == 256
                                 and c["cols"] == 320 and c["objective"] == "optional" and c["metadata"]["q"] == q
                                 and c["metadata"]["motion"] == motion and c["metadata"].get("limit_over_delta") == lim
                                 and c["metadata"].get("gate_over_delta") == gate)
                        v = ratio(d, c["id"], method)
                        row.append(v); trow.append(f"{v:.2f}" if v < 10 else f"{v:.1f}")
                grid.append(row); text.append(trow)
        norm = LogNorm(.1, 10) if method != "jvrust" else LogNorm(1, 12)
        image = ax.imshow(grid, cmap=cmap if method == "jvrust" else "RdBu", norm=norm, aspect="auto")
        bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=.05, pad=.2, aspect=30,
                           ticks=[1, 2, 5, 10, 12] if method == "jvrust" else [.2, .5, 1, 2, 5])
        bar.ax.set_xticklabels([f"{t:g}×" for t in bar.get_ticks()], fontsize=6.3)
        bar.outline.set_visible(False)
        for i, trow in enumerate(text):
            for j, t in enumerate(trow):
                v = grid[i][j]
                if .97<=v<=1.03:ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=TIE,lw=0))
                dark = (math.log10(v) > .65) if method == "jvrust" else abs(math.log10(v)) > .6
                ax.text(j, i, t, ha="center", va="center", fontsize=6.3, color="white" if dark else INK)
        for b in (2.5, 5.5):
            ax.axvline(b, color="white", lw=1.2)
        for b in range(4, 16, 4):
            ax.axhline(b - .5, color="white", lw=1.2)
        ax.set_xticks(range(9), [f"{l:g}" for _ in gates for l in limits])
        ax.set_xlabel("L/δ, grouped by gate 0.75δ | 1.5δ | none", fontsize=6.3)
        ax.set_yticks(range(16), list(data)*4 if method=="jvrust" else [])
        if method=="jvrust":
            for block,(q,m) in enumerate(settings):ax.text(-2.25,block*4+1.5,f"q={q:g}, sd={m:g}δ",ha="center",va="center",rotation=90,fontsize=6.3,clip_on=False)
        ax.tick_params(axis="both", length=0)
        ax.set_title(f"({'ab'[method != 'jvrust']}) {STYLE[method]['label']} vs capped SciPy",
                     loc="left", fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(.01, .985, "Within each setting the four rows are L311, L314, W311, W314.", fontsize=6,
             color=MUTED, va="top")
    fig.subplots_adjust(left=.1, right=.99, top=.88, bottom=.05)
    save(fig, "A2_sensitivity_heatmap")


def sparse(data):
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.4),sharex=True,sharey=True)
    records=[]
    for ax,(env,d) in zip(axes.flat,data.items()):
        for method in ['sparse:csgraph','sparse:lapmod']:
            xx=[];yy=[]
            for cid,c in d['cases'].items():
                if c['kind']!='single' or c['objective']!='optional':continue
                x=ratio(d,cid,method);y=ratio(d,cid,method,'jvrust')
                assert x and y
                xx.append(x);yy.append(y)
                records.append(dict(environment=env,case_id=cid,method=method,capped_over_sparse=x,rust_over_sparse=y))
            st=STYLE[method]
            ax.plot(xx,yy,**env_marker(env,st['color'],3,st['marker']),alpha=.75)
        ratio_axis(ax,.009,9);ratio_axis(ax,.00065,3.2,'y')
        ax.xaxis.set_major_locator(FixedLocator([.01,.1,1,5]));ax.yaxis.set_major_locator(FixedLocator([.001,.01,.1,1]))
        ax.set_title(env,loc='left',fontweight='bold')
        method_legend(ax,['sparse:csgraph','sparse:lapmod'],marker_only=True,loc='lower right')
        wins=sum(r['environment']==env and r['rust_over_sparse']>1.03 for r in records)
        ax.text(.03,.94,f'{wins}/332 points beat JVRUST',transform=ax.transAxes,va='top',fontsize=6.5,bbox=dict(fc='white',ec='none',alpha=.8))
    for ax in axes[:,0]:ax.set_ylabel('JVRUST / sparse time')
    for ax in axes[1,:]:ax.set_xlabel('Capped SciPy / sparse time (log)')
    env_key(fig,.5,.995,'marker')
    fig.text(.5,.015,'Upper right: faster than both references. Graph construction is included.',ha='center',fontsize=6.5)
    fig.subplots_adjust(left=.11,right=.985,top=.90,bottom=.12,hspace=.23,wspace=.16)
    save(fig,'F5_sparse_scatter');plt.close(fig)
    writecsv('F5_sparse_scatter.csv',records)


def rejection_lines(data):
    rows=list(csv.DictReader((DATA/'F11_rejection.csv').open(encoding='utf-8-sig')))
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,4.8),sharex='col',sharey='row',gridspec_kw={'height_ratios':[1.7,1]})
    for j,dist in enumerate(['uniform','blocks']):
        for method in ['jvrust','jvcpp','scipy:capped']:
            for env in data:
                rr=sorted([r for r in rows if r['environment']==env and r['distribution']==dist and r['method']==method],key=lambda r:float(r['limit']))
                assert len(rr)==11
                st=STYLE[method];kwargs={**env_line(env),**environment_marker(env,st['color'],2.6,st['marker'])}
                axes[0,j].plot([float(r['limit']) for r in rr],[float(r['microseconds']) for r in rr],color=st['color'],**kwargs)
                if method=='jvrust':axes[1,j].plot([float(r['limit']) for r in rr],[float(r['mean_rejected_rows_percent']) for r in rr],color=st['color'],**kwargs)
        axes[0,j].set_title(('Congested blocks' if dist=='blocks' else 'Uniform'),loc='left',fontweight='bold');axes[0,j].set_yscale('log');axes[0,j].yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));axes[0,j].yaxis.set_minor_formatter(NullFormatter())
        method_legend(axes[0,j],['jvrust','jvcpp','scipy:capped'],loc='lower right')
        for ax in axes[:,j]:
            ax.set_xscale('symlog',linthresh=.0005,linscale=.6);ax.set_xlim(0,5)
            ax.grid(color=GRID,lw=.4);ax.set_xticks([0,.0005,.002,.01,.05,.5,4]);ax.set_xticklabels(['0','.0005','.002','.01','.05','.5','4'],rotation=45,ha='right')
            if dist=='blocks':ax.axvspan(.05,.5,color='.94',zorder=-1)
        axes[1,j].set_xlabel('Limit L (linear near zero, log above .0005)')
    axes[0,0].set_ylabel('Complete call (µs)');axes[1,0].set_ylabel('Unmatched rows (%)')
    axes[1,1].annotate('K = 144; 43.75% unmatched',xy=(.15,43.75),xytext=(.003,75),fontsize=6.3,arrowprops=dict(arrowstyle='-',lw=.5))
    env_key(fig,.5,.995)
    fig.subplots_adjust(left=.10,right=.99,top=.90,bottom=.18,hspace=.15,wspace=.17)
    save(fig,'F11_rejection_lines');plt.close(fig)


def tables_and_evidence(data):
    families=['base','formulations','scaling','large_square','large_rectangle','extreme_rectangle','congestion','forbidden','ties','signed','rejection','geometry']
    counts=[];exceptions=[]
    case0=data['L311']['cases']
    for env,d in data.items():
        for family in families:
            ids=[cid for cid,c in d['cases'].items() if c['kind']=='single' and c['objective']=='optional' and c['family']==family]
            for m in ['jvrust','jvcpp']:
                cc=collections.Counter(classification(ratio(d,cid,m)) for cid in ids)
                counts.append(dict(environment=env,family=family,method=m,n=len(ids),wins=cc['win'],ties=cc['tie'],losses=cc['loss']))
    ids=[cid for cid,c in case0.items() if c['kind']=='single' and c['objective']=='optional' and any(ratio(d,cid,m)<=1.03 for d in data.values() for m in ['jvrust','jvcpp'])]
    assert len(ids)==3
    for cid in ids:
        for m in ['jvrust','jvcpp']:
            for env,d in data.items():exceptions.append(dict(case_id=cid,method=m,environment=env,ratio=ratio(d,cid,m),classification=classification(ratio(d,cid,m))))
    writecsv('S_native_family_counts.csv',counts);writecsv('S_native_exceptions.csv',exceptions)
    lines=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'& \multicolumn{2}{c}{L311} & \multicolumn{2}{c}{L314} & \multicolumn{2}{c}{W311} & \multicolumn{2}{c}{W314} \\',r'Family ($n$) & Rust & C++ & Rust & C++ & Rust & C++ & Rust & C++ \\',r'\midrule']
    for family in families:
        selected=[r for r in counts if r['family']==family]
        nums=['{wins}/{ties}/{losses}'.format(**r) for env in data for m in ['jvrust','jvcpp'] for r in selected if r['environment']==env and r['method']==m]
        lines.append(family.replace('_',' ').capitalize()+f' ({selected[0]["n"]}) & '+' & '.join(nums)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(TABLES/'S_native_family_counts.tex').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    labels={'extended:041_congestion_optional_256x320':r'Congestion 041, $256\times320$','formulations:optional_blocks256x320':r'Formulation blocks, $256\times320$','scaling:optional_blocks_128x160':r'Scaling blocks, $128\times160$'}
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Workload & Method & L311 & L314 & W311 & W314 \\',r'\midrule']
    for cid in ids:
        for m in ['jvrust','jvcpp']:lines.append(labels[cid]+' & '+STYLE[m]['label']+' & '+' & '.join(f'{ratio(d,cid,m):.3f}' for d in data.values())+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}'];(TABLES/'S_native_exceptions.tex').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    observations=[];forests=[]
    for env,d in data.items():
        for obj,methods in [('required',['jvrust','jvhybrid','jvcpp','jvs','jvx']),('optional',['jvrust','jvcpp','sparse:lapmod','sparse:csgraph','jvx'])]:
            for m in methods:
                rr=[(cid,ratio(d,cid,m)) for cid,c in d['cases'].items() if c['kind']=='single' and c['objective']==obj and ratio(d,cid,m)]
                assert len(rr)==(88 if obj=='required' and m in ['jvs','jvx'] else 145 if obj=='optional' and m=='jvx' else 102 if obj=='required' else 166)
                observations.extend(dict(environment=env,objective=obj,case_id=cid,method=m,ratio=v,n=len(rr)) for cid,v in rr)
        for lang in ['rust','cpp']:
            for arm in ['no_reductions','budget1','budget2','budget8','ties_off','cutoff_off','ties_cutoff_off']:
                vv=[d['seconds'][cid,f'policy:{lang}:{arm}']/d['seconds'][cid,f'policy:{lang}:reference'] for cid,c in d['cases'].items() if (cid,f'policy:{lang}:{arm}') in d['seconds']]
                assert len(vv)==(13 if 'cutoff' in arm else 31)
                q1,med,q3=statistics.quantiles(vv,n=4,method='inclusive')
                forests.append(dict(environment=env,language=lang,arm=arm,n=len(vv),minimum=min(vv),q1=q1,median=med,q3=q3,maximum=max(vv),definition='intervention / reference'))
    assert len(observations)==5164
    writecsv('F1_overview_ccdf.csv',observations);writecsv('F4_policy_forest_summary.csv',forests)
    points=[]
    for env,d in data.items():
        for cid,c in d['cases'].items():
            groups=[]
            if c['family']=='geometry' and c['rows']==c['cols'] and c['metadata']['q']==.9 and c['metadata']['motion']==.25 and (c['objective']=='required' or c['metadata'].get('limit_over_delta')==1.5 and c['gate'] is None):
                groups.append(('F2_geometry_lines',['jvrust','jvhybrid','jvcpp','jvs','jvx','scipy'] if c['objective']=='required' else ['jvrust','jvcpp','sparse:lapmod','sparse:csgraph','jvx','scipy:square','scipy:capped']))
            if c['objective']=='optional' and any(m.startswith('cost:') for m in c['methods']):groups.append(('F3_representations_strip',['jvrust','jvcpp','scipy:dummies','scipy:square','jvx']))
            if c['family'] in ['congestion','forbidden','ties']:groups.append(('F_structure_lines',['jvrust','jvhybrid','jvcpp'] if c['objective']=='required' else ['jvrust','jvcpp']))
            if c['family']=='geometry' and c['rows']==256 and c['cols']==320 and c['objective']=='optional':groups.append(('A2_sensitivity_heatmap',['jvrust','sparse:csgraph']))
            for figure,methods in groups:
                for m in methods:
                    if (cid,m) not in d['seconds']:continue
                    ref=c['metadata']['primary_comparator'];a=d['raw']['single'][cid,m];b=d['raw']['single'][cid,ref]
                    assert a['attempt_id']==b['attempt_id'] and a['input_hashes']==b['input_hashes']
                    points.append(dict(figure=figure,environment=env,case_id=cid,method=m,reference=ref,seconds=a['seconds'],reference_seconds=b['seconds'],ratio=b['seconds']/a['seconds'],attempt_id=a['attempt_id'],input_hashes=a['input_hashes'],metadata=c['metadata'],source=(d['root']/'campaign-summary.json').relative_to(REPO).as_posix()))
    assert dict(collections.Counter(r['figure'] for r in points))=={'F2_geometry_lines':488,'F3_representations_strip':280,'F_structure_lines':260,'A2_sensitivity_heatmap':288}
    writecsv('chart_points.csv',points)



def line_family(data,ids,name,title):
    first=data['L311'];obj=first['cases'][ids[0]]['objective'];assert all(first['cases'][cid]['objective']==obj for cid in ids)
    ref='scipy' if obj=='required' else 'scipy:capped'
    groups=[['jvrust','jvhybrid','jvcpp',ref],['jvs','jvx',ref]] if obj=='required' else [['jvrust','jvcpp',ref],['jvx','scipy:dummies','scipy:diagonal_dummies','scipy:square','sparse:csgraph','sparse:lapmod',ref]]
    fig,axes=plt.subplots(2,2,figsize=(WIDTH,5.3),sharex='col',gridspec_kw=dict(height_ratios=[1.4,1],hspace=.16,wspace=.24))
    fixed=len({first['cases'][cid]['rows'] for cid in ids})==1
    xs=[first['cases'][cid]['cols' if fixed else 'rows'] for cid in ids]
    order=np.argsort(xs);ids=[ids[i] for i in order];xs=sorted(xs)
    for col,methods in enumerate(groups):
        methods=[m for m in methods if any((cid,m) in first['seconds'] for cid in ids)]
        ratios_all=[];times=[]
        for m in methods:
            st=STYLE[m]
            for env,d in data.items():
                planned=np.array([m in d['cases'][cid]['methods'] for cid in ids])
                # Only declared absences are gaps; invalid observed values must
                # reach the bounds guard rather than being hidden as missing.
                ts=np.ma.array([d['seconds'][cid,m]*1000 if present else 0
                                for cid,present in zip(ids,planned)],mask=~planned)
                rr=np.ma.array([ratio(d,cid,m) if present else 0
                                for cid,present in zip(ids,planned)],mask=~planned)
                kwargs=dict(color=st['color'],**environment_marker(env,st['color'],2.8,st['marker']),**env_line(env))
                axes[0,col].plot(xs,ts,**kwargs)
                if m!=ref:axes[1,col].plot(xs,rr,**kwargs);ratios_all.extend(rr.compressed())
                times.extend(ts.compressed())
        top,bottom=axes[:,col]
        top.set_yscale('log');top.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));top.yaxis.set_minor_formatter(NullFormatter());top.set_ylim(min(times)*.7,max(times)*100);top.grid(color=GRID,lw=.4)
        method_legend(top,methods,loc='upper left',ncol=2,columnspacing=.6)
        ratio_axis(bottom,min(ratios_all)*.7,max(ratios_all)*1.4,'y')
        top.set_title('Native methods' if col==0 else 'Other planned methods',loc='left')
        for ax in [top,bottom]:
            ax.set_xscale('log',base=2);ax.set_xlim(min(xs)*.8,max(xs)*1.25);ax.set_xticks(xs,[f'{x:g}' for x in xs]);ax.xaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis='x',labelrotation=25 if len(xs)>5 else 0)
        bottom.set_xlabel('Columns' if fixed else 'Rows');top.set_ylabel('Complete call (ms)');bottom.set_ylabel('Matching SciPy / method')
        if any(any((cid,m) not in first['seconds'] for cid in ids) for m in methods):top.text(.98,.03,'Absent methods: not planned',transform=top.transAxes,ha='right',fontsize=6.3)
    env_key(fig,.5,.995);fig.subplots_adjust(left=.10,right=.985,top=.88,bottom=.14)
    save(fig,name);plt.close(fig)


def scaling_atlas(data):
    groups=collections.defaultdict(list)
    for cid,c in data['L311']['cases'].items():
        if c['family']=='scaling':groups[cid.split(':')[1].rsplit('_',1)[0]].append(cid)
        elif c['family'] in ['large_square','large_rectangle','extreme_rectangle']:
            key=c['family']+'_'+c['objective']+('_tall' if c['rows']>c['cols'] else '_wide' if c['rows']<c['cols'] else '')
            groups[key].append(cid)
    assert sum(map(len,groups.values()))==79
    atlas=[]
    for i,(group,ids) in enumerate(groups.items(),1):
        name=f'S_scaling_lines_{i:02d}';line_family(data,ids,name,group)
        atlas.append(dict(name=name,title=group.replace('_',' ').capitalize(),scope=f'{len(ids)} saved sizes. Top: complete-call milliseconds. Bottom: matching SciPy/method ratio. Each curve is one method and one environment, distinguished by four keyed line styles. Only planned observations are drawn. Exact values are in A1_scaling.csv.'))
    return atlas


def policy_atlas(data):
    atlas=[];arms=['reference','budget1','budget2','budget8']
    cases=[c for c in data['L311']['cases'].values() if 'policy:rust:budget1' in c['methods']];assert len(cases)==31
    # Four workloads/page: budgets are ordered; nominal interventions are
    # categorical dots and are never connected to the budget axis.
    for lang in ['rust','cpp']:
        language_label='Rust' if lang=='rust' else 'CPP'
        for page,start in enumerate(range(0,len(cases),4),1):
            part=cases[start:start+4]
            fig,axes=plt.subplots(len(part),2,figsize=(WIDTH,1.65*len(part)+.7),squeeze=False,gridspec_kw=dict(wspace=.18,hspace=.7))
            colour=STYLE['jvrust' if lang=='rust' else 'jvcpp']['color']
            for row,c in enumerate(part):
                for env,d in data.items():
                    a=axes[row,0];b=axes[row,1]
                    values=[1/ratio(d,c['id'],f'policy:{lang}:{arm}',f'policy:{lang}:reference') for arm in ['budget1','budget2','reference','budget8']]
                    a.plot([1,2,4,8],values,color=colour,**environment_marker(env,colour,3),**env_line(env))
                    noms=['no_reductions','ties_off','cutoff_off','ties_cutoff_off']
                    for j,arm in enumerate(noms):
                        v=ratio(d,c['id'],f'policy:{lang}:{arm}',f'policy:{lang}:reference')
                        if v:b.plot([j+(list(data).index(env)-1.5)*.08],[1/v],**env_marker(env,colour,3))
                    ratio_axis(a,.7,12,'y');ratio_axis(b,.7,12,'y')
                    a.set_xticks([1,2,4,8]);a.set_xlim(.5,8.5);b.set_xticks(range(4),['Bypass','Ties off','Cutoff off','Both off']);b.set_xlim(-.3,3.3)
                label=('R · ' if c['objective']=='required' else 'O · ')+c['id'].replace('geometry:','geo:').replace('formulations:','form:').replace('required','R').replace('optional','O')
                axes[row,0].set_title(label,loc='left',fontsize=6.3,pad=5);axes[row,0].set_ylabel('Intervention / reference')
                axes[row,0].set_xlabel('Reduction budget');axes[row,1].set_xlabel('Nominal intervention; absent = not planned')
                axes[row,1].set_yticklabels([])
            env_key(fig,.5,.995);fig.subplots_adjust(left=.10,right=.99,top=.91,bottom=.09)
            name=f'S_policy_lines_{lang}_{page}';save(fig,name);plt.close(fig)
            atlas.append(dict(name=name,title=f'{language_label} policy cases, page {page}/8',scope='One row per workload; R/O mean required/optional. Left: the ordered reduction-budget sweep. Right: independent nominal interventions, drawn only where planned. Every mark/curve is one environment. Nominal-intervention positions run L311, L314, W311, W314 from left to right. Ratios are intervention/reference, so above one is slower. Exact case identities and all values remain in the indexed S3 policy timing exports and F5_budget_data.csv.'))
    return atlas


def layout_atlas(data):
    atlas=[]
    for category in ['shape','cost']:
        cases=[c for c in data['L311']['cases'].values() if any(m.startswith(category+':') for m in c['methods'])]
        for backend in ['scipy','jvx','jvs']:
            methods=list(dict.fromkeys(m for c in cases for m in c['methods'] if m.startswith(f'{category}:{backend}:')))
            fig,ax=plt.subplots(figsize=(WIDTH,max(3.3,.31*len(methods)+1.4)))
            vals=[]
            for row,m in enumerate(methods):
                for k,(env,d) in enumerate(data.items()):
                    rr=[ratio(d,c['id'],m) for c in cases if ratio(d,c['id'],m)]
                    vals.extend(rr);ax.plot(rr,[row+(k-1.5)*.13]*len(rr),**env_marker(env,STYLE[backend]['color'],3,STYLE[backend]['marker']),alpha=.75)
            ratio_axis(ax,min(vals)*.7,max(vals)*1.35)
            ax.set_yticks(range(len(methods)),[m.split(':',2)[2].replace('_',' ') for m in methods]);ax.set_ylim(len(methods)-.5,-.5)
            ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0);ax.set_xlabel('Matching SciPy / complete representation time (log)')
            env_key(fig,.59,.99,'marker');fig.subplots_adjust(left=.33,right=.98,top=.83,bottom=.17)
            name=f'S_layout_strip_{category}_{backend}';save(fig,name);plt.close(fig)
            atlas.append(dict(name=name,title=f'{category.capitalize()} representations: {backend}',scope=f'All {len(cases)} eligible workloads per layout. Each marker is one workload/environment. Within each layout, lanes run L311, L314, W311, W314 from top to bottom. Layouts are categorical and are therefore not connected by lines. Exact case-level values remain in S3_layout_timing.csv.'))
    return atlas


def write_atlas(repo=REPO):
    repo=Path(repo)
    summary=json.loads((DATA/'campaign_chart_summary.json').read_text(encoding='utf-8'))
    extra=json.loads((DATA/'support_atlas.json').read_text(encoding='utf-8')) if (DATA/'support_atlas.json').exists() else {}
    atlas=summary['atlas_figures']+extra.get('atlas_figures',[])
    def tex(s):
        return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('δ',r'$\delta$').replace('×',r'$\times$')
    lines=[]
    for i,item in enumerate(atlas,1):
        lines += [r'\clearpage',r'\section*{Atlas '+str(i)+'. '+tex(item['title'])+'}',r'\noindent '+tex(item['scope']),r'\par\medskip',r'\begin{center}',r'\includegraphics[width=\textwidth,height=.82\textheight,keepaspectratio]{images/current/'+item['name']+'.pdf}',r'\end{center}']
    for table in extra.get('atlas_tables',[]):lines += [r'\clearpage',r'\input{generated/'+table['path']+'}']
    (TABLES/'supplementary_chart_atlas.tex').write_bytes(('\n'.join(lines)+'\n').encode('utf-8'))
    source=r'''\documentclass[10pt]{article}
\usepackage[margin=0.9in,a4paper]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{graphicx,booktabs,array}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\title{Native Rectangular Assignment: Saved-Result Figure Atlas}
\author{}
\date{}
\begin{document}
\maketitle
This supplement shows ordered workload families and named controlled comparisons from the four saved light sessions. Each curve or marker represents one environment. The four line styles identify L311 (solid), L314 (dashed), W311 (dotted) and W314 (dash-dot). Markers are filled for Linux and open for Windows; smaller markers identify Python 3.11 and larger markers Python 3.14. Each figure keys the environments explicitly. Coincident values can still overlap. Colors identify methods consistently with the main paper. Gray bands mark ratios in $[0.97,1.03]$; classifications use unrounded values. Missing observations mean ``not planned'', never zero. Policy ratios are intervention/reference (above one is slower); other ratio axes state their direction. Exact case values, attempt identities and saved validation outcomes remain in the indexed CSV exports; environments are never averaged.
\input{generated/supplementary_chart_atlas.tex}
\end{document}
'''
    (repo/'latex/supplementary_figures.tex').write_bytes(source.encode('utf-8'))
    result={'atlas_figures':atlas,'atlas_tables':extra.get('atlas_tables',[]),'source':'latex/supplementary_figures.tex'}
    (DATA/'supplementary_chart_atlas.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    return result


def remove_retired_figures():
    import re
    names={'F2_coverage','F3_geometry_grid','F4_formulations_grid','F5_policy_anchors','F6_sparse_grid','F8_batch_grid','F12_orientation_grid',
           'F9_applications','F10_lsape','S6_lsape_grid_1','S6_lsape_grid_2','S6_lsape_grid_3','S_geometry_5to4_grid',
           'S_batch_standard','S_batch_worker_scaling','S_batch_reuse_1w','S_batch_reuse_4w','S_rejection_uniform','S_rejection_blocks'}
    for p in FIGURES.iterdir():
        stem=p.stem
        if p.suffix in ['.pdf','.png'] and (stem in names or stem.startswith(('S_cases_','S_geometry_runtime_')) or re.fullmatch(r'S_layout_(shape|cost)_(scipy|jvx|jvs)_[12]',stem) or re.fullmatch(r'S_policy_(rust|cpp)_[12]',stem)):
            p.unlink()
    for p in [DATA/'supplementary_case_grid_index.csv',DATA/'main_grid_ratio_values.csv',DATA/'F8_batch_grid_values.csv',DATA/'F6_sparse_grid.csv',DATA/'F2_coverage_counts.csv',DATA/'F2_native_exceptions.csv',TABLES/'supplementary_grid_atlas.tex',TABLES/'A1_scaling_figures.tex',DATA/'grid_redesign_summary.json',DATA/'grid_figure_manifest.json',DATA/'supplementary_grid_atlas.json']:
        if p.exists():p.unlink()


def generate_charts(runs):
    import paper_artifacts_common as common
    common.TYPOGRAPHY.clear()
    data=data_from_runs(runs);atlas=[]
    titles={
        'F1_overview_ccdf':'Complementary empirical distributions of matching SciPy/method ratios. Each curve is one solver and one environment over its planned cases. Required natives n=102 and optional natives/sparse n=166; required JVX/JVS n=88 and optional JVX n=145. Line styles identify L311 solid, L314 dashed, W311 dotted and W314 dash-dot. Ratios below the gray tie band are losses.',
        'F2_geometry_lines':'Square geometry at q=0.9 and motion standard deviation 0.25δ. Required assignment is left; optional assignment with L=1.5δ and no gate is right. Top: complete-call milliseconds. Bottom: matching SciPy/method ratios. Each curve is one method/environment. JVX/JVS (required) and JVX/square SciPy (optional) are not planned above 1024 rows.',
        'F3_representations_strip':'The 14 optional formulation workloads, with one marker per case/environment in each representation row. Ratios are capped SciPy/complete representation time. Filled/open markers identify Linux/Windows and smaller/larger markers identify Python 3.11/3.14. Within-row lanes run L311, L314, W311, W314 from top to bottom; the congested-block workload at 256×320 is labelled.',
        'F_structure_lines':'Congestion, forbidden-entry and quantized-tie sweeps, required above and optional below. Each line is one native method/environment, and ordered x positions retain the generator sweep values. The marked case 041 contains the Windows JVRUST and Linux JVCPP losses. Wall time alone does not identify their internal work.',
        'F4_policy_forest':'Each row is an intervention, separated into L311, L314, W311, W314 lanes from top to bottom. Filled/open markers identify Linux/Windows and smaller/larger markers identify Python 3.11/3.14. Markers are within-environment medians across eligible cases; thick bars are interquartile ranges and thin lines the full range. Parentheses give case counts. Ratios are intervention/release-policy time: above one is slower. Initializer bypass is a nominal intervention, not a budget value.',
        'F5_sparse_scatter':'All 166 optional workloads for CSGRAPH and LAPMOD, separated by environment. Each point compares the same sparse complete-call time with capped SciPy (x) and JVRUST (y). Upper-right points beat both references. Graph construction is timed.',
        'A2_sensitivity_heatmap':'The only heatmap: optional 256×320 geometry, four survival/motion settings crossed with three gates and three limits. Each setting has four environment rows in L311,L314,W311,W314 order. Printed values are capped SciPy/method ratios, JVRUST left and CSGRAPH right. Values are individual environment observations, not pooled summaries.',
        'F11_rejection_lines':'Uniform and congested-block rejection-limit sweeps. Linked upper panels show complete-call latency, lower panels the unmatched fraction from five saved validation cardinalities. Each line is one environment. The shaded block interval has K=144 throughout while native latency changes; no new solve determines these cardinalities.'}
    jobs=[('overview',lambda:overview(data)),('geometry',lambda:geometry(data)),('representations',lambda:representations(data)),('structure',lambda:structure(data)),('policy',lambda:policy(data)),('sparse',lambda:sparse(data)),('A2',lambda:sensitivity_heatmap(data)),('rejection',lambda:rejection_lines(data))]
    for name,fn in jobs:
        print('Drawing',name,flush=True)
        with plt.rc_context(plt.rcParamsDefault):
            with plt.rc_context(RC):fn()
        plt.close('all')
    tables_and_evidence(data)
    for q,motion,rectangle,label in [(.6,1.,False,'q06_square'),(.9,.25,True,'q09_5to4'),(.6,1.,True,'q06_5to4')]:
        name='S_geometry_lines_'+label
        with plt.rc_context(plt.rcParamsDefault):
            with plt.rc_context(RC):geometry(data,q,motion,name,rectangle)
        plt.close('all')
        atlas.append(dict(name=name,title=f'Geometry: q={q:g}, sd={motion:g}δ, '+('5:4' if rectangle else 'square'),scope='Ten saved sizes per objective. Runtime above, matching SciPy/method ratios below. Each curve is one method/environment. Only planned methods are connected; the light cutoff is marked.'))
    for fn in [scaling_atlas,policy_atlas,layout_atlas]:
        with plt.rc_context(plt.rcParamsDefault):
            with plt.rc_context(RC):atlas.extend(fn(data))
        plt.close('all')
    result={'main_figures':titles,'atlas_figures':atlas,'counts':{'ccdf_observations':5164,'square_geometry_cases':20,'formulation_cases':14,'structure_cases':26,'policy_eligible_cases':31,'cutoff_eligible_cases':13,'heatmap_cells':288,'sparse_points':1328,'scaling_cases':79},'sources':{d['env']:(d['root']/'campaign-summary.json').relative_to(REPO).as_posix() for d in runs}}
    (DATA/'campaign_chart_summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'))
    write_atlas();remove_retired_figures()
    return result
