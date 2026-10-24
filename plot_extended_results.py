"""Plot the separate extension and focused views of preserved existing results."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np

PAPER = Path(__file__).resolve().parent
STYLES = {
    "jvrust": ("JVRUST", "#176ca4", "o", "-"),
    "jvhybrid": ("JVHYBRID", "#78519a", "s", "--"),
    "jvcpp": ("JVCPP", "#2a9d8f", "X", "-."),
    "scipy": ("SciPy", "#bd4141", "D", ":"),
    "scipy_capped": ("SciPy capped", "#35616f", "P", "--"),
    "scipy_dummies": ("SciPy dummy columns", "#32905d", "v", "-."),
    "scipy_square": ("SciPy square", "#bd4141", "D", ":"),
    "jvx": ("JVX", "#d17c26", "^", "-"),
}
REQ = ("jvrust", "jvhybrid", "jvcpp", "scipy")
OPT = ("jvrust", "jvcpp", "scipy_capped", "scipy_dummies")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table(path):
    return {(r["case"],r["solver"]):r for r in rows(path)}


def decorate(ax, xlabel, ylabel="Time per matrix (ms)", logx=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log", base=2)
    ax.grid(alpha=.2, linewidth=.6)
    ax.set_axisbelow(True)


def curves(ax, selected, summary, candidates, xs=None):
    xs = np.asarray(xs if xs is not None else [r["n"] for r in selected])
    for method in candidates:
        name, color, marker, style = STYLES[method]
        values = [summary.get((r["case"],method)) for r in selected]
        med, lo, hi = [np.array([float(r[field])/1e6 if r else np.nan for r in values])
                       for field in ("median_ns", "q25_ns", "q75_ns")]
        ax.plot(xs, med, label=name, color=color, marker=marker, linestyle=style, markersize=4,
                markerfacecolor="white", linewidth=1.15)
        ax.fill_between(xs, lo, hi, color=color, alpha=.09)
        ax.errorbar(xs, med, yerr=(med-lo,hi-med), fmt="none", ecolor=color, linewidth=.65, capsize=1.5)


def legend_and_save(fig, axes, destination, stem, title, note, manifest):
    handles, labels = [], []
    for ax in np.asarray(axes).ravel():
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h,l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.suptitle(title, y=.995, fontsize=11)
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5,.955),
                   ncol=min(4,len(labels)), frameon=False, fontsize=8, columnspacing=1.5)
    fig.text(.015, .012, note, fontsize=7, va="bottom")
    fig.tight_layout(rect=(0,.065,1,.865))
    files = []
    for suffix in ("png", "svg", "pdf"):
        target = destination / f"{stem}.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        files.append(target.name)
    plt.close(fig)
    manifest.append(dict(stem=stem, title=title, note=note, files=files))
    print(stem, flush=True)


def new_timing_charts(output, manifest):
    plan = read(output / "plan.json")
    specs = plan["timing"]
    summary = table(output / "summary.csv")
    destination = output / "figures"
    note = "Four paired float64 inputs; nine shuffled rounds. Medians and timing-round IQRs. Focused solver subset; gaps denote exclusions."
    fig, axes = plt.subplots(1,2,figsize=(9.5,4.8))
    for ax, regime, candidates in zip(axes,("required","optional"),(REQ,OPT)):
        selected = sorted((r for r in specs if r["family"]=="large_square" and r["regime"]==regime),key=lambda r:r["n"])
        curves(ax,selected,summary,candidates)
        decorate(ax,"Square dimension N",logx=True)
        ax.set_title("Required assignment" if regime=="required" else r"Finite limit: $L=2/N$")
        ticks=[r["n"] for r in selected]
        ax.set_xticks(ticks,[str(n) for n in ticks]); ax.xaxis.set_minor_locator(NullLocator())
    legend_and_save(fig,axes,destination,"large_square","Large square matrices",note,manifest)
    for family,stem,title in (("large_rectangle","large_rectangles","Large paired rectangles"),
                              ("extreme_rectangle","extreme_rectangles","Extreme aspect ratios: smaller dimension 64")):
        fig,axes=plt.subplots(2,2,figsize=(10,7.5))
        for row,(regime,candidates) in enumerate((("required",REQ),("optional",OPT))):
            for col,tall in enumerate((False,True)):
                ax=axes[row,col]
                selected=sorted((r for r in specs if r["family"]==family and r["regime"]==regime
                                 and (r["n"]>r["m"])==tall),key=lambda r:max(r["n"],r["m"]))
                xs=[max(r["n"],r["m"]) for r in selected]
                curves(ax,selected,summary,candidates,xs)
                decorate(ax,"Original matrix shape (N x M)",logx=True)
                ax.set_xticks(xs,[f"{r['n']} x {r['m']}" for r in selected],rotation=20)
                ax.xaxis.set_minor_locator(NullLocator())
                ax.set_title(f"{'Tall' if tall else 'Wide'}; "+("required" if regime=="required" else "L = 2 / max(N,M)"))
        legend_and_save(fig,axes,destination,stem,title,note+"\nWide/tall inputs are exact transposes.",manifest)
    for regime,candidates in (("required",REQ),("optional",OPT)):
        shapes=sorted({(r["n"],r["m"]) for r in specs if r["family"]=="congestion"})
        fig,axes=plt.subplots(len(shapes),3,figsize=(12,3.2*len(shapes)+1.2),squeeze=False)
        for row,(n,m) in enumerate(shapes):
            for col,(family,field,label) in enumerate((("congestion","row_fraction","Competing-row fraction (nominal)"),
                    ("forbidden","forbidden","Nominal forbidden probability"),("ties","levels","Number of quantization levels"))):
                selected=sorted((r for r in specs if r["family"]==family and r["regime"]==regime and (r["n"],r["m"])==(n,m)),key=lambda r:r[field])
                curves(axes[row,col],selected,summary,candidates,[r[field] for r in selected])
                decorate(axes[row,col],label,logx=family=="ties")
                axes[row,col].set_title(f"{n} x {m}: {family}")
                if family=="ties":
                    ticks=[r[field] for r in selected]
                    axes[row,col].set_xticks(ticks,[str(x) for x in ticks]); axes[row,col].xaxis.set_minor_locator(NullLocator())
        legend_and_save(fig,axes,destination,f"structure_{regime}",f"Structured costs: {regime} assignment",
            note+"\nInputs are paired across each sweep. Forbidden cases retain a feasible diagonal. Optional L: blocks 0.5; forbidden 2/M; ties 0.25.",manifest)
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.8))
    for ax,(regime,candidates) in zip(axes,(("required",REQ),("optional",OPT))):
        selected=sorted((r for r in specs if r["family"]=="signed" and r["regime"]==regime),key=lambda r:r["n"])
        curves(ax,selected,summary,candidates,list(range(len(selected))))
        decorate(ax,"Original matrix shape (N x M)")
        ax.set_xticks(range(len(selected)),[f"{r['n']} x {r['m']}" for r in selected])
        ax.set_title("Required" if regime=="required" else "Finite limit L = -0.25")
    legend_and_save(fig,axes,destination,"signed_orientation","Signed costs and orientation",note+"\nCosts in [-1,1), 50% nominal forbidden entries, feasible diagonal retained.",manifest)


def rejection_chart(output, manifest):
    specs=read(output / "plan.json")["timing"]
    summary=table(output / "summary.csv")
    checks=read(output / "checks.json")
    groups=sorted({(r["n"],r["m"],r["distribution"]) for r in specs if r["family"]=="rejection"})
    fig,axes=plt.subplots(2,len(groups),figsize=(3.2*len(groups),7),squeeze=False)
    for col,(n,m,distribution) in enumerate(groups):
        selected=sorted((r for r in specs if r["family"]=="rejection" and (r["n"],r["m"],r["distribution"])==(n,m,distribution)),key=lambda r:r["limit"])
        xs=[r["limit"] for r in selected]
        curves(axes[0,col],selected,summary,OPT,xs)
        decorate(axes[0,col],"Rejection limit L")
        axes[0,col].set_title(f"{n} x {m}; "+("congested" if distribution=="blocks" else "uniform, 50% forbidden"))
        for method in OPT:
            label,color,marker,style=STYLES[method]
            values=[]
            for spec in selected:
                counts=[r["pairs"] for r in checks if r["case"]==spec["case"] and r["solver"]==method]
                values.append(100*(1-np.mean(counts)/min(n,m)) if counts else np.nan)
            axes[1,col].plot(xs,values,label=label,color=color,marker=marker,linestyle=style,markersize=3,markerfacecolor="white")
        axes[1,col].set(xlabel="Rejection limit L",ylabel="Smaller-side rows unmatched (%)",ylim=(-3,103))
        axes[1,col].grid(alpha=.2)
        for ax in axes[:,col]:
            ax.set_xscale("symlog",linthresh=.001,linscale=.6)
            ax.set_xlim(-.00008,4.5)
            ticks=[0,.001,.01,.1,1,4]
            ax.set_xticks(ticks,["0",".001",".01",".1","1","4"])
            ax.xaxis.set_minor_locator(NullLocator())
    legend_and_save(fig,axes,output/"figures","rejection_sweep","Runtime and realized rejection on fixed inputs",
        "Same four cost matrices and forbidden masks at every limit. Top: median and timing-round IQR. Bottom: mean unmatched fraction.\nDifferent optimal pair sets/cardinalities at ties are permitted; objectives are checked. The limit axis includes zero using a symmetric-log scale.",manifest)


def memory_chart(output, manifest):
    data=rows(output/"memory_summary.csv")
    fig,axes=plt.subplots(2,2,figsize=(10,7.5))
    for row,regime in enumerate(("required","optional")):
        candidates=("jvrust","jvcpp","scipy","jvx") if regime=="required" else ("jvrust","jvcpp","scipy_capped","scipy_dummies","scipy_square")
        for col,tall in enumerate((False,True)):
            ax=axes[row,col]
            max_upper=0.
            for method in candidates:
                selected=sorted((r for r in data if r["regime"]==regime and r["solver"]==method
                                 and (int(r["n"])>=int(r["m"]) if tall else int(r["n"])<=int(r["m"]))),
                                key=lambda r:max(int(r["n"]),int(r["m"])))
                x=np.array([max(int(r["n"]),int(r["m"])) for r in selected])
                med,lo,hi=[np.array([float(r[key])/2**20 for r in selected]) for key in ("median_bytes","q25_bytes","q75_bytes")]
                max_upper=max(max_upper,float(hi.max()))
                label,color,marker,style=STYLES[method]
                ax.plot(x,med,label=label,color=color,marker=marker,linestyle=style,markersize=4,markerfacecolor="white",clip_on=False)
                ax.fill_between(x,lo,hi,color=color,alpha=.1)
                ax.errorbar(x,med,yerr=(med-lo,hi-med),fmt="none",ecolor=color,linewidth=.7,capsize=2)
            ax.set_xscale("log",base=2)
            ax.set_yscale("symlog",linthresh=.25,linscale=.7)
            ax.set_ylim(0,max(.5,max_upper*1.4))
            ax.set(xlabel="Larger matrix dimension; smaller dimension = 256",
                   ylabel="Additional peak working set (MiB)",
                   title=f"{'Tall' if tall else 'Wide'}; "+("required" if regime=="required" else "finite limit L = 0.5"))
            ax.set_xticks([256,512,1024,2048,4096],["256","512","1024","2048","4096"])
            ax.xaxis.set_minor_locator(NullLocator()); ax.grid(alpha=.2)
    legend_and_save(fig,axes,output/"figures","memory_sweeps","Repeated fresh-process memory measurements",
        "Median and IQR across five fresh processes per condition. Input already resident. High-water increments are not allocation counts.\nThe vertical scale is linear from zero to 0.25 MiB and logarithmic above it; zero increments remain visible. Compact SciPy formulations are included.",manifest)


def dots(ax, selected, summary, candidates):
    x=np.arange(len(selected))
    for k,method in enumerate(candidates):
        label,color,marker,_=STYLES[method]
        points=[summary.get((r["case"],method)) for r in selected]
        med,lo,hi=[np.array([float(r[key])/1e6 if r else np.nan for r in points]) for key in ("median_ns","q25_ns","q75_ns")]
        ax.errorbar(x+(k-(len(candidates)-1)/2)*.16,med,yerr=(med-lo,hi-med),label=label,
                    color=color,marker=marker,linestyle="none",markersize=4,markerfacecolor="white",capsize=2)
    decorate(ax,"Workload")
    ax.set_xticks(x)


def case_label(spec):
    text=f"{spec['n']} x {spec['m']}"
    if spec.get("blocks"):
        text+=" blocks"
    if spec.get("ties"):
        text+=" ties"
    if spec.get("signed"):
        text+=" signed"
    limit=spec.get("limit")
    if limit is not None and np.isfinite(float(limit)):
        text+=f"\nL={float(limit):g}"
    return text


def focused_existing(output, manifest):
    import run_experiments as base
    old=PAPER/"results"
    required=[r for r in base.case_specs() if np.isinf(r["limit"])]
    finite=read(old/"formulations/datasets.json")
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    for ax,selected,summary,candidates,title in ((axes[0],required,table(old/"single_summary.csv"),REQ,"Required assignment: original main suite"),
            (axes[1],finite,table(old/"formulations/summary.csv"),OPT,"Finite-limit assignment: original paired formulation suite")):
        dots(ax,selected,summary,candidates)
        ax.set_xticklabels([case_label(r) for r in selected],rotation=30,ha="right",fontsize=7)
        ax.set_title(title)
    legend_and_save(fig,axes,output/"figures","focused_existing_cases","Focused views of the preserved Python 3.14 results",
        "Original measurements retained unchanged; no timings combined across suites. Medians and timing-round IQRs.\nThe finite-limit panel uses capped and dummy-column SciPy from the paired formulation comparison. Congested cases are retained.",manifest)
    samples=rows(old/"formulations/samples.csv")
    paired={(r["case"],r["solver"],int(r["round"])):float(r["nanoseconds"]) for r in samples}
    fig,ax=plt.subplots(figsize=(8.6,6.6))
    ratios=[]
    for k,method in enumerate(("jvrust","jvcpp")):
        label,color,marker,_=STYLES[method]
        med,lo,hi=[],[],[]
        for spec in finite:
            values=[paired[spec["case"],"scipy_capped",r]/paired[spec["case"],method,r] for r in range(9)]
            q=np.percentile(values,[25,50,75]); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
            ratios.append(dict(case=spec["case"],solver=method,median_paired_speedup=float(q[1]),q25=float(q[0]),q75=float(q[2])))
        med,lo,hi=map(np.asarray,(med,lo,hi))
        ax.errorbar(med,np.arange(len(finite))+(k-.5)*.2,xerr=(med-lo,hi-med),label=label,color=color,
                    marker=marker,linestyle="none",markersize=5,markerfacecolor="white",capsize=2)
    ax.axvline(1,color="#555555",linestyle=":",linewidth=1)
    ax.set_xscale("log"); ax.grid(axis="x",alpha=.2)
    ax.xaxis.set_major_locator(FixedLocator([.25,.5,1,2,5,10,20]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(range(len(finite)),[case_label(r).replace("\n",", ") for r in finite],fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("Capped SciPy time / native time (>1: native faster; <1: SciPy faster)")
    legend_and_save(fig,[ax],output/"figures","formulation_speedup","Paired speedups against capped SciPy",
        "Median and IQR of nine within-round time ratios, using the preserved formulation samples. Ratios are not confidence intervals.\nAll ten cases are shown, including the congested-block reversal.",manifest)
    (output/"paired_formulation_speedups.json").write_text(json.dumps(ratios,indent=2)+"\n",encoding="utf-8")
    datasets=read(old/"scaling/datasets.json"); summary=table(old/"scaling/summary.csv")
    fig,axes=plt.subplots(2,3,figsize=(11.5,7.5))
    names={"square":"N = M","wide":"M = 4N","fixed":"N = 64"}
    for row,(regime,candidates) in enumerate((("required",REQ),("optional",OPT))):
        for col,family in enumerate(("square","wide","fixed")):
            selected=sorted((r for r in datasets if r["regime"]==regime and r["family"]==family),key=lambda r:r["n"]+r["m"])
            xs=[r["n"]+r["m"] for r in selected]
            curves(axes[row,col],selected,summary,candidates,xs)
            decorate(axes[row,col],"Sum of dimensions N + M",logx=True)
            ticks=sorted(set(xs[::2]+[xs[-1]]))
            axes[row,col].set_xticks(ticks,[str(x) for x in ticks])
            axes[row,col].xaxis.set_minor_locator(NullLocator())
            axes[row,col].set_title(f"{names[family]}; "+("required" if regime=="required" else "L = 0.5"))
    legend_and_save(fig,axes,output/"figures","focused_existing_scaling","Focused scaling views of the preserved run",
        "Same saved timing samples as the original scaling figures; selected solver subset. Medians and timing-round IQRs.\nAll original uniform finite-limit instances match the complete smaller side; the new rejection sweep characterizes other regimes.",manifest)
    selected=sorted((r for r in datasets if r["family"]=="blocks"),key=lambda r:r["n"])
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    curves(ax,selected,summary,OPT,[r["n"] for r in selected]); decorate(ax,"Smaller dimension N; M = 5N/4",logx=True)
    ax.set_xticks([r["n"] for r in selected],[str(r["n"]) for r in selected]); ax.xaxis.set_minor_locator(NullLocator())
    legend_and_save(fig,[ax],output/"figures","focused_existing_blocks","Congested-block crossover in the preserved run",
        "Original finite-limit sweep, L = 0.5. Selected native and compact SciPy methods; medians and timing-round IQRs.\nThe full original comparison remains available in the preserved figures and results.",manifest)


def implementation_ratios(output, manifest):
    specs=read(output/"plan.json")["timing"]
    families=("large_square","large_rectangle","extreme_rectangle")
    required=[r for r in specs if r["family"] in families and r["regime"]=="required"]
    finite=[r for r in specs if r["family"] in families and r["regime"]=="optional"]
    samples={(r["case"],r["solver"],int(r["round"])):float(r["nanoseconds"]) for r in rows(output/"samples.csv")}
    complete=set(table(output/"summary.csv"))
    fig,axes=plt.subplots(1,2,figsize=(10.5,8),sharey=True)
    export=[]
    for ax,selected,candidates,title in ((axes[0],required,("jvcpp","jvhybrid"),"Required"),
                                       (axes[1],finite,("jvcpp",),"Finite limit L = 2 / max(N,M)")):
        for k,method in enumerate(candidates):
            label,color,marker,_=STYLES[method]
            y,med,lo,hi=[],[],[],[]
            for i,spec in enumerate(selected):
                if (spec["case"],method) not in complete or (spec["case"],"jvrust") not in complete:
                    continue
                values=[samples[spec["case"],"jvrust",r]/samples[spec["case"],method,r] for r in range(9)]
                q=np.percentile(values,[25,50,75]); lo.append(q[0]); med.append(q[1]); hi.append(q[2]); y.append(i+(k-(len(candidates)-1)/2)*.2)
                export.append(dict(case=spec["case"],solver=method,median_paired_speedup=float(q[1]),q25=float(q[0]),q75=float(q[2])))
            med,lo,hi=map(np.asarray,(med,lo,hi))
            ax.errorbar(med,y,xerr=(med-lo,hi-med),label=label,color=color,marker=marker,linestyle="none",markersize=4,markerfacecolor="white",capsize=2)
        ax.axvline(1,color="#555555",linestyle=":",linewidth=1)
        ax.set_xscale("log"); ax.grid(axis="x",alpha=.2)
        ax.xaxis.set_major_locator(FixedLocator([.25,.5,.75,1,1.25]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f"{value:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_title(title)
        ax.set_xlabel("JVRUST time / implementation time\n>1: implementation faster; <1: JVRUST faster")
    axes[0].set_yticks(range(len(required)),[f"{r['n']} x {r['m']}" for r in required])
    axes[0].invert_yaxis()
    legend_and_save(fig,axes,output/"figures","large_implementation_ratios","Implementation differences on the large inputs",
        "Median and IQR of nine paired round ratios. Same inputs for all implementations; baseline is JVRUST = 1.\nCompanion large-matrix runtime charts include the SciPy baselines. These ratios compare implementations/builds, not programming languages in general.",manifest)
    (output/"paired_implementation_speedups.json").write_text(json.dumps(export,indent=2)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    output=parser.parse_args().output.resolve()
    assert read(output/"run.json")["status"] in ("completed","completed_with_exclusions")
    destination=output/"figures"; destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.titlesize":9,"axes.labelsize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False,
        "svg.fonttype":"none","pdf.fonttype":42})
    manifest=[]
    new_timing_charts(output,manifest)
    rejection_chart(output,manifest)
    memory_chart(output,manifest)
    focused_existing(output,manifest)
    implementation_ratios(output,manifest)
    report=dict(plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                figures=manifest, total_figures=len(manifest),
                source_runs=["results/",output.relative_to(PAPER).as_posix()],
                note="Each comparison uses samples from its own measurement suite. Original figures and paper files are unchanged.")
    (output/"figure_manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (output/"sources"/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(f"Generated {len(manifest)} figures in PNG/SVG/PDF.")


if __name__=="__main__":
    main()
