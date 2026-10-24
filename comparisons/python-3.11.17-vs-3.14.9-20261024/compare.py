"""Compare frozen measurement sessions; never rerun solvers or edit the paper."""
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[1]
OLD = PAPER
NEW = PAPER / "collections/python-3.11.17-20261024"
OLD_EXT = PAPER / "extensions/python-3.14.9-20261024"
NEW_EXT = NEW / "extended"
COLORS = {"3.14.9":"#176ca4", "3.11.17":"#d17c26"}
SUITES = {
    "single": (OLD/"results/single_samples.csv", NEW/"results/single_samples.csv", 84),
    "batch": (OLD/"results/batch_samples.csv", NEW/"results/batch_samples.csv", 30),
    "formulations": (OLD/"results/formulations/samples.csv", NEW/"results/formulations/samples.csv", 50),
    "scaling": (OLD/"results/scaling/samples.csv", NEW/"results/scaling/samples.csv", 302),
    "extended": (OLD_EXT/"samples.csv", NEW_EXT/"samples.csv", 528),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def csv_save(name, values):
    with (OUT/name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protect():
    target = OUT/"protected_files.json"
    if not target.exists():
        files = {p.relative_to(PAPER).as_posix():digest(p) for p in sorted(PAPER.rglob("*"))
                 if p.is_file() and not any(x in p.relative_to(PAPER).parts
                 for x in ("comparisons", "tmp", "__pycache__", ".pytest_cache", ".git"))}
        save("protected_files.json", files)
    files = read(target)
    for path, expected in files.items():
        assert digest(PAPER/path) == expected, path
    return len(files)


def load_summary(path):
    data = rows(path)
    grouped = defaultdict(list)
    for row in data:
        grouped[row["case"],row["solver"]].append(row)
    summary_path = path.with_name(path.name.replace("samples", "summary"))
    summary = {(r["case"],r["solver"]):r for r in rows(summary_path)}
    assert grouped.keys() == summary.keys()
    for key, group in grouped.items():
        assert len(group) == 9 and {int(r["round"]) for r in group} == set(range(9))
        times = sorted(float(r["nanoseconds"]) for r in group)
        assert times[0] > 0 and all(math.isfinite(v) for v in times)
        for field, index in (("q25_ns",2), ("median_ns",4), ("q75_ns",6)):
            assert math.isclose(float(summary[key][field]), times[index], rel_tol=1e-12)
    return summary, data


def analyze():
    assert read(NEW/"audit.json")["status"] == read(OLD_EXT/"audit.json")["status"] == "passed"
    assert read(OLD/"results/collection_audit.json")["status"] == "passed"
    count = protect()
    environments=[read(root/"results/environment.json") for root in (OLD,NEW)]
    for key in ("cpu","logical_cpus","physical_cpus","seed","rounds","instances_per_single_case",
                "target_block_seconds","single_affinity","batch_affinity"):
        assert environments[0][key]==environments[1][key],key
    assert {k:v["version"] for k,v in environments[0]["packages"].items()} == \
           {k:v["version"] for k,v in environments[1]["packages"].items()}
    for name in ("run_experiments.py","compare_optional.py","benchmark_scaling.py"):
        assert digest(OLD/name)==digest(NEW/name)
    data, comparisons, reversals, winners, primary, overview = {}, [], [], [], [], {}
    pair_count = 0
    for suite, (old_path, new_path, expected) in SUITES.items():
        old, old_raw = load_summary(old_path)
        new, new_raw = load_summary(new_path)
        assert old.keys() == new.keys() and len(old) == expected
        schedule = lambda rr: {(r["case"],r["solver"],int(r["round"])):int(r["position"]) for r in rr}
        assert schedule(old_raw) == schedule(new_raw)
        data[suite] = old, new
        for key in sorted(old):
            a,b = old[key],new[key]
            comparisons.append(dict(suite=suite, case=key[0], solver=key[1],
                median_ns_314=float(a["median_ns"]), median_ns_311=float(b["median_ns"]),
                q25_ns_314=float(a["q25_ns"]), q75_ns_314=float(a["q75_ns"]),
                q25_ns_311=float(b["q25_ns"]), q75_ns_311=float(b["q75_ns"]),
                time_311_over_314=float(b["median_ns"])/float(a["median_ns"])))
        cases = sorted({k[0] for k in old})
        for case in cases:
            methods = sorted(k[1] for k in old if k[0] == case)
            win = [min(methods,key=lambda m:float(t[case,m]["median_ns"])) for t in (old,new)]
            winners.append(dict(suite=suite,case=case,winner_314=win[0],winner_311=win[1],changed=win[0]!=win[1]))
            for a,b in itertools.combinations(methods,2):
                pair_count += 1
                ratio = [float(t[case,a]["median_ns"])/float(t[case,b]["median_ns"]) for t in (old,new)]
                if (ratio[0]-1)*(ratio[1]-1) < 0:
                    reversals.append(dict(suite=suite,case=case,method_a=a,method_b=b,
                        a_over_b_314=ratio[0],a_over_b_311=ratio[1],
                        margin_over_5pct_in_both_runs=min(max(r,1/r) for r in ratio)>1.05))
            if suite != "batch":
                reference = "scipy_capped" if (case,"scipy_capped") in old else "scipy"
                for method in ("jvrust","jvcpp"):
                    ratio = [float(t[case,reference]["median_ns"])/float(t[case,method]["median_ns"]) for t in (old,new)]
                    primary.append(dict(suite=suite,case=case,solver=method,reference=reference,
                        reference_over_native_314=ratio[0],reference_over_native_311=ratio[1],
                        same_order=(ratio[0]-1)*(ratio[1]-1)>0))
        ratios = [r["time_311_over_314"] for r in comparisons if r["suite"]==suite]
        overview[suite] = dict(settings=len(old),cases=len(cases),timing_samples_per_run=len(old_raw),
            median_time_311_over_314=statistics.median(ratios),
            q25_time_ratio=float(np.percentile(ratios,25)),q75_time_ratio=float(np.percentile(ratios,75)),
            min_time_ratio=min(ratios),max_time_ratio=max(ratios),
            winner_changes=sum(r["changed"] for r in winners if r["suite"]==suite),
            pairwise_reversals=sum(r["suite"]==suite for r in reversals))
    assert len(comparisons)==994 and len(winners)==204 and len(primary)==402
    correctness = {}
    for suite,old_path,new_path in (
        ("formulations",OLD/"results/formulations/checks.json",NEW/"results/formulations/checks.json"),
        ("scaling",OLD/"results/scaling/checks.json",NEW/"results/scaling/checks.json"),
        ("extended",OLD_EXT/"checks.json",NEW_EXT/"checks.json"),
    ):
        aa,bb=read(old_path),read(new_path)
        a={(r["case"],r["solver"],r["instance"]):r for r in aa}
        b={(r["case"],r["solver"],r["instance"]):r for r in bb}
        assert len(a)==len(aa) and len(b)==len(bb) and a.keys()==b.keys()
        error=max(abs(a[k]["objective"]-b[k]["objective"]) for k in a)
        changes=sum(a[k]["pairs"]!=b[k]["pairs"] for k in a)
        assert error==0 and changes==0
        correctness[suite]=dict(records=len(a),maximum_objective_difference=error,cardinality_changes=changes)
    assert read(OLD_EXT/"edge_validation.json")==read(NEW_EXT/"edge_validation.json")
    assert read(OLD_EXT/"plan.json")==read(NEW_EXT/"plan.json")
    pairing={}
    for suite,old_path,new_path in (
        ("formulations",OLD/"results/formulations/datasets.json",NEW/"results/formulations/datasets.json"),
        ("scaling",OLD/"results/scaling/datasets.json",NEW/"results/scaling/datasets.json"),
        ("extended",OLD_EXT/"datasets.json",NEW_EXT/"datasets.json"),
    ):
        extract=lambda rr:{r["case"]:([v["sha256"] for v in r["inputs"]] if "inputs" in r else r["input_sha256"]) for r in rr}
        a,b=extract(read(old_path)),extract(read(new_path))
        assert a==b
        pairing[suite]=sum(len(v) for v in a.values())
    memory=[]
    memdata=[]
    for root in (OLD_EXT,NEW_EXT):
        raw=read(root/"memory_samples.json")
        summary={(r["case"],r["solver"]):r for r in rows(root/"memory_summary.csv")}
        for key,r in summary.items():
            values=sorted(v["additional_peak"] for v in raw if (v["case"],v["solver"])==key)
            assert len(values)==5
            for field,i in (("q25_bytes",1),("median_bytes",2),("q75_bytes",3)):
                assert float(r[field])==values[i]
        memdata.append((raw,summary))
    a,b=[{r["job"]:r["input_sha256"] for r in raw} for raw,_ in memdata]
    assert a==b and len(a)==405
    pairing["memory_trials"]=len(a)
    batch_hashes=[[r["input_sha256"] for r in read(root/"results/batch_validation.json")["checks"]] for root in (OLD,NEW)]
    assert batch_hashes[0]==batch_hashes[1]
    pairing["batches"]=len(batch_hashes[0])
    ma,mb=[t for _,t in memdata]
    assert ma.keys()==mb.keys() and len(ma)==81
    for key in ma:
        a,b=ma[key],mb[key]
        assert a["transformed_dense_buffer_bytes"]==b["transformed_dense_buffer_bytes"]
        memory.append(dict(case=key[0],solver=key[1],n=int(a["n"]),m=int(a["m"]),regime=a["regime"],
            median_mib_314=float(a["median_bytes"])/2**20,median_mib_311=float(b["median_bytes"])/2**20,
            q25_mib_314=float(a["q25_bytes"])/2**20,q75_mib_314=float(a["q75_bytes"])/2**20,
            q25_mib_311=float(b["q25_bytes"])/2**20,q75_mib_311=float(b["q75_bytes"])/2**20,
            difference_mib=(float(b["median_bytes"])-float(a["median_bytes"]))/2**20))
    primary_overview={}
    for solver in ("jvrust","jvcpp"):
        primary_overview[solver]={}
        for suite in ("single","formulations","scaling","extended"):
            rr=[r for r in primary if r["solver"]==solver and r["suite"]==suite]
            primary_overview[solver][suite]=dict(cases=len(rr),wins_314=sum(r["reference_over_native_314"]>1 for r in rr),
                wins_311=sum(r["reference_over_native_311"]>1 for r in rr),ordering_changes=sum(not r["same_order"] for r in rr))
    report=dict(status="passed",created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="Read-only comparison of the completed 3.14.9 and 3.11.17 collections; no manuscript update.",
        environments=[{k:e[k] for k in ("python","numpy","scipy","cpu","single_affinity","batch_affinity")}
                      for e in environments],
        suites=overview,settings=994,cases=204,primary_comparisons=primary_overview,
        pairwise_comparisons=pair_count,pairwise_reversals=len(reversals),winner_changes=sum(r["changed"] for r in winners),
        reversal_margin_sensitivity={str(m):sum(min(max(r[k],1/r[k]) for k in ("a_over_b_314","a_over_b_311"))>1+m for r in reversals) for m in (0,.02,.05,.1)},
        correctness=correctness,edge_checks_identical=129,input_hashes_identical=pairing,
        memory_conditions=81,max_absolute_memory_median_difference_mib=max(abs(r["difference_mib"]) for r in memory),
        methods="Within-run ratios divide reported median times. Cross-run ratios divide the 3.11 median by the 3.14 median. Round numbers are not treated as paired observations across sessions. IQRs are descriptive, not confidence intervals. The 5% ranking margin is a post-hoc descriptive filter; 0/2/5/10% sensitivity is reported.",
        limitations="One session per environment on one host; the 204 cases are a designed grid, not an independent application sample. NumPy, SciPy, native wheels, and session state also differ. No causal interpreter-speed claim or population significance test.",
        previous_files_unchanged=count,source_script_sha256=digest(Path(__file__)))
    save("comparison.json",report)
    csv_save("timing_comparison.csv",comparisons)
    csv_save("primary_orderings.csv",primary)
    csv_save("pairwise_reversals.csv",reversals)
    csv_save("case_winners.csv",winners)
    csv_save("memory_comparison.csv",memory)
    metrics=[]
    for label,suite,case,numerator,denominator in (
        ("JVRUST advantage over capped SciPy, 128x160 L=.5","formulations","optional128x160","scipy_capped","jvrust"),
        ("Capped SciPy advantage on formulation congestion","formulations","optional_blocks256x320","jvrust","scipy_capped"),
        ("Capped SciPy advantage on 1024x1280 congestion","scaling","optional_blocks_1024x1280","jvrust","scipy_capped"),
        ("JVRUST small-batch loop / one worker","batch","batch1024x8x10","jvrust_loop","jvrust_batch_t1"),
        ("JVRUST small-batch loop / four workers","batch","batch1024x8x10","jvrust_loop","jvrust_batch_t4"),
        ("JVCPP / JVRUST four-worker time, 64x64x80 batch","batch","batch64x64x80","jvcpp_batch_t4","jvrust_batch_t4"),
    ):
        values=[float(t[case,numerator]["median_ns"])/float(t[case,denominator]["median_ns"]) for t in data[suite]]
        metrics.append(dict(metric=label,unit="ratio of medians",python_314=values[0],python_311=values[1]))
    csv_save("key_metrics.csv",metrics)
    return data,comparisons,primary,memory,report


def finish(fig,stem,title,note):
    fig.suptitle(title,fontsize=14,y=.99)
    fig.text(.015,.012,note,ha="left",va="bottom",fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.94))
    for suffix in ("png","svg"):
        fig.savefig(OUT/"figures"/f"{stem}.{suffix}",dpi=170,bbox_inches="tight")
    plt.close(fig)


def plots(data,comparisons,primary,memory):
    (OUT/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"svg.fonttype":"none",
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.7))
    palette={"single":"#78519a","formulations":"#d17c26","scaling":"#32905d","extended":"#176ca4"}
    for suite,color in palette.items():
        rr=[r for r in primary if r["suite"]==suite and r["solver"]=="jvrust"]
        axes[0].scatter([r["reference_over_native_314"] for r in rr],[r["reference_over_native_311"] for r in rr],
                        label=f"{suite} ({len(rr)})",color=color,s=20,alpha=.7)
    axes[0].set_xscale("log");axes[0].set_yscale("log")
    axes[0].plot([.05,200],[.05,200],color="#777777",ls=":",lw=1)
    axes[0].set_xlim(.05,200);axes[0].set_ylim(.05,200)
    axes[0].axhline(1,color="#333333",lw=.8);axes[0].axvline(1,color="#333333",lw=.8)
    axes[0].set(xlabel="SciPy / JVRUST time, Python 3.14.9",ylabel="SciPy / JVRUST time, Python 3.11.17",title="Same ordering in 201 / 201 cases")
    axes[0].legend(fontsize=7,loc="upper left",frameon=False)
    sizes=[16,32,64,128,256,512,1024]
    for version,table,style in zip(COLORS,data["scaling"],("-","--")):
        vals=[float(table[f"optional_blocks_{n}x{5*n//4}","scipy_capped"]["median_ns"])/float(table[f"optional_blocks_{n}x{5*n//4}","jvrust"]["median_ns"]) for n in sizes]
        axes[1].plot(sizes,vals,marker="o",ms=4,color=COLORS[version],ls=style,label=version)
    axes[1].set_xscale("log",base=2);axes[1].set_yscale("log")
    axes[1].set_xticks(sizes,[str(n) for n in sizes]);axes[1].tick_params(axis="x",labelsize=7)
    axes[1].axhline(1,color="#333333",ls=":",lw=1)
    axes[1].set(xlabel="Smaller dimension N; M = 5N/4",ylabel="Capped SciPy / JVRUST time",title="Same congestion crossover")
    axes[1].legend(title="Python",frameon=False,fontsize=8)
    cases=["batch64x64x80","batch16x128x160_optional","batch1024x8x10"]
    for vi,(version,table) in enumerate(zip(COLORS,data["batch"])):
        for wi,workers in enumerate((1,4)):
            values=[float(table[c,"jvrust_loop"]["median_ns"])/float(table[c,f"jvrust_batch_t{workers}"]["median_ns"]) for c in cases]
            axes[2].bar(np.arange(3)+(vi*2+wi-1.5)*.19,values,width=.18,color=COLORS[version],alpha=.65 if workers==1 else 1,
                hatch="//" if workers==1 else None,label=f"{version}, {workers} worker"+("s" if workers==4 else ""))
    axes[2].axhline(1,color="#333333",ls=":",lw=1)
    axes[2].set_xticks(range(3),["64 problems\n64 x 80","16 problems\n128 x 160, L=.5","1024 problems\n8 x 10"],fontsize=8)
    axes[2].set(ylabel="JVRUST Python loop / native batch time",title="Batching benefit remains")
    axes[2].legend(fontsize=7,frameon=False)
    for ax in axes: ax.grid(alpha=.15);ax.set_axisbelow(True)
    finish(fig,"principal_findings","The principal performance findings repeat across environments",
        "Ratios of within-run medians; >1 favors JVRUST/native batching. SciPy uses the suite's matched formulation (capped where available).\nSame inputs and method order; separate sessions. NumPy, SciPy and native wheels differ between environments.")
    fig,ax=plt.subplots(figsize=(10,4.8))
    rng=np.random.default_rng(24)
    for index,suite in enumerate(SUITES):
        values=[r["time_311_over_314"] for r in comparisons if r["suite"]==suite]
        ax.scatter(values,index+rng.uniform(-.23,.23,len(values)),s=12,alpha=.35,color="#176ca4")
        med=statistics.median(values);ax.scatter([med],[index],marker="D",s=55,color="#a63a26",zorder=5)
        ax.text(1.345,index,f"median {med:.3f}",fontsize=9,va="center")
    ax.axvline(1,color="#333333",ls=":")
    ax.set_yticks(range(5),[f"{s} (n={SUITES[s][2]})" for s in SUITES]);ax.invert_yaxis()
    ax.set(xlim=(.63,1.50),xlabel="Time in Python 3.11.17 / time in Python 3.14.9",title="<1: shorter time in 3.11; >1: shorter time in 3.14")
    ax.grid(axis="x",alpha=.2)
    finish(fig,"runtime_shifts","Absolute runtimes shift by suite; there is no universal environment speedup",
        "All 994 method/case settings shown. Each dot compares two medians of nine rounds; diamonds are medians across settings.\nThis is a descriptive workload grid, not a causal interpreter comparison or a population confidence interval.")
    fig,axes=plt.subplots(1,2,figsize=(10,4.7))
    specs=[("scaling","required_square_32x32",("jvc","scipy"),1e3,"Microseconds per matrix","Small-square baseline ordering changes"),
           ("batch","batch64x64x80",("jvrust_batch_t4","jvcpp_batch_t4"),1e6,"Milliseconds per complete batch","Four-worker batch difference is unstable")]
    for ax,(suite,case,methods,unit,ylabel,title) in zip(axes,specs):
        for vi,(version,table) in enumerate(zip(COLORS,data[suite])):
            r=[table[case,m] for m in methods]
            med=np.array([float(x["median_ns"])/unit for x in r]);lo=np.array([float(x["q25_ns"])/unit for x in r]);hi=np.array([float(x["q75_ns"])/unit for x in r])
            positions=np.arange(2)+(vi-.5)*.32
            bars=ax.bar(positions,med,width=.30,color=COLORS[version],label=version)
            ax.errorbar(positions,med,yerr=(med-lo,hi-med),fmt="none",color="#333333",capsize=3,lw=1)
            ax.bar_label(bars,fmt="%.3f",padding=5,fontsize=8)
        ax.set_xticks(range(2),["JVC","SciPy"] if suite=="scaling" else ["JVRUST","JVCPP"])
        ax.set(ylabel=ylabel,title=title);ax.set_ylim(0,ax.get_ylim()[1]*1.14);ax.legend(title="Python",frameon=False)
        ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    finish(fig,"ranking_exceptions","Important qualifications to the consistency finding",
        "Left: required 32 x 32 scaling case. Right: 64 problems of size 64 x 80, four workers.\nBars are medians, whiskers are timing-round IQRs (not confidence intervals). Full rank-change lists accompany the report.")
    fig=plt.figure(figsize=(11,8));grid=fig.add_gridspec(2,2)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    methods=["jvrust","jvcpp","scipy_capped","scipy_dummies","scipy_square"]
    for ax,shape in zip(axes[:2],((256,4096),(4096,256))):
        rr={r["solver"]:r for r in memory if (r["n"],r["m"])==shape and r["regime"]=="optional"}
        for vi,version in enumerate(COLORS):
            suffix="314" if vi==0 else "311"
            med=np.array([rr[m][f"median_mib_{suffix}"] for m in methods])
            lo=np.array([rr[m][f"q25_mib_{suffix}"] for m in methods]);hi=np.array([rr[m][f"q75_mib_{suffix}"] for m in methods])
            pos=np.arange(5)+(vi-.5)*.34
            ax.bar(pos,med,width=.32,color=COLORS[version],label=version)
            ax.errorbar(pos,med,yerr=(med-lo,hi-med),fmt="none",color="#333333",capsize=2,lw=.8)
        ax.set_yscale("symlog",linthresh=.25);ax.set_ylim(0,260)
        ax.set_xticks(range(5),["JVRUST","JVCPP","SciPy\ncapped","SciPy\ndummies","SciPy\nsquare"],fontsize=8)
        ax.set(ylabel="Additional peak working set (MiB)",title=f"{shape[0]} x {shape[1]}, finite limit L = 0.5")
        ax.legend(title="Python",frameon=False,fontsize=8);ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    plan=read(OLD_EXT/"plan.json")["timing"];checks=read(OLD_EXT/"checks.json")
    for size,color in ((128,"#176ca4"),(256,"#d17c26")):
        for distribution,style in (("uniform","-"),("blocks","--")):
            selected=sorted([s for s in plan if s["family"]=="rejection" and s["n"]==size and s["distribution"]==distribution],key=lambda s:s["limit"])
            values=[100*(1-statistics.mean(r["pairs"] for r in checks if r["case"]==s["case"] and r["solver"]=="jvrust")/size) for s in selected]
            axes[2].plot([s["limit"] for s in selected],values,color=color,ls=style,marker="o",ms=3,
                         label=f"{size} x {5*size//4}, "+("uniform" if distribution=="uniform" else "congested"))
    axes[2].set_xscale("symlog",linthresh=.001);axes[2].set_xlim(-.00008,4.5)
    ticks=[0,.001,.01,.1,1,4];axes[2].set_xticks(ticks,[f"{t:g}" for t in ticks]);axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].set(xlabel="Rejection limit L",ylabel="Smaller side unmatched (%)",title="Identical rejection profiles in both runs (curves drawn once)")
    axes[2].legend(ncol=2,frameon=False,fontsize=8);axes[2].grid(alpha=.2)
    finish(fig,"memory_and_rejection","Memory structure and realized rejection are consistent",
        "Memory: five fresh processes per condition; median and IQR. High-water increments are not allocation counts; tall inputs have larger measured increments.\nRejection profiles match exactly by method and input across environments. Four inputs per setting; other optimal pair identities are not compared.")


if __name__=="__main__":
    values=analyze()
    plots(*values[:4])
    assert protect()==values[4]["previous_files_unchanged"]
    save("artifact_manifest.json",{p.relative_to(OUT).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p))
         for p in sorted(OUT.rglob("*")) if p.is_file() and p.name!="artifact_manifest.json"})
    print(json.dumps(values[4],indent=2))
