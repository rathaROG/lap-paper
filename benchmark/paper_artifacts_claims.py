"""Build a source-indexed ledger of quantitative manuscript paragraphs and tables.

This is an editorial index, not a claim that text can be verified by a numeric
regular expression. Sources and derivations below were mapped after reading the
manuscript, generators, saved observations and pinned implementation sources.
Regenerate after manuscript edits to refresh line numbers and exact excerpts.
"""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import median

try:
    from .paper_artifacts_sensitivity import LIGHT, FULL
except ImportError:
    from paper_artifacts_sensitivity import LIGHT, FULL


NUMBER = re.compile(r"\d|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|all|every|none)\b", re.I)
SKIP = ("%", r"\begin", r"\end", r"\centering", r"\includegraphics", r"\label", r"\input", r"\Float", r"\clearpage", r"\bibliograph", r"\renewcommand", r"\setlength", r"\toprule", r"\midrule", r"\bottomrule", r"\addlinespace", r"\tag", r"\section", r"\subsection", r"\appendix", r"\noindent")


def audit_empirical_claims(repo, roots):
    """Independently recompute the manuscript's core numerical comparisons."""
    read=lambda p:json.loads(p.read_text(encoding="utf-8"))
    classify=lambda x:"loss" if x<.97 else "win" if x>1.03 else "tie"
    count=lambda values:{k:v for k,v in sorted(Counter(classify(x) for x in values).items())}
    results={}
    for environment,relative in roots.items():
        root=repo/relative
        summary=read(root/"campaign-summary.json")
        manifest=read(root/"campaign-session/manifest.json")
        cases={c["id"]:c for c in manifest["cases"]}
        estimates={(e["case_id"],e["method"]):e for e in summary["estimates"] if e["status"]=="complete"}
        paired=summary["paired"]
        out=dict(source_root=relative,manifest_counts=manifest["counts"],native={},sparse={},bypass_medians={},geometry_first_csgraph_win={},orientation={})
        for method in ("jvrust","jvcpp","jvhybrid"):
            for objective in ("required","optional"):
                values=[p["ratio"] for p in paired if p["method"]==method and cases[p["case_id"]]["kind"]=="single" and cases[p["case_id"]]["objective"]==objective and p["reference"]==cases[p["case_id"]]["metadata"]["primary_comparator"]]
                if values:out["native"][objective+":"+method]=dict(n=len(values),classifications=count(values),median_ratio=median(values))
        out["hard_optional_congestion_ratios"]={method:next(p["ratio"] for p in paired if p["method"]==method and p["case_id"]=="extended:041_congestion_optional_256x320") for method in ("jvrust","jvcpp")}
        out["hard_optional_congestion_native_extra_latency_percent"]={method:100*(1/value-1) for method,value in out["hard_optional_congestion_ratios"].items()}
        for language in ("rust","cpp"):
            values=[1/p["ratio"] for p in paired if p["method"]==f"policy:{language}:no_reductions"]
            out["bypass_medians"][language]=dict(n=len(values),median_treatment_over_reference=median(values))
        for method in ("sparse:csgraph","sparse:lapmod"):
            points=[p for p in paired if p["method"]==method]
            out["sparse"][method]=dict(n=len(points),vs_capped=count(p["ratio"] for p in points),vs_rust=count(estimates[p["case_id"],"jvrust"]["seconds"]/p["seconds"] for p in points))
        for q,motion in ((.9,.25),(.6,1)):
            points=[]
            for p in paired:
                c=cases[p["case_id"]]
                if p["method"]=="sparse:csgraph" and c["family"]=="geometry" and c["rows"]==c["cols"] and c["metadata"]["q"]==q and c["metadata"]["motion"]==motion and c["metadata"]["limit_over_delta"]==1.5 and c["metadata"]["gate_over_delta"] is None and p["ratio"]>1.03:
                    points.append((c["rows"],p["ratio"]))
            out["geometry_first_csgraph_win"][f"q{q}:motion{motion}"]=min(points)
        wide=next(c for c in cases.values() if c["kind"]=="single" and c["rows"]==64 and c["cols"]==16384 and c["objective"]=="optional")
        tall=next(c for c in cases.values() if c["kind"]=="single" and c["rows"]==16384 and c["cols"]==64 and c["objective"]=="optional")
        for method in ("jvrust","jvcpp","sparse:csgraph","sparse:lapmod"):
            out["orientation"][method]=dict(wide_case=wide["id"],tall_case=tall["id"],tall_over_wide=estimates[tall["id"],method]["seconds"]/estimates[wide["id"],method]["seconds"])
        conditions=summary["memory"]["conditions"]
        trials=[t for c in conditions for t in c["trials"]]
        out["memory"]=dict(conditions=len(conditions),trials=len(trials),zero_condition_medians=sum(median(t["additional_peak"] for t in c["trials"])==0 for c in conditions),zero_trials=sum(t["additional_peak"]==0 for t in trials),zero_peak_and_resident_growth_above_1MiB=sum(t["additional_peak"]==0 and t["after"]["rss"]-t["before"]["rss"]>2**20 for t in trials))
        with (root/"application-session/tracking_summary.csv").open(encoding="utf-8",newline="") as f:tracking=list(csv.DictReader(f))
        out["native_tracker_classifications"]={method:count(float(x["scipy_speedup"]) for x in tracking if x["solver"]==method) for method in ("laprust","lapcpp","lapjvh")}
        if environment.startswith("W"):
            with (root/"lsape-summary/all_cases.csv").open(encoding="utf-8",newline="") as f:lsape=list(csv.DictReader(f))
            out["lsape_classifications"]={contrast:count(float(x["median_estimate"]) for x in lsape if x["comparison"]==contrast) for contrast in ("original_over_padded","original_over_one_sided","one_sided_over_padded","padded_over_compact","scan_over_compact")}
        results[environment]=out
    return results


def generate(repo: Path) -> dict:
    repo = Path(repo).resolve()
    directory = repo / "benchmark/paper-artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    roots = {e: "benchmark/runs/" + path for e, path in LIGHT.items()}
    evidence = {}

    def source(key, paths, fields, derivation, kind="saved_measurements", artifact=None):
        evidence[key] = dict(source_files=paths, fields_or_selection=fields,
                             derivation=derivation, kind=kind,
                             generator_output=artifact)

    def all_paths(suffix):
        return [p + suffix for p in roots.values()]

    campaign_sources = "/campaign-session/sources/benchmark/campaign/"
    source("timed_adapters", all_paths(campaign_sources + "adapters.py") + all_paths(campaign_sources + "worker.py") + all_paths("/campaign-session/source-hashes.json"),
           "adapters._native_solver, prepare_method and _decode; worker.phase_call; source-hashes entries for adapters.py and worker.py",
           "The timed callable includes output decoding. For optional calls, _decode selects pairs against the original matrix, requiring finite cost, cost <= gate when supplied, and cost < limit. Native IDs jvrust, jvhybrid and jvcpp select laprust.lapjv, lap.lapjvh and lapcpp.lapjv with float64 precision, pair output and return_cost=False.",
           kind="pinned_campaign_source")
    source("protocol", all_paths("/campaign-session/manifest.json") + all_paths("/campaign-session/configuration.json") + all_paths("/campaign-summary.json"),
           "manifest.counts and cases[]; configuration.{inputs,rounds,target_seconds,memory_trials,session_id}; summary.{coverage,malformed_records,unobserved_planned_conditions}",
           "Read saved coverage and settings separately for each environment; planned_case_method_jobs minus memory_conditions gives timing conditions.", artifact="T1_protocol_inventory.json")
    source("environment", all_paths("/campaign-session/environment.json") + all_paths("/campaign-session/configuration.json") + all_paths("/status.json"),
           "environment.{packages,python,system_settings,topology}; configuration.{single_cpu,batch_cpus,supervisor_cpus}; status.{started_utc,finished_utc,wall_seconds}",
           "Read recorded versions and CPU model; join each selected logical CPU to core_id and highest efficiency_class; wall_seconds/60 for minutes.", artifact="s1_light_protocol_audit.json")
    source("native", all_paths("/campaign-summary.json") + all_paths("/campaign-session/manifest.json"),
           "paired[]: complete canonical jvrust/jvcpp/jvhybrid, case.kind=single, reference=case.metadata.primary_comparator; split by environment/objective",
           "Reference/method ratio: classify <0.97 loss, [0.97,1.03] tie, >1.03 win. Required count=102 per method; optional count=166 for Rust/CPP. Compute case-median ratios within each environment/objective; JVRUST median ranges round to3.67-4.66 required and4.72-5.23 optional. No environment pooling.", artifact="campaign_summary.json:environments.*.native; F1_overview_ccdf.csv; S_native_family_counts.csv; S_native_exceptions.csv")
    source("geometry", all_paths("/campaign-session/manifest.json") + all_paths("/campaign-summary.json"),
           "cases[family=geometry].{rows,cols,objective,metadata.q,metadata.motion,metadata.delta,metadata.limit_over_delta,metadata.gate_over_delta,methods}; estimates[].seconds; paired[].ratio",
           "Main square sweep selects (q,motion)=(.9,.25),required or optional L/delta=1.5 with no gate;20 cases/environment. Companion panels retain (.6,1) and5:4. Fixed256x320 heatmap retains all36settings,4environments,2methods=288cells. Methods absent above1024rows remain not planned.", artifact="chart_points.csv:geometry/A2; F3_geometry.csv; F3_geometry_5to4_companion.csv; A2_geometry_sensitivity.csv; A2_geometry_panel.csv")
    source("layout", all_paths("/campaign-summary.json") + all_paths("/campaign-session/manifest.json"),
           "cases[].metadata.layout_panel; estimates[method starts cost: or shape:]; canonical formulation panel; primary comparator from manifest",
           "All624 layout conditions per environment retained; ratio matching unpadded SciPy / representation. Constructor-only observations have separate boundary.", artifact="chart_points.csv:representations; F4_representations.csv; S3_layout_timing.csv; S3_layout_construction.csv; S3_layout_coverage.csv")
    source("policy", all_paths("/campaign-summary.json") + all_paths("/campaign-session/build-index.json"),
           "paired[method starts policy:].{case_id,method,reference,ratio,attempt_id,input_hashes}; build artifacts and policy hashes",
           "Invert saved reference/treatment ratios to treatment/reference. Forest markers summarize eligible cases separately within environment/language: median,IQR,min,max; descriptive workload spread,not uncertainty. Supplement connects ordered allowances1,2,4,8 only; nominal switches are categorical.", artifact="F4_policy_forest_summary.csv; F5_budget_data.csv; F5_nominal_policy_data.csv")
    source("counters", all_paths("/campaign-diagnostics/summary.json"),
           "Each summary.jobs[].directory -> attempts/<basename>/diagnostics.jsonl; study=native_counters; policy methods; counters.{initialization_row_scans,search_row_scans,initialization_displacements,initialization_chain_starts,augmentations,rejection_endings}; input_index,input_sha256",
           "Verify all six event vectors across all environments/languages per input; each count uses counted_calls=problems=1. Anchor values are means over five inputs. Optional base reference scans512/0 and bypass0/700.2; augmentations0->256.", artifact="S3_native_counters.csv.gz; T3_counter_anchors.csv")
    source("sparse", all_paths("/campaign-summary.json"),
           "estimates: methods sparse:csgraph,sparse:lapmod,jvrust,scipy:capped; same case_id/environment/attempt/input identities",
           "x=scipy:capped.seconds/sparse.seconds; y=jvrust.seconds/sparse.seconds; all166 optional cases per method/environment; practical classes on displayed ratios.", artifact="F5_sparse_scatter.csv; campaign_summary.json:environments.*.sparse")
    source("structure", all_paths("/campaign-session/manifest.json") + all_paths("/campaign-summary.json") + all_paths(campaign_sources + "datasets.py"),
           "cases[family in congestion,forbidden,ties].input_group.parameters; primary canonical paired ratios; summary.attempts[].{case_id,input_hashes}; datasets._extended_cost",
           "26 cases;260 native ratio points across environments. Separate family/objective/limit/method/environment curves; varying declared structure parameter only. With levels=1, floor(U)=0 for every U in [0,1). Each recorded attempt for extended:050_ties_required_256x320 and extended:051_ties_optional_256x320 has five input indices with the same digest.", artifact="chart_points.csv:structure; F7_structure_lines_data.csv; F7_structure_lines_assertions.json")
    source("batch", all_paths("/campaign-summary.json"),
           "batch_contrasts[].{case_id,contrast,method,reference,ratio,ratio_definition,seconds,reference_seconds}; estimates[].round_means_seconds",
           "Use saved ratio for worker/dispatch; invert solver-state and orientation lifetime contrasts for recreation/retained direction. Worker-count contrasts use matched inputs in distinct placement-process attempts; within-placement contrasts use the same attempt. Total-worker lifetime contrast is supplement-only; no pooling.", artifact="F6_batch_dot_values.csv; S5_batch_contrasts.csv; S5_batch_timing.csv; campaign_summary.json:environments.*.batch")
    source("orientation", all_paths("/campaign-summary.json") + all_paths("/campaign-session/manifest.json") + all_paths(campaign_sources + "datasets.py") + all_paths(campaign_sources + "adapters.py"),
           "large_rectangle/extreme_rectangle pairs share input_group,objective,limit,gate; reversed dimensions and transpose flags; estimates[].seconds from separate attempts; adapters._sparse_solver uses original input dimensions n,m",
           "tall.seconds/wide.seconds;12 case pairs(6 shapes×2 objectives),74 method pairs/environment. For optional 64×16384, the four ratios span 60.51–70.54 for CSGRAPH and 1.011–1.065 for LAPMOD. CSGRAPH adds one rejection column per original row: its n×(n+m) graph changes from 64×16448 to 16384×16448. LAPMOD uses (n+m)² in both orientations. Export preserves both attempts and their input identities.", artifact="F9_orientation_dot_values.csv; F9_orientation_not_planned.csv; F12_orientation_pairs.csv")
    source("memory", all_paths("/campaign-summary.json") + all_paths("/campaign-session/manifest.json"),
           "memory.conditions[].{case_id,method,counter,trials}; trials[].{additional_peak,baseline_peak_headroom,before.rss,after.rss,before.peak,after.peak}",
           "Bytes/2^20; medians/ranges over five trials. Count zero condition medians84/84/18/17; zero trials420/420/88/82; zero increment and RSS growth>2^20 count80/80/0/0.91×5=455 trials/environment. Extended endpoint selection memory:extended:*:256x4096 and transposes.", artifact="S5_memory_conditions.csv; S5_memory_trials.csv; T4_memory_endpoints.csv; support_summary.json:memory")
    source("applications", all_paths("/application-session/protocol.json") + all_paths("/application-session/replay_summary.csv") + all_paths("/application-session/tracking_summary.csv") + all_paths("/application-session/profiles"),
           "protocol.{frames_per_clip,pipeline_configurations,solvers,rounds,minimum_replay_block_seconds,warmup_frames,cpu,replay_precisions}; summaries[].{case,solver,precision,scipy_speedup,median_*}; separate SciPy profiles retain assignment fractions of tracker-update time",
           "Join same case/API/environment. F7 float32 only:7 nonreference APIs×8 pipelines×4 environments=224. Main envelopes are observed min/max across12 native or16 legacy API/environment pairs,not averages. Classify each native API's eight update ratios separately:4 wins4 ties. Float64 replay remains in S4.", artifact="F7_application_envelopes.csv; F7_applications_api.csv; F9_applications.csv; S4_replay.csv; S4_tracking.csv; support_summary.json:applications")
    source("accuracy", all_paths("/application-session/tracking_summary.csv") + all_paths("/application-validation/validation.json"),
           "tracking_summary per pipeline/solver metrics and prediction hashes; validation.{matrices,checks,results[].objective_nonzero_gap_count,objective_max_abs_gap,objective_mismatch_count}",
           "7,539 matrices×2 precision views×9 validated entries=135,702 checks; all recorded objective gaps zero. Metric deltas subtract SciPy,percentage scores multiply by100; compare prediction hashes independently. Exact same deltas validated across environments before table deduplication.", artifact="S4_accuracy.csv; S4_validation.csv; support_summary.json:applications")
    source("microscopy", ["benchmark/cell_tracking/observations.json", "benchmark/cell_tracking/manifest.json", "benchmark/cell_tracking/benchmark.py", "benchmark/cell_tracking/provenance.json"],
           "datasets[].{frames,input,rounds,profiles,validation}; versions; seed; measured_utc; source-frame windows and recorded track/cost digests",
           "Nine excerpts,816frames,807nonempty calls per arm. Whole-linker means per frame:median of six rounds; ratios divide arm medians. Profiles:one separate instrumented run per arm/excerpt; aggregate seconds=sum(frames*association_ns_per_frame)/1e9. Adapter,solver,pair conversion included; no pure-kernel or exact decomposition claim. All18 recorded cross-arm cost/track comparisons agree and recorded objective gaps are zero; original matrices/tracks are unavailable.", artifact="S7_microscopy.csv; S7_microscopy_rounds.csv; microscopy_summary.json")
    source("workloads", all_paths("/application-session/protocol.json") + all_paths("/application-session/profiles"),
           "profiles/<case>__scipy.json.matrix_metadata[].{shape,frame,call_index,source_dtype,solver_input_dtype,key}; protocol.frames_per_clip",
           "Count matrices and min/max original dimensions per pipeline; verify equality across environments then show once. Captured total7,539; eight pipelines999frames each.", artifact="A_application_workloads.csv; S4_capture_shapes.csv")
    lsape_reports = [f"{roots[env]}/lsape-session-{session}/report.json"
                     for env in ("W311", "W314") for session in (1, 2, 3)]
    source("lsape", [roots[e] + "/lsape-summary/all_cases.csv" for e in ("W311","W314")] + lsape_reports + ["benchmark/lsape/README.md", "benchmark/lsape/data/rectangular-grid-v1.json", "benchmark/lsape/rectangular_campaign.py", "benchmark/lsape/lsape_bridge.cpp", "benchmark/lsape/src/specialized/hungarian-lsape-uniform.hh"],
           "all_cases[comparison in original_over_padded,original_over_one_sided,one_sided_over_padded,padded_over_compact,scan_over_compact].{median_estimate,min_session,max_session,s1_ratio,s2_ratio,s3_ratio}; reports: session_index, manifest.target_block_ms, cases[].{case,input_identities,round_order,methods}; bridge.cpp:132–195 floating-point mode,dimensions and magnitude guards; specialized header's guarded-rejection dispatch",
           "For each interpreter and case, the five arms share four inputs per process. The union of input_identities[].input_sha256 across the three processes has size 12 for every case. Each process shuffles arm order in each of nine rounds, with a 30 ms calibration target. 67 cases × 5 contrasts × 2 interpreters = 670 displayed points; each center is the median of three process paired-log estimates; full export retains process min/max. O/P,O/U,U/P,P/T,S/T definitions; do not multiply separately summarized medians. Full buffer(n+1)(m+1) minus compact nm=n+m+1 doubles.", artifact="F8_lsape_values.csv; F8_lsape_family_counts.csv; S6_lsape_comparisons.csv; S6_lsape_process_estimates.csv; S6_lsape_identity.json; support_summary.json:lsape")
    source("validation", all_paths("/campaign-validation.json") + all_paths("/application-validation/validation.json"),
           "campaign.methods[].{passed,failures,numerical_boundary_limitations}; methods[].results[].{case,status,exact_actual,exact_optimum,exact_gap}; campaign.correctness_failed; application.results[]",
           "Per environment101 methods;4195 passed top-level fixtures,2864 unsupported fixture/method pairs,11 numerical boundary limitations across7layouts,0 blocking failures. All 11 saved exact gaps are 1/9007199254740992 = 2^-53. Tolerance validation does not establish universal exactness.", artifact="s1_light_protocol_audit.json; s1_numerical_boundary_records.json; S4_validation.csv")
    source("binary64_example", ["latex/main.tex", "benchmark/campaign/adapters.py"],
           "Worked example in Limitations and threats to validity; prepare_method for jvrust, jvhybrid, jvcpp and scipy:capped; single optional case with rows=cols=2, limit=1, no gate, and binary64 costs [[0,1/2],[1/2-2^-54,1]].",
           "In a separate solver check all four adapters returned just (0,0), with penalized objective 1. The off-diagonal objective is 1-2^-54; the empty matching costs 2, every singleton costs at least 1 and the diagonal costs 1. The off-diagonal value lies halfway between 1 and the preceding binary64 value, so ties-to-even rounds it to 1. This example is outside the frozen validation counts.",
           kind="standalone_solver_check_and_exact_derivation")
    source("sensitivity", ["benchmark/runs/" + FULL + "/campaign-summary.json", "benchmark/runs/" + FULL + "/campaign-session/manifest.json", "benchmark/runs/" + FULL + "/application-session/tracking_samples.json", "benchmark/runs/" + FULL + "/application-session/replay"],
           "FULL estimates[].round_means_seconds,paired canonical native comparisons; applications samples[].{solver,round,nanoseconds}; manifests.counts",
           "Recompute median first3 versus5 assignment round means; all10 subsets;818 comparisons,0first-subset changes,max2. Apps ratio of medians first3 versus9; max absolute relative percent change.7106409294 tracking,.3768586954 replay. FULL374/340/192 versus light302/268/120 total/single/geometry. FULL timings supplementary only.", artifact="s1_sensitivity.json; s1_assignment_comparisons.csv; s1_application_round_truncation.csv; s1_geometry_cutoff.csv")
    source("scaling", all_paths("/campaign-session/manifest.json") + all_paths("/campaign-summary.json"),
           "cases family in scaling,large_square,large_rectangle,extreme_rectangle; estimates[].seconds and primary paired ratio",
           "47+8+12+12=79 cases. Curves hold objective/generator/orientation/limit policy fixed; geometry companion40 cases per environment.", artifact="A1_scaling.csv; F3_geometry_5to4_companion.csv")
    source("rejection", all_paths("/campaign-session/manifest.json") + all_paths("/campaign-summary.json"),
           "family=rejection; case.limit; primary attempts/<attempt_id>/measurements.jsonl saved validation records; original cardinalities[] and input hashes",
           "Use saved validation cardinality K:unmatched percent=100*(r-K)/r; no input generation or solve. Timings and cardinalities remain separate measured quantities.", artifact="F11_rejection.csv; S2_rejection_validation.csv")
    source("arithmetic", ["latex/main.tex (formulation retained from commit398b813ac1fd7437d8f95e8ad72b8ed3a6e381a8)"],
           "Equations1–7 and associated proofs; explicit illustrative matrix; binary64 cost-buffer dimensions",
           "Algebra checked: F=P+L(c-r)/2; primal-dual gap nonnegative; capped extension proof.256*4096*8/2^20=8MiB;4096^2*8/2^20=128MiB;4352^2*8/2^20=144.5MiB. These are definitions/arithmetic,not empirical run claims.", kind="mathematical_derivation")
    source("implementation", ["../laprust/src/_lapjv/src/rectangular.rs@v1.2.0", "../laprust/src/_lapjv/src/ffi.rs@v1.2.0", "../laprust/src/_lapjv/src/square.rs@v1.2.0", "../laprust/src/_lapjv/_lapjv.pyx@v1.2.0", "../lapcpp/src/_lapjv/solver.cpp@v1.2.0", "../lapcpp/src/_lapjv/_lapjv.pyx@v1.2.0", "../lapx/src/_lapjvr/src/rectangular.rs@v0.13.0", "../lapx/src/_lapjvr/src/ffi.rs@v0.13.0", "../lapx/src/_lapjvr/src/square.rs@v0.13.0", "../lapx/src/_lapjvr/src/cpp_completion.cpp@v0.13.0", "../lapx/src/_lapjvr/_lapjv_rh.pyx@v0.13.0"],
           "Rust rectangular.rs:18–22,176–279,search below279; CPP solver.cpp:285–432,540–542,597–598; Cython wrappers' original/work buffers, total accumulation and required-square dispatch in FFI/solver.cpp; square completion routines",
           "Verified cost storage T and float64work;two passes,B=min(c,4),required cutoff max(r,8),free-column ties;implicit terminal;half-limit rounding;chunk≈4096costs. When requested, totals sum selected entries from the original float32/float64 buffer in float64; other accepted real dtypes first convert to float64. For required squares JVRUST uses Rust completion, JVHYBRID calls C++ completion on Rust-owned workspace and JVCPP uses C++ throughout. Mathematical operation bounds derive from bounded scans/augmentations,not fitted timing.", kind="pinned_source_and_derivation")
    source("literature", ["latex/references.bib", "https://arxiv.org/abs/1603.04380v3", "benchmark/lsape/src/upstream_download_check.json"],
           "Citation keys on indexed line or table caption. For the LSAPE design comparison, Bougleux and Brun (2016), Eq. (19), gives the edit-cost matrix; Section 4 and Algorithms 2–3 describe the augmenting paths and Hungarian algorithm; Figures 4–5 compare LSAPE and squared-LSAPE on synthetic edit costs. Source download check: retrieved_utc,matches_supplied,official_download_sha256,supplied_archive_sha256",
           "Retained literature attribution from398b813; no new empirical inference from historical timings. Download record confirms27October2026 archive equality,not2016 implementation identity.", kind="retained_cited_literature")
    source("reporting", ["benchmark/generate_paper_artifacts.py", "benchmark/paper_artifacts_common.py", "benchmark/paper_artifacts_campaign.py", "benchmark/paper_artifacts_support.py", "benchmark/paper_artifacts_sensitivity.py"],
           "Saved-run allowlist,classification function,generator outputs and source indices",
           "Editorial convention ±3% is specified,not estimated. Separate environment observations,planned exclusions and source identities are retained. No significance or universal-performance interpretation.", kind="declared_reporting_convention")

    source("reference_sensitivity", all_paths("/campaign-summary.json") + all_paths("/campaign-session/manifest.json"),
           "All1072 directed single-case sibling-environment comparisons; primary-reference latency ratio>=1.30 and JVRUST latency ratio in[.90,1.10]; matched input hashes; no upper reference threshold",
           "Seven post-hoc screened events. Three observed CSGRAPH sibling class disagreements; two changes holding target CSGRAPH latency fixed and substituting sibling reference. Native classifications unchanged. Cross-interpreter/session differences are confounded; no primary observation is corrected or removed.", artifact="reference_sensitivity.json; reference_sensitivity_events.csv; reference_sensitivity_methods.csv")

    figure_keys = {"F1":"native","F2":"geometry","F3":"layout","F4":"policy","F5":"sparse","F6":"batch","F7":"applications","F8":"lsape","F9":"orientation","F_structure":"structure","F11":"rejection","A1":"scaling","A2":"geometry","A4":"memory"}
    table_keys = {"T1_protocol":"protocol","T2_methods":"implementation","T3_counters":"counters","T4_memory":"memory","A_application_workloads":"workloads","A_application_accuracy":"accuracy","s1_sensitivity":"sensitivity","S_native_family_counts":"native","S_native_exceptions":"native","S_lsape_family_counts":"lsape","T_microscopy":"microscopy","A_microscopy_profiles":"microscopy"}

    def mapping(text, section, subsection, image="", table=""):
        if table:
            k=table_keys.get(Path(table).stem)
            if k:return [k,"environment","applications","lsape"] if k=="protocol" else [k,"arithmetic","timed_adapters"] if Path(table).stem=="T2_methods" else [k]
        if image:
            stem=Path(image).stem
            if stem.startswith("A1_geometry"):return ["geometry"]
            for key in sorted(figure_keys,key=len,reverse=True):
                if stem.startswith(key+"_"):return [figure_keys[key]]
            if "rejection_paths" in stem:return ["implementation","arithmetic"]
        if section=="Abstract":return ["protocol","environment","native","applications","accuracy","microscopy","lsape","reporting"]
        if section=="Introduction":
            if "MiB" in text:return ["arithmetic"]
            return ["literature","implementation","protocol","reporting"]
        if section=="Background and related work":
            if r"\ref{tab:lsape-design}" in text:return ["literature","implementation","arithmetic","lsape"]
            if text.startswith("Evaluation &"):return ["literature","implementation","protocol","layout","batch"]
            return ["literature","implementation","arithmetic"]
        if section=="Problem formulation":return ["arithmetic"]
        if section.startswith("Implementation"):
            if "post-filter" in text:return ["implementation","arithmetic","validation","timed_adapters"]
            return ["implementation","arithmetic","validation"]
        if section=="Experimental design":
            if "common output filter" in text:return ["environment", "protocol", "timed_adapters"]
            if "Quantized costs" in text:return ["protocol", "structure"]
            if text.startswith("The Windows-only LSAPE study"):return ["lsape"]
            if "LSAPE" in subsection:return ["applications","lsape","accuracy","microscopy"]
            if "Geometry" in subsection:return ["geometry","layout","policy","counters"]
            if "Validation" in subsection:return ["validation","batch","memory"]
            if "coverage" in subsection:return ["protocol","native","batch","reporting"]
            return ["environment","protocol","reporting","microscopy"]
        if section=="Results":
            keys=[("Coverage","native"),("Geometry","geometry"),("Representation","layout"),("Initialization","counters"),("Sparse","sparse"),("Ordered","structure"),("Batching","batch"),("Applications","applications"),("LSAPE","lsape"),("Orientation","orientation"),("Memory","memory")]
            for needle,key in keys:
                if needle in subsection:
                    if key=="counters" and "counters" not in text.lower():return ["policy"]
                    if key=="applications":return ["applications","accuracy","microscopy"]
                    if key=="structure" and "rejection" in text.lower():return ["structure","rejection"]
                    if key=="sparse" and "post-hoc" in text.lower():return ["sparse","reference_sensitivity"]
                    return [key]
        if section=="Conclusion":return ["native","applications","microscopy","lsape","reporting"]
        if section=="Discussion":return ["native","policy","counters","batch","orientation","applications","accuracy","microscopy","implementation","reporting"]
        if section=="Limitations and threats to validity":
            if "worked example outside the frozen preflight" in text:return ["binary64_example", "validation"]
            return ["protocol","environment","geometry","validation","applications","microscopy","lsape","memory","reference_sensitivity","reporting"]
        if section=="Classification tables":return ["native","lsape"]
        if section=="Scaling and geometric sensitivity":return ["scaling","geometry"]
        if section=="Rejection-limit sweep":return ["rejection"]
        if section=="Memory-counter visibility":return ["memory"]
        if section=="Application workload and accuracy":return ["applications","accuracy","workloads"]
        if section=="Microscopy inputs and association profiles":return ["microscopy"]
        if section=="Protocol sensitivity and indexed supplement":return ["sensitivity","reporting"]
        if section in ("Data and software availability","AI assistance disclosure","Acknowledgements"):return ["reporting"]
        return []

    records, unknown = [], []
    path=repo/"latex/main.tex"
    lines=path.read_text(encoding="utf-8").splitlines()
    section=subsection=image=""
    table_part=""
    active=False
    math_block=None
    fragments=[]
    for i,line in enumerate(lines,1):
        if line==r"\maketitle":active=True
        if not active:continue
        if line==r"\begin{abstract}":section="Abstract"
        sm=re.match(r"\\section\*?\{([^}]*)\}",line)
        if sm:section=sm.group(1);subsection=""
        sm=re.match(r"\\subsection\{([^}]*)\}",line)
        if sm:subsection=sm.group(1)
        if line.startswith((r"\begin{figure}",r"\begin{table}")):image=""
        if line.startswith(r"\toprule"):table_part="header"
        elif line.startswith(r"\midrule"):table_part="body"
        elif line.startswith((r"\bottomrule",r"\end{tabular")):table_part=""
        sm=re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}",line)
        if sm:image=sm.group(1)
        sm=re.search(r"\\input\{([^}]*)\}",line)
        if sm and sm.group(1).startswith("generated/"):fragments.append(sm.group(1))
        if line==r"\[":
            math_block=(i,[])
            continue
        if math_block is not None:
            if line==r"\]":
                start,content=math_block
                statement="\n".join(content)
                keys=mapping(statement,section,subsection)
                record=dict(id=f"M{len(records)+1:04d}",file="latex/main.tex",line=start,section=section,subsection=subsection,
                            statement=statement,numeric_tokens=re.findall(r"\d[\d,.]*",statement),evidence_keys=keys)
                if not keys:unknown.append(record)
                records.append(record)
                math_block=None
            else:
                math_block[1].append(line)
            continue
        if line.startswith(SKIP) or not line.strip():continue
        cleaned=re.sub(r"\\(?:cite|ref|label)\{[^}]*\}","",line)
        inline_row = " & " in line and bool(table_part)
        if inline_row and table_part=="header":continue
        if not inline_row and not NUMBER.search(cleaned) and not re.search(r"\\\(|(?<!\\)\$",cleaned):continue
        table=""
        if line.startswith(r"\caption"):
            for following in lines[i:i+8]:
                if following.startswith(r"\end{"):break
                sm=re.search(r"\\input\{([^}]*)\}",following)
                if sm:table=sm.group(1);break
        keys=mapping(line,section,subsection,image if line.startswith(r"\caption") else "",table)
        record=dict(id=f"M{len(records)+1:04d}",file="latex/main.tex",line=i,section=section,subsection=subsection,
                    statement=line,numeric_tokens=re.findall(r"\d[\d,.]*(?:--\d+)?|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|all|every|none)\b",cleaned,re.I),evidence_keys=keys)
        if not keys:unknown.append(record)
        records.append(record)
    for relative in dict.fromkeys(fragments):
        fp=repo/"latex"/relative
        if not fp.exists():unknown.append(dict(file=str(fp),reason="referenced fragment does not exist"));continue
        stem=fp.stem
        key=table_keys.get(stem,"scaling" if stem=="A1_scaling_figures" else None)
        for i,line in enumerate(fp.read_text(encoding="utf-8").splitlines(),1):
            if line.startswith(SKIP) or not NUMBER.search(line):continue
            keys=[key] if key else []
            if stem=="T1_protocol":keys=["protocol","environment","applications","lsape"]
            if stem=="T2_methods":keys=["implementation","arithmetic","lsape","timed_adapters"]
            record=dict(id=f"M{len(records)+1:04d}",file=fp.relative_to(repo).as_posix(),line=i,section="Generated table/appendix fragment",subsection=stem,statement=line,numeric_tokens=re.findall(r"\d[\d,.]*",line),evidence_keys=keys)
            if not keys:unknown.append(record)
            records.append(record)
    result=dict(scope="Quantitative paragraphs (including symbolic formulas),displayed equations,captions,inline table body rows and generated numeric table rows; technical preamble,automatic numbering and citation-key digits excluded. Each record identifies sources for the claims in its excerpt. This editorial index does not replace the underlying observations or constitute automatic proof.",
                manuscript_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),environment_order=list(roots),primary_roots=roots,
                quantitative_records=len(records),unmapped_records=unknown,evidence=evidence,claims=records)
    result["independent_empirical_recomputation"]=audit_empirical_claims(repo,roots)
    (directory/"manuscript_claims.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    with (directory/"manuscript_claims.csv").open("w",encoding="utf-8",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=["id","file","line","section","subsection","statement","evidence_keys"])
        writer.writeheader()
        for record in records:writer.writerow({k:(";".join(record[k]) if k=="evidence_keys" else record[k]) for k in writer.fieldnames})
    assert not unknown, f"Unmapped quantitative text: {unknown}"
    return dict(quantitative_records=len(records),unmapped_records=0,manuscript_sha256=result["manuscript_sha256"])


if __name__=="__main__":
    print(json.dumps(generate(Path(__file__).resolve().parents[1]),indent=2))
