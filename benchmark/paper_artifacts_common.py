"""Shared identities and saved-result helpers; never imports the benchmark runner."""
from pathlib import Path
import csv
import json
import gzip
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from paper_artifacts_bounds import validate_figure_bounds

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "benchmark/paper-artifacts"
FIGURES = REPO / "latex/images/current"
TABLES = REPO / "latex/generated"
for directory in (DATA, FIGURES, TABLES):
    directory.mkdir(parents=True, exist_ok=True)
ENVS = [
    ("L311", "Linux 3.11.18", "linux-py3.11.18-20261031-0455", "-"),
    ("L314", "Linux 3.14.10", "linux-py3.14.10-20261031-0630", "--"),
    ("W311", "Windows 3.11.18", "windows-py3.11.18-20261030-2322", ":"),
    ("W314", "Windows 3.14.10", "windows-py3.14.10-20261031-0322", "-."),
]
STYLE = json.loads((REPO / "benchmark/paper_artifact_panels/style_mapping.json").read_text(encoding='utf-8'))
TYPOGRAPHY={}
SUPPRESS_FIGURES=False
LEGEND_LABELS={'scipy':'SciPy required','scipy:capped':'SciPy capped',
               'scipy:dummies':'Dummy columns','scipy:diagonal_dummies':'Diagonal dummies',
               'scipy:square':'Square SciPy'}

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def save(fig, name):
    if SUPPRESS_FIGURES:
        return
    validate_figure_bounds(fig, name)
    # The manuscript scales empirical plots to at most 6.46 inches. Keep every
    # visible text label at least 6.2 points at that width; figures use compact
    # titles/legends and explicit margins to accommodate this publication size.
    fig.canvas.draw()
    minimum=6.2*fig.get_figwidth()/6.46
    for label in fig.findobj(Text):
        if label.get_visible() and label.get_text().strip():
            label.set_fontsize(max(label.get_fontsize(),minimum))
    fig.canvas.draw()
    texts=[t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    TYPOGRAPHY[name]={'canvas_inches':list(fig.get_size_inches()),'reference_print_width_inches':6.46,
                      'minimum_label_points_at_reference_width':min(t.get_fontsize() for t in texts)*6.46/fig.get_figwidth(),
                      'visible_text_labels':len(texts)}
    (DATA/'campaign_typography.json').write_text(json.dumps(TYPOGRAPHY,indent=2)+'\n',encoding='utf-8')
    # Publish each completed file atomically, so a simultaneous LaTeX build
    # cannot read a half-written PDF.
    temporary=FIGURES/(name+'.pdf.tmp')
    fig.savefig(temporary, format='pdf', facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    temporary.replace(FIGURES/(name+'.pdf'))
    temporary=FIGURES/(name+'.png.tmp')
    fig.savefig(temporary, format='png', dpi=180, facecolor="white")
    temporary.replace(FIGURES/(name+'.png'))

def writecsv(name, rows, compress=False):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    target=DATA/name
    if compress:
        raw=(DATA/(name+'.gz')).open('wb')
        stream=gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0)
        f=io.TextIOWrapper(stream,encoding='utf-8',newline='')
    else:
        raw=None
        f=target.open('w',encoding='utf-8',newline='')
    with f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in r.items()} for r in rows)
    if raw is not None:
        raw.close()
        # Only replace the known generated counterpart, never a source file.
        if target.exists():target.unlink()

def classification(value):
    return "win" if value > 1.03 else "loss" if value < .97 else "tie"

def style(method):
    if method.startswith("policy:") or method.startswith("workspace:"):
        return STYLE["jvrust" if ":rust:" in method else "jvcpp"]
    if method.startswith(("cost:", "shape:")):
        return STYLE[method.split(":")[1]]
    return STYLE[method]

def legends(fig, methods, top=.96, environment_top=.91):
    handles = [Line2D([], [], color=style(m)["color"], marker=style(m)["marker"], lw=1, ms=4, label=LEGEND_LABELS.get(m,style(m)["label"])) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, top), ncol=min(5,len(methods)), frameon=False, fontsize=8)
    fig.legend(handles=[Line2D([],[],color=".4",ls=ls,label=env) for env,_,_,ls in ENVS], loc="upper center", bbox_to_anchor=(.53,environment_top),ncol=4,frameon=False,fontsize=8)

def read_runs():
    result = []
    for env,label,run,ls in ENVS:
        root = REPO / "benchmark/runs" / run / "paper-run-light"
        summary = load(root / "campaign-summary.json")
        manifest = load(root / "campaign-session/manifest.json")
        assert summary["profile"] == manifest["profile"] == "light"
        assert len(manifest["cases"]) == 302
        assert summary["coverage"]["complete_attempts"] == 3286
        assert not summary["unobserved_planned_conditions"]
        result.append(dict(env=env,label=label,run=run,ls=ls,root=root,summary=summary,manifest=manifest,
                           cases={c["id"]:c for c in manifest["cases"]},
                           estimates={(e["case_id"],e["method"],e.get("workers",1)):e for e in summary["estimates"]},
                           single={(e["case_id"],e["method"]):e for e in summary["estimates"] if e["status"]=="complete"}))
    return result
