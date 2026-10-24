"""Render all figure PDFs and prepare a review record; human/visual review is separate."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

ROOT = Path(__file__).resolve().parent
BASE = ("single_runtime", "batch_runtime", "memory_peak", "scaling_required", "scaling_optional", "scaling_blocks")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pages(path):
    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    return int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))


def main():
    rendered = ROOT / "tmp/pdf-review"
    rendered.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "extended/figure_manifest.json").read_text())
    items = [(ROOT / "figures", stem) for stem in BASE]
    items += [(ROOT / "extended/figures", item["stem"]) for item in manifest["figures"]]
    records = []
    for folder, stem in items:
        pdf = folder / f"{stem}.pdf"
        assert pages(pdf) == 1
        ET.parse(folder / f"{stem}.svg")
        with Image.open(folder / f"{stem}.png") as im:
            dimensions = list(im.size)
            im.verify()
        subprocess.run(["pdftoppm", "-r", "120", "-singlefile", "-png", str(pdf), str(rendered / stem)],
                       check=True, capture_output=True)
        records.append(dict(stem=stem, pdf_pages=1, svg_parse="passed", png_dimensions=dimensions,
            sha256={str((folder / f"{stem}.{ext}").relative_to(ROOT)).replace("\\", "/"):digest(folder / f"{stem}.{ext}")
                    for ext in ("png", "svg", "pdf")},
            rendered_preview_sha256=digest(rendered / f"{stem}.png")))
    packet = ROOT / "charts.pdf"
    subprocess.run(["pdfunite", *[str(folder / f"{stem}.pdf") for folder, stem in items], str(packet)], check=True)
    assert pages(packet) == len(items) == 19
    # Check that concatenation preserves each page's rendered appearance.
    for index, (_, stem) in enumerate(items, 1):
        target = rendered / f"packet-{index:02d}"
        subprocess.run(["pdftoppm", "-f", str(index), "-l", str(index), "-r", "120", "-singlefile", "-png",
                        str(packet), str(target)], check=True, capture_output=True)
        with Image.open(rendered / f"{stem}.png") as a, Image.open(target.with_suffix(".png")) as b:
            assert a.size == b.size and a.tobytes() == b.tobytes(), stem
    review = dict(status="awaiting_visual_review", figures=records,
        chart_packet=dict(file="charts.pdf",pages=19,sha256=digest(packet),rendered_pages_identical=True),
        separate_base_png_layouts=["single_runtime", "batch_runtime", "memory_peak"])
    (ROOT / "figure_review_pending.json").write_text(json.dumps(review, indent=2)+"\n", encoding="utf-8")
    print("Rendered 19 individual PDFs; combined packet has 19 visually identical pages. Awaiting visual inspection.")


if __name__ == "__main__":
    main()
