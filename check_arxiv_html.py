"""Check the actual upload ZIP with LaTeXML and make a local HTML preview.

Requires LaTeXML 0.8.8 or later, TeX Live and Poppler's pdftoppm on PATH.
Portable Perl installations can use --perl, --latexml-bin and --perl-lib.
The upload stays unchanged; Poppler renders its PDF figures for this preview.
"""
from __future__ import annotations

import argparse
from collections import Counter
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from build_pdf import PAPER, sha256


class HTMLCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.classes = Counter()
        self.tags = Counter()
        self.ids = set()
        self.links = []
        self.images = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags[tag] += 1
        self.classes.update(attrs.get("class", "").split())
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.links.append(attrs["href"][1:])
        if tag == "img" and "ltx_graphics" in attrs.get("class", "").split():
            self.images.append(attrs)

    def handle_data(self, text):
        self.text.append(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latexml-bin", type=Path)
    parser.add_argument("--perl", type=Path)
    parser.add_argument("--perl-lib", type=Path)
    options = parser.parse_args()
    env = os.environ.copy()
    env.update(LC_ALL="C", LANG="C")
    if options.perl_lib:
        env["PERL5LIB"] = str(options.perl_lib.resolve())
    for key in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFOUTPUT"):
        env.pop(key, None)

    def tool(name):
        path = (options.latexml_bin.resolve() / name if options.latexml_bin
                else shutil.which(name))
        if not path or not Path(path).is_file():
            parser.error(f"Cannot find {name}; install LaTeXML or use --latexml-bin.")
        return ([str(options.perl.resolve())] if options.perl else []) + [str(path)]

    latexml, post = tool("latexml"), tool("latexmlpost")
    poppler = shutil.which("pdftoppm")
    if not poppler:
        parser.error("Poppler's pdftoppm must be on PATH.")
    output = PAPER / "output" / "arxiv"
    archive = output / "native_rectangular_assignment_arxiv.zip"
    scratch = PAPER / "tmp" / "arxiv"
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="html-", dir=scratch))
    source = work / "source"
    with zipfile.ZipFile(archive) as upload:
        upload.extractall(source)
    preview = work / "html"
    (preview / "images").mkdir(parents=True)
    commands = []

    def run(command, label):
        result = subprocess.run(command, cwd=source, env=env, text=True,
                                errors="replace", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (work / f"{label}.log").write_text(result.stdout, encoding="utf-8")
        commands.append({"step": label, "command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {work / (label + '.log')}\n"
                               + result.stdout[-4000:])
        return result.stdout

    xml_path = work / "main.xml"
    run(latexml + ["--strict", f"--destination={xml_path}", "main.tex"], "conversion")
    xml_text = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_text)
    ns = {"ltx": "http://dlmf.nist.gov/LaTeXML"}
    tex = (source / "main.tex").read_text(encoding="utf-8")
    graphics = root.findall(".//ltx:graphics", ns)
    math_nodes = root.findall(".//ltx:Math", ns)
    # LaTeXML also creates duplicate math for list-of-figures captions and
    # linear alternatives to aligned equations. HTML renders the display branch.
    hidden_math = set(root.findall(".//ltx:toccaption//ltx:Math", ns))
    hidden_math.update(root.findall(".//ltx:MathFork/ltx:Math", ns))
    visible_math = [node for node in math_nodes if node not in hidden_math]
    expected = {
        "figures": len(re.findall(r"\\begin\{figure\}", tex)),
        "tables": len(re.findall(r"\\begin\{table\}", tex)),
        "references": len(re.findall(r"\\bibitem", (source / "main.bbl").read_text(encoding="utf-8"))),
        "algorithms": len(re.findall(r"\\begin\{algorithm\}", tex)),
        "math_expressions": len(visible_math),
    }
    assert len(graphics) == expected["figures"]
    assert len(root.findall("ltx:title", ns)) == 1
    assert len(root.findall("ltx:creator", ns)) == 1
    assert len(root.findall("ltx:date", ns)) == 1
    assert len(root.findall("ltx:abstract", ns)) == 1
    assert not root.findall(".//ltx:ERROR", ns)

    # Keep the exact converter XML as evidence. Only image delivery metadata is
    # added to a second XML file, using Poppler instead of optional PerlMagick.
    image_attributes = []
    for number, graphic in enumerate(graphics, 1):
        name = Path(graphic.attrib["graphic"])
        assert graphic.get("description"), f"Figure lacks alt text: {name}"
        target = preview / "images" / name.stem
        run([poppler, "-r", "144", "-png", "-singlefile", str(source / name),
             str(target)], f"figure-{number}")
        png = target.with_suffix(".png")
        width, height = struct.unpack(">II", png.read_bytes()[16:24])
        shown_width = min(width, 800)
        image_attributes.append(
            f' imagesrc="images/{png.name}" imagewidth="{shown_width}"'
            f' imageheight="{round(height * shown_width / width)}"')
    attributes = iter(image_attributes)
    web_xml = re.sub(r"<graphics\b[^>]*?/>",
                     lambda match: match[0][:-2] + next(attributes) + "/>", xml_text)
    assert web_xml.count(' imagesrc="images/') == expected["figures"]
    web_path = work / "web.xml"
    web_path.write_text(web_xml, encoding="utf-8")
    # These rules only fit the local preview into a narrow viewer. arXiv applies
    # its own stylesheet, so this file is never included in the upload ZIP.
    css = work / "paper-preview.css"
    css.write_text("img.ltx_graphics { max-width: 100%; height: auto; }\n"
                   ".ltx_tabular, .ltx_eqn_table { display: block; max-width: 100%; overflow-x: auto; }\n",
                   encoding="utf-8")
    run(post + ["--format=html5", f"--sourcedirectory={source}",
                f"--css={css}", f"--destination={preview / 'index.html'}", str(web_path)], "postprocess")

    check = HTMLCheck()
    html = (preview / "index.html").read_text(encoding="utf-8")
    check.feed(html)
    actual = {
        "figures": check.classes["ltx_figure"],
        "tables": check.classes["ltx_table"],
        "references": check.classes["ltx_bibitem"],
        "algorithms": check.classes["ltx_float_algorithm"],
        "math_expressions": check.tags["math"],
    }
    assert actual == expected, (actual, expected)
    assert all(node.get("{http://www.w3.org/XML/1998/namespace}id") in check.ids
               for node in visible_math), "A displayed formula is missing from HTML"
    assert check.tags["h1"] == 1
    assert check.classes["ltx_abstract"] == 1
    assert not [key for key in check.classes if "error" in key.lower() or "missing" in key.lower()]
    assert all(link in check.ids for link in check.links), "Broken internal HTML link"
    assert len(check.images) == expected["figures"]
    assert all(image.get("alt") and (preview / image["src"]).is_file() for image in check.images)
    visible = " ".join(check.text)
    for leaked in ("ItalicFont=", "BoldFont=", "Scale=MatchLowercase", "setmathfont", ".otf"):
        assert leaked not in visible, f"Leaked TeX font setting: {leaked}"
    assert "Native Rectangular Linear Assignment with Implicit Rejection" in visible
    assert "Ratha SIV" in visible and "22 October 2026" in visible
    diagnostics = []
    for label in ("conversion", "postprocess"):
        log = (work / f"{label}.log").read_text(encoding="utf-8")
        diagnostics.extend(line for line in log.splitlines()
                           if re.match(r"(?:Warning|Error|Fatal):", line))
    assert not diagnostics, diagnostics
    def copy_preview_file(source_file, target_file):
        target_path = Path(target_file)
        if target_path.exists():
            target_path.chmod(0o644)
        return shutil.copyfile(source_file, target_file)

    shutil.copytree(preview, output / "html", dirs_exist_ok=True,
                    copy_function=copy_preview_file,
                    ignore=shutil.ignore_patterns("*.cache"))
    report = {
        "archive_sha256": sha256(archive),
        "converter": run(latexml + ["--VERSION"], "version").strip(),
        "source": "Unmodified main.tex and main.bbl extracted from the upload ZIP",
        "figure_rendering": "Original PDF figures rendered by Poppler for the local HTML preview only",
        "counts": actual, "diagnostics": diagnostics,
        "parsed_math_expressions_including_alternatives": len(math_nodes),
        "font_settings_leaked": False, "all_internal_links_resolve": True,
        "all_figures_have_alt_text": True,
        "preview": "output/arxiv/html/index.html",
        "preview_sha256": sha256(output / "html" / "index.html"),
        "build_directory": work.relative_to(PAPER).as_posix(), "commands": commands,
        "scope": "Local LaTeXML validation; arXiv's deployed version and styling may differ.",
    }
    (output / "html_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(actual))
    print("HTML verified:", output / "html" / "index.html")


if __name__ == "__main__":
    main()
