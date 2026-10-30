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

from build_publication import PAPER, sha256


EQUATION_TAG_BINDING = r"""# Preview-only compatibility with LaTeXML's explicit equation tags.
use LaTeXML::Package;
AtBeginDocument(sub {
  DefMacro('\[', '\begin{equation*}');
  DefMacro('\]', '\end{equation*}');
  return;
});
1;
"""
LTX = "http://dlmf.nist.gov/LaTeXML"
XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


class HTMLCheck(HTMLParser):
    def __init__(self):
        super().__init__()
        self.classes = Counter()
        self.tags = Counter()
        self.ids = set()
        self.links = []
        self.images = []
        self.text = []
        self.equation_tags = []
        self._elements = []

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
        classes = attrs.get("class", "").split()
        hidden = ("hidden" in attrs or attrs.get("aria-hidden", "").lower() == "true"
                  or bool(re.search(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)",
                                    attrs.get("style", ""), flags=re.I)))
        element = {"tag": tag, "hidden": hidden}
        if attrs.get("id") and {"ltx_equation", "ltx_equationgroup"} & set(classes):
            element["equation_id"] = attrs["id"]
            element["equation_kind"] = "group" if "ltx_equationgroup" in classes else "equation"
        if {"ltx_tag_equation", "ltx_tag_equationgroup"} & set(classes):
            kind = "group" if "ltx_tag_equationgroup" in classes else "equation"
            number = {"text": [], "hidden": hidden or any(e["hidden"] for e in self._elements),
                      "equation_id": next((e["equation_id"] for e in reversed(self._elements)
                                           if e.get("equation_kind") == kind), None)}
            self.equation_tags.append(number)
            element["equation_tag"] = number
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self._elements.append(element)

    def handle_endtag(self, tag):
        for index in range(len(self._elements) - 1, -1, -1):
            if self._elements[index]["tag"] == tag:
                del self._elements[index:]
                break

    def handle_data(self, text):
        self.text.append(text)
        for element in self._elements:
            if "equation_tag" in element:
                element["equation_tag"]["text"].append(text)


def numbered_equation_ids(root, expected_labels):
    """Require semantic visible/refnum tags on the intended display containers."""
    numbered = {}
    for node in root.iter():
        if node.tag not in {f"{{{LTX}}}equation", f"{{{LTX}}}equationgroup"}:
            continue
        tags = node.find(f"{{{LTX}}}tags")
        if tags is None:
            continue
        refnums = tags.findall(f"{{{LTX}}}tag[@role='refnum']")
        visible = [tag for tag in tags.findall(f"{{{LTX}}}tag") if not tag.get("role")]
        assert len(refnums) == len(visible) == 1, "Equation tag lacks unique display/refnum metadata"
        label = "".join(refnums[0].itertext()).strip()
        assert "".join(visible[0].itertext()).strip() == f"({label})", "Equation display label differs"
        assert label not in numbered, "Duplicate numbered equation: " + label
        assert node.get(XML_ID) and node.findall(f".//{{{LTX}}}Math"), "Equation label lacks its formula container"
        numbered[label] = node.get(XML_ID)
    assert list(numbered) == expected_labels, "Numbered XML equations differ from the source tags"
    return numbered


def check_equation_numbers(check, numbered):
    """Verify each number is visible inside its corresponding HTML equation."""
    observed = {}
    for record in check.equation_tags:
        text = "".join(record["text"]).strip()
        match = re.fullmatch(r"\((\d+)\)", text)
        assert match and not record["hidden"], "Equation label is missing or hidden: " + text
        label = match[1]
        assert label not in observed, "Duplicate HTML equation number: " + label
        assert numbered.get(label) == record["equation_id"], "Equation label is outside its formula container: " + label
        observed[label] = record["equation_id"]
    assert observed == numbered, "HTML is missing one or more numbered equations"
    return observed


def normalize_source_comments(xml_text):
    """Remove trailing whitespace inside converter comments, preserving nodes."""
    return re.sub(r"<!--.*?-->",
                  lambda match: re.sub(r"[ \t]+$", "", match[0], flags=re.M),
                  xml_text, flags=re.S)


def portable_path(value, work, paper=PAPER):
    """Describe a build path without recording a machine-specific root."""
    value = str(value)
    if not re.match(r"(?:[A-Za-z]:[\\/]|\\\\|/)", value):
        return value
    path = Path(value)
    for root, prefix in ((Path(work), "[BUILD]"), (Path(paper), "")):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        return f"{prefix}/{relative}" if prefix else relative
    # External executables are identified by their command name and recorded
    # version, rather than by a private installation directory.
    return re.split(r"[\\/]", value.rstrip("\\/"))[-1]


def portable_command(command, work, paper=PAPER):
    """Return a publication-safe copy; subprocess arguments stay untouched."""
    result = []
    for argument in command:
        value = str(argument)
        if value.startswith("--") and "=" in value:
            option, path = value.split("=", 1)
            result.append(option + "=" + portable_path(path, work, paper))
        else:
            result.append(portable_path(value, work, paper))
    return result


def portable_xml_metadata(xml_text, work, paper=PAPER):
    """Redact only local paths in LaTeXML processing instructions after use."""
    def instruction(match):
        def attribute(item):
            value = portable_path(item["value"], work, paper)
            return item["key"] + "=" + item["quote"] + value + item["quote"]
        return re.sub(r"(?P<key>\b(?:searchpaths|package))\s*=\s*"
                      r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
                      attribute, match[0], flags=re.S)
    return re.sub(r"<\?latexml\b.*?\?>", instruction, xml_text, flags=re.S)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latexml-bin", type=Path)
    parser.add_argument("--perl", type=Path)
    parser.add_argument("--perl-lib", type=Path)
    options = parser.parse_args()
    env = os.environ.copy()
    env.update(LC_ALL="C", LANG="C")
    if options.perl:
        # Portable Strawberry Perl's XS modules depend on DLLs in c/bin.
        # Change only the child-process environment, preserving the user's PATH.
        perl_bin = options.perl.resolve().parent
        portable_root = perl_bin.parent.parent
        runtime_bins = (perl_bin, perl_bin.parent / "site/bin", portable_root / "c/bin")
        env["PATH"] = os.pathsep.join([str(path) for path in runtime_bins if path.is_dir()]
                                     + [env.get("PATH", "")])
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
    scratch = PAPER / "tmp" / "pdfs" / "html"
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
        commands.append({"step": label, "command": portable_command(command, work),
                         "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{label} failed; see {work / (label + '.log')}\n"
                               + result.stdout[-4000:])
        return result.stdout

    xml_path = work / "main.xml"
    # LaTeXML 0.8.8's \[ constructor omits equation tags. Its supported
    # equation* constructor retains explicit \tag metadata, including tags on
    # aligned groups. Defer this local binding until package loading finishes.
    # The manuscript extracted from the upload is never rewritten.
    equation_binding = work / "paper-equation-tags.ltxml"
    equation_binding.write_text(EQUATION_TAG_BINDING, encoding="utf-8", newline="\n")
    # Use the supplied BibTeX output, as an arXiv-style build does. Without
    # nobibtex, LaTeXML rebuilds references.bib and can renumber citations.
    run(latexml + ["--strict", "--preload=[nobibtex]latexml.sty",
                   f"--preload={equation_binding}",
                   f"--destination={xml_path}", "main.tex"], "conversion")
    xml_text = xml_path.read_text(encoding="utf-8")
    xml_text = normalize_source_comments(xml_text)
    xml_path.write_text(xml_text, encoding="utf-8", newline="\n")
    root = ET.fromstring(xml_text)
    ns = {"ltx": "http://dlmf.nist.gov/LaTeXML"}
    tex = (source / "main.tex").read_text(encoding="utf-8")
    equation_labels = re.findall(r"\\tag\{(\d+)\}", tex)
    assert equation_labels == [str(number) for number in range(1, 8)], "Expected all seven source equation tags"
    numbered = numbered_equation_ids(root, equation_labels)
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

    # Keep working converter XML with only comment whitespace normalized. Its
    # published copy loses local processing-instruction paths after conversion.
    # A second XML
    # file supplies preview images and corrects a numeric-label bug in LaTeXML's natbib
    # binding: its citation style is 'numbers', but NAT@wrout tests 'number'.
    bbl_keys = re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}",
                          (source / "main.bbl").read_text(encoding="utf-8"))
    bibitems = root.findall(".//ltx:bibitem", ns)
    assert [node.get("key") for node in bibitems] == bbl_keys
    for number, node in enumerate(bibitems, 1):
        assert node.find("ltx:tags/ltx:tag[@role='number']", ns).text == str(number)
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
    def numeric_bibitem(match):
        number = re.search(r'<tag role="number">(\d+)</tag>', match[0]).group(1)
        corrected, count = re.subn(r'<tag role="refnum">.*?</tag>',
                                   f'<tag role="refnum">[{number}]</tag>',
                                   match[0], flags=re.S)
        assert count == 1
        return corrected
    web_xml, corrected = re.subn(r"<bibitem\b.*?</bibitem>", numeric_bibitem,
                                  web_xml, flags=re.S)
    assert corrected == expected["references"]
    web_path = work / "web.xml"
    web_path.write_text(web_xml, encoding="utf-8")
    # These rules only fit the local preview into a narrow viewer. arXiv applies
    # its own stylesheet, so this file is never included in the upload ZIP.
    css = work / "paper-preview.css"
    css.write_text("img.ltx_graphics { max-width: 100%; height: auto; }\n"
                   ".ltx_tabular, .ltx_eqn_table { display: block; max-width: 100%; overflow-x: auto; }\n",
                   encoding="utf-8")
    # Emit MathML explicitly while retaining its parsed source annotation.
    # This avoids relying on LaTeXML's default format selection when XMath is
    # retained, and keeps the preview's mathematical source inspectable.
    run(post + ["--format=html5", "--presentationmathml", "--keepXMath", f"--sourcedirectory={source}",
                f"--css={css}", f"--destination={preview / 'index.html'}", str(web_path)], "postprocess")

    check = HTMLCheck()
    html = (preview / "index.html").read_text(encoding="utf-8")
    # Discard indentation on empty lines without altering text or MathML nodes.
    html = re.sub(r"(?m)^[ \t]+$", "", html)
    (preview / "index.html").write_text(html, encoding="utf-8", newline="\n")
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
    equation_numbers = check_equation_numbers(check, numbered)
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
    date = re.search(r"\\date\{([^}]+)\}", tex).group(1)
    assert all(name in visible for name in ("Ratha Siv", "Matei Mancas", "Bernard Gosselin", "University of Mons, Belgium")) and date in visible
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
    (output / "latexml.xml").write_text(portable_xml_metadata(xml_text, work),
                                       encoding="utf-8", newline="\n")
    report = {
        "archive_sha256": sha256(archive),
        "converter": run(latexml + ["--VERSION"], "version").strip(),
        "source": "Unmodified main.tex and main.bbl extracted from the upload ZIP",
        "figure_rendering": "Original PDF figures rendered by Poppler for the local HTML preview only",
        "bibliography": "Supplied main.bbl via LaTeXML nobibtex; preview-only numeric refnum labels use the converter's number tags to correct its natbib numbers/number bug",
        "bibliography_order": bbl_keys,
        "equation_numbering": "Preview-only LaTeXML binding routes display delimiters through equation* so explicit source tags remain semantic XML and visible HTML labels",
        "numbered_equations": equation_numbers,
        "all_seven_equation_numbers_visible": True,
        "equation_binding_sha256": sha256(equation_binding),
        "xml": "output/arxiv/latexml.xml",
        "xml_sha256": sha256(output / "latexml.xml"),
        "source_comment_whitespace_normalized": True,
        "local_build_paths_redacted": True,
        "counts": actual, "diagnostics": diagnostics,
        "parsed_math_expressions_including_alternatives": len(math_nodes),
        "font_settings_leaked": False, "all_internal_links_resolve": True,
        "all_figures_have_alt_text": True,
        "preview": "output/arxiv/html/index.html",
        "preview_sha256": sha256(output / "html" / "index.html"),
        "build_directory": work.relative_to(PAPER.parent).as_posix(), "commands": commands,
        "scope": "Local LaTeXML validation; arXiv's deployed version and styling may differ.",
    }
    (output / "html_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(actual))
    print("HTML verified:", output / "html" / "index.html")


if __name__ == "__main__":
    main()
