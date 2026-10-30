"""Focused HTML equation-label and path regressions; no converter or solver."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from check_publication_html import (HTMLCheck, check_equation_numbers,
                                    numbered_equation_ids, portable_command,
                                    portable_xml_metadata)


def parse_html(source):
    check = HTMLCheck()
    check.feed(source)
    return check


def equation_html(label="1", identifier="eq-one", extra=""):
    return (f'<table id="{identifier}" class="ltx_equation"><tbody><tr><td>'
            f'<span class="ltx_tag ltx_tag_equation" {extra}>({label})</span>'
            '</td><td><math><mi>x</mi></math></td></tr></tbody></table>')


class EquationLabels(unittest.TestCase):
    def test_tags_belong_to_simple_and_aligned_formula_containers(self):
        root = ET.fromstring('''<document xmlns="http://dlmf.nist.gov/LaTeXML">
          <equation xml:id="eq-one"><tags><tag>(1)</tag><tag role="refnum">1</tag></tags><Math/></equation>
          <equationgroup xml:id="eq-two"><tags><tag>(2)</tag><tag role="refnum">2</tag></tags>
            <equation><Math/></equation><equation><Math/></equation></equationgroup></document>''')
        numbered = numbered_equation_ids(root, ["1", "2"])
        # LaTeXML emits an aligned group's number inside its first numbered
        # row. It belongs to the group, rather than that inner row's ID.
        aligned = ('<table id="eq-two" class="ltx_equationgroup"><tbody>'
                   '<tr id="eq-two-row" class="ltx_equation"><td>'
                   '<span class="ltx_tag ltx_tag_equationgroup">(2)</span>'
                   '</td><td><math><mi>y</mi></math></td></tr></tbody></table>')
        html = equation_html() + aligned
        self.assertEqual(check_equation_numbers(parse_html(html), numbered), numbered)

    def test_formula_presence_without_its_number_fails(self):
        html = '<table id="eq-one" class="ltx_equation"><math><mi>x</mi></math></table>'
        with self.assertRaisesRegex(AssertionError, "missing"):
            check_equation_numbers(parse_html(html), {"1": "eq-one"})

    def test_number_in_prose_does_not_replace_an_equation_label(self):
        with self.assertRaisesRegex(AssertionError, "missing"):
            check_equation_numbers(parse_html('<p>Equation (1)</p>'), {"1": "eq-one"})

    def test_duplicate_or_wrong_container_fails(self):
        for html in (equation_html() * 2, equation_html(identifier="different-formula")):
            with self.subTest(html=html), self.assertRaises(AssertionError):
                check_equation_numbers(parse_html(html), {"1": "eq-one"})

    def test_hidden_label_or_hidden_ancestor_fails(self):
        for html in (equation_html(extra='style="display: none"'),
                     '<div hidden>' + equation_html() + '</div>'):
            with self.subTest(html=html), self.assertRaisesRegex(AssertionError, "hidden"):
                check_equation_numbers(parse_html(html), {"1": "eq-one"})

    def test_void_elements_do_not_break_equation_ancestry(self):
        html = '<meta charset="utf-8"><img src="example.png">' + equation_html()
        self.assertEqual(check_equation_numbers(parse_html(html), {"1": "eq-one"}), {"1": "eq-one"})


class PublicationPaths(unittest.TestCase):
    def test_saved_commands_keep_flags_and_filenames_without_private_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper, work = root / "paper", root / "paper" / "tmp" / "build"
            command = [str(root / "tools" / "perl.exe"),
                       str(root / "tools" / "latexml"),
                       "--preload=" + str(work / "paper-equation-tags.ltxml"),
                       "--destination=" + str(work / "main.xml"),
                       str(work / "source" / "images" / "figure.pdf"),
                       str(paper / "benchmark" / "input.json"), "--strict",
                       "https://example.org/converter/help"]
            original = command.copy()
            self.assertEqual(portable_command(command, work, paper), [
                "perl.exe", "latexml", "--preload=[BUILD]/paper-equation-tags.ltxml",
                "--destination=[BUILD]/main.xml", "[BUILD]/source/images/figure.pdf",
                "benchmark/input.json", "--strict", "https://example.org/converter/help"])
            self.assertEqual(command, original)

    def test_published_xml_paths_change_after_converter_use_without_changing_nodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper, work = root / "paper", root / "paper" / "tmp" / "build"
            body = '<document xmlns="http://dlmf.nist.gov/LaTeXML"><Math><mi>x</mi></Math></document>'
            xml = ('<?xml version="1.0"?>\n'
                   f'<?latexml searchpaths="{work / "source"}"?>\n'
                   f'<?latexml package="{work / "paper-equation-tags.ltxml"}"?>\n'
                   '<?latexml package="latexml" options="nobibtex"?>\n' + body)
            cleaned = portable_xml_metadata(xml, work, paper)
            self.assertNotIn(str(root), cleaned)
            self.assertIn('searchpaths="[BUILD]/source"', cleaned)
            self.assertIn('package="[BUILD]/paper-equation-tags.ltxml"', cleaned)
            self.assertIn('package="latexml" options="nobibtex"', cleaned)
            self.assertTrue(cleaned.endswith(body))
            self.assertEqual(ET.tostring(ET.fromstring(xml)),
                             ET.tostring(ET.fromstring(cleaned)))

    def test_portable_metadata_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "build"
            command = ["latexml", "--destination=[BUILD]/main.xml", "main.tex"]
            xml = '<?latexml searchpaths="[BUILD]/source" package="latexml"?><document/>'
            self.assertEqual(portable_command(command, work), command)
            self.assertEqual(portable_xml_metadata(xml, work), xml)


if __name__ == "__main__":
    unittest.main()
