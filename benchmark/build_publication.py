"""Build the authoritative LaTeX source and verify a standalone arXiv archive.

Requires PDFLaTeX and BibTeX on PATH. Does not generate or rewrite main.tex.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
import zipfile

PAPER = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnostics(log):
    return [line for line in log.splitlines() if any(term in line for term in (
        'Overfull', 'Missing character', 'undefined', 'Rerun to get',
        'LaTeX Warning', 'Package natbib Warning', 'Package rerunfilecheck Warning'))]


def main():
    engine, bibtex = shutil.which('pdflatex'), shutil.which('bibtex')
    if not engine or not bibtex:
        raise SystemExit('PDFLaTeX and BibTeX must be on PATH.')
    scratch = PAPER/'tmp/pdfs'
    scratch.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='publication-', dir=scratch))
    source = work/'source'
    source.mkdir()
    tex = (PAPER/'latex/main.tex').read_text(encoding='utf-8')
    images = sorted(set(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex)))
    members = ['main.tex', 'references.bib', 'main.bbl'] + images
    for name in ['main.tex', 'references.bib'] + images:
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name:
            raise ValueError('Unsafe publication source path: '+name)
        original = PAPER/'latex'/name
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    env = os.environ.copy()
    for key in ('TEXINPUTS', 'BIBINPUTS', 'BSTINPUTS', 'TEXMFOUTPUT'):
        env.pop(key, None)
    commands = []
    def run(command, folder, label):
        result = subprocess.run(command, cwd=folder, env=env, text=True,
                                encoding='utf-8', errors='replace',
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (work/(label+'.log')).write_text(result.stdout, encoding='utf-8')
        commands.append({'step': label, 'exit_code': result.returncode})
        if result.returncode:
            raise RuntimeError(label+' failed:\n'+result.stdout[-6000:])
        print(label+' completed.', flush=True)
        return result.stdout
    command = [engine, '-no-shell-escape', '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error', '-recorder', 'main.tex']
    run(command, source, 'source-pass-1')
    run([bibtex, 'main'], source, 'bibliography')
    for index in (2, 3):
        run(command, source, f'source-pass-{index}')
    archive = work/'native_rectangular_assignment_arxiv.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            output.writestr(info, (source/name).read_bytes())
    extracted = work/'extracted'
    with zipfile.ZipFile(archive) as incoming:
        if incoming.testzip() is not None:
            raise ValueError('Source archive CRC failure')
        incoming.extractall(extracted)  # Members above come only from our explicit allowlist.
    for index in (1, 2, 3):
        run(command, extracted, f'archive-pass-{index}')
    issues = diagnostics((extracted/'main.log').read_text(encoding='utf-8', errors='replace'))
    if issues:
        raise RuntimeError('Final PDF diagnostics:\n'+'\n'.join(issues))
    bbl = (extracted/'main.bbl').read_text(encoding='utf-8')
    cited = {key.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex) for key in group.split(',')}
    bibliography = set(re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}', bbl))
    if not cited.issubset(bibliography):
        raise ValueError('Missing bibliography references')
    log = (extracted/'main.log').read_text(encoding='utf-8', errors='replace')
    page_count = int(re.findall(r'Output written on .*?\((\d+) pages?', log)[-1])
    for folder in ('output/pdf', 'output/arxiv'):
        target = PAPER/folder
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted/'main.pdf', target/'native_rectangular_assignment.pdf')
    shutil.copy2(archive, PAPER/'output/arxiv'/archive.name)
    shutil.copy2(extracted/'main.bbl', PAPER/'latex/main.bbl')
    report = {
        'source': 'latex/main.tex', 'source_sha256': sha256(PAPER/'latex/main.tex'),
        'pdf': 'output/pdf/native_rectangular_assignment.pdf',
        'pdf_sha256': sha256(PAPER/'output/pdf/native_rectangular_assignment.pdf'),
        'archive': 'output/arxiv/'+archive.name, 'archive_sha256': sha256(archive),
        'verified_from_extracted_archive': True, 'verification_uses_bibtex': False,
        'shell_escape': False, 'pages': page_count, 'figures': len(images),
        'tables': len(re.findall(r'\\begin\{table\}', tex)), 'references': len(bibliography),
        'files': {name: sha256(extracted/name) for name in sorted(members)},
        'diagnostics': issues, 'commands': commands,
        'compiler': subprocess.check_output([engine, '--version'], text=True).splitlines()[0],
    }
    (PAPER/'output/arxiv/build_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('pages','figures','tables','references','diagnostics')}))


if __name__ == '__main__':
    main()
