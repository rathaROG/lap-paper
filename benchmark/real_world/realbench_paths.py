"""Portable artifact paths and the optional pinned pyppbox dependency."""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import site
import sys
import sysconfig
import tempfile
from uuid import uuid4

PACKAGE = Path(__file__).resolve().parent
# Provenance only: the published collection the bundled caches and matrices come from.
COLLECTION_NAME = "real-world-python-3.11.17-20261027-lapx-0.12.0"
_pyppbox_root = None
_validated_module = None

# Saved records never contain local absolute paths: known roots become placeholders
# (<real_world>, <python>, <site-packages>, <tmp>, ~, <cwd>), any other absolute path
# becomes <external>/<name>. Hashes identify files.
REDACT_ROOT = (PACKAGE, "<real_world>")
# URLs are protected before path matching, including path-like query values.
_URL = re.compile(r"(?:https?|ftp)://[^\s\"'<>]+", re.IGNORECASE)
# A directory component may contain spaces; the unquoted final component stops
# at whitespace so surrounding prose is retained. Quoted/whole paths handle
# filenames with spaces separately.
_COMPONENT = r"[^\s\"'<>|:\\/]+(?:[ \t]+[^\s\"'<>|:\\/]+)*"
_WINDOWS_ABS = re.compile(r"(?:(?<![A-Za-z0-9])|(?<=/I)|(?<=/Fo)|(?<=/Fe)|(?<=/LIBPATH:))"
                          r"[A-Za-z]:[\\/](?:" + _COMPONENT + r"[\\/])*[^\s\"'<>|\\/]*", re.IGNORECASE)
_QUOTED_ABS = re.compile(r"(?P<quote>[\"'])(?P<path>(?:[A-Za-z]:[\\/]|\\\\|/)[^\"'\r\n]*)(?P=quote)")
_WHOLE_WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/][^\"'<>|\r\n]*$")
_WHOLE_UNC_ABS = re.compile(r"^\\\\[^\\/\s]+[\\/][^\"'<>|\r\n]*$")
_WHOLE_POSIX_ABS = re.compile(r"^/(?:[^/\r\n]+/)+[^\r\n]*$")
_UNC_ABS = re.compile(r"(?<![\\\w])\\\\[^\\/\s\"'<>|]+[\\/](?:" + _COMPONENT + r"[\\/])*[^\s\"'<>|\\/]*")
_POSIX_ABS = re.compile(r"(?:(?<![\w:/.>~-])|(?<=-I)|(?<=-L))/(?:(?:" + _COMPONENT + r")/)+[^\s\"'<>/]*")
_REDACT_ROOTS = None


def _redaction_roots():
    global _REDACT_ROOTS
    if _REDACT_ROOTS is None:
        sites = list(getattr(site, "getsitepackages", lambda: [])()) + [site.getusersitepackages()]
        sites += [sysconfig.get_paths().get(key) for key in ("purelib", "platlib")]
        # Later entries win for identical paths: Windows lists sys.prefix as a site directory.
        candidates = [(os.getcwd(), "<cwd>"), (Path.home(), "~"), (tempfile.gettempdir(), "<tmp>"),
                      *((p, "<site-packages>") for p in sites if p),
                      *((p, "<python>") for p in (sys.prefix, sys.base_prefix, sys.exec_prefix)), REDACT_ROOT]
        chosen = {}
        for path, placeholder in candidates:
            try:
                text = str(Path(path).resolve())
            except (OSError, ValueError):
                continue
            chosen[text.lower() if os.name == "nt" else text] = (text, placeholder)
        roots = []
        for text, placeholder in chosen.values():
            parts = [part for part in re.split(r"[\\/]+", text) if part]
            if len(parts) < 2:
                continue  # never redact a bare drive or "/"
            body = r"[\\/]+".join(re.escape(part) for part in parts)
            lead = "" if re.fullmatch(r"[A-Za-z]:", parts[0]) else r"[\\/]"
            roots.append((len(text), re.compile(lead + body + r"(?=$|[\\/\s\"'])",
                                                re.IGNORECASE if os.name == "nt" else 0), placeholder))
        _REDACT_ROOTS = [(pattern, placeholder) for _, pattern, placeholder in sorted(roots, key=lambda r: -r[0])]
    return _REDACT_ROOTS


def _external_path(value):
    parts = [part for part in re.split(r"[\\/]", value.rstrip("\\/")) if part]
    return "<external>/" + parts[-1] if parts else "<external>"


def _external(match):
    return _external_path(match.group(0))


def redact(value):
    """Replace local paths without changing numeric data or discarding records."""
    if isinstance(value, dict):
        items = [(redact(key) if isinstance(key, str) else key, key, item) for key, item in value.items()]
        counts = {}
        for key, _, _ in items:
            counts[key] = counts.get(key, 0) + 1
        result = {}
        for key, original, item in items:
            if counts[key] > 1 or key in result:
                # Distinct external paths may have the same basename. Keep every
                # record and give its key a stable identity without exposing roots.
                identity = hashlib.sha256(str(original).encode("utf-8")).hexdigest()[:12]
                key = f"{key}#<id:{identity}>"
                while key in result:
                    key += "_"
            result[key] = redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if not isinstance(value, str) or not value:
        return value
    urls = []

    def protect_url(match):
        urls.append(match.group(0))
        return f"<url:{len(urls) - 1}>"

    value = _URL.sub(protect_url, value)
    for pattern, placeholder in _redaction_roots():
        value = pattern.sub(placeholder, value)
    value = _QUOTED_ABS.sub(lambda match: match.group("quote") + _external_path(match.group("path")) + match.group("quote"), value)
    if any(pattern.fullmatch(value) for pattern in (_WHOLE_WINDOWS_ABS, _WHOLE_UNC_ABS, _WHOLE_POSIX_ABS)):
        value = _external_path(value)
    else:
        value = _WINDOWS_ABS.sub(_external, value)
        value = _UNC_ABS.sub(_external, value)
        value = _POSIX_ABS.sub(_external, value)
    for index, url in enumerate(urls):
        value = value.replace(f"<url:{index}>", url)
    return value

def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_manifest():
    return json.loads((PACKAGE / "provenance" / "data_manifest.json").read_text(encoding="utf-8"))


def materialize_inputs():
    """Expand only the four packaged caches into a content-addressed temp cache."""
    manifest_path = PACKAGE / "provenance" / "data_manifest.json"
    manifest = data_manifest()
    destination = Path(tempfile.gettempdir()) / "lap-paper-real-world-inputs" / sha256(manifest_path)
    destination.mkdir(parents=True, exist_ok=True)
    for name, identity in manifest["files"].items():
        if not name.startswith("inputs/"):
            continue
        source = PACKAGE / "data" / name
        if sha256(source) != identity["sha256"]:
            raise ValueError(f"Packaged input checksum mismatch: {source}")
        target = destination / Path(name).name.removesuffix(".gz")
        if target.exists():
            if sha256(target) != identity["uncompressed_sha256"]:
                raise ValueError(f"Expanded input checksum mismatch: {target}")
            continue
        temporary = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        with gzip.open(source, "rb") as reader, temporary.open("xb") as writer:
            while block := reader.read(1024 * 1024):
                writer.write(block)
        if sha256(temporary) != identity["uncompressed_sha256"]:
            raise ValueError(f"Expanded input checksum mismatch: {temporary}")
        temporary.replace(target)
    return destination


def collection_path(explicit=None):
    """Directory with matrices/ (replay) or summary.json (report).

    Without --collection, replay uses the eight captured corpora bundled in
    data/matrices, checked against provenance/data_manifest.json.
    """
    if explicit is None:
        for name, identity in data_manifest()["files"].items():
            if name.startswith("matrices/"):
                path = PACKAGE / "data" / name
                if path.stat().st_size != identity["bytes"] or sha256(path) != identity["sha256"]:
                    raise ValueError(f"Bundled matrix checksum mismatch: {path}")
        return PACKAGE / "data"
    result = Path(explicit).expanduser().resolve(strict=True)
    if (result / "matrices").is_dir() or (result / "summary.json").is_file():
        return result
    raise FileNotFoundError(f"--collection needs a directory with matrices/ or summary.json: {result}")


def configure_pyppbox(root=None):
    global _pyppbox_root
    if root is None:
        root = os.environ.get("LAP_REALWORLD_PYPPBOX_ROOT")
    if root:
        selected = Path(root).expanduser().resolve(strict=True)
        package = selected / "pyppbox" if (selected / "pyppbox" / "__init__.py").is_file() else selected
        if not (package / "__init__.py").is_file():
            raise ValueError(f"Not a pyppbox checkout/package: {selected}")
        if _validated_module is not None and Path(_validated_module.__file__).resolve().parent != package:
            raise RuntimeError("A different pyppbox has already been loaded in this process")
        _pyppbox_root = package
        if str(package.parent) not in sys.path:
            sys.path.insert(0, str(package.parent))


def pyppbox_module():
    """Require the measured tracker implementation, installed or from checkout."""
    global _validated_module
    if _validated_module is not None:
        return _validated_module
    configure_pyppbox()
    module = importlib.import_module("pyppbox")
    package = Path(module.__file__).resolve().parent
    if _pyppbox_root is not None and package != _pyppbox_root:
        raise RuntimeError(f"Expected pyppbox at {_pyppbox_root}; loaded {package}")
    identity = json.loads((PACKAGE / "provenance" / "pyppbox_identity.json").read_text(encoding="utf-8"))
    if getattr(module, "__version__", None) != identity["version"]:
        raise RuntimeError(f"Need pyppbox {identity['version']}; select the pinned checkout with --pyppbox-root")
    for name, expected in identity["tracker_sha256"].items():
        # A pinned checkout may use CRLF on Windows and LF on other systems.
        normalized = hashlib.sha256((package / name).read_text(encoding="utf-8").encode()).hexdigest()
        if sha256(package / name) != expected and normalized != identity.get("tracker_text_sha256", {}).get(name):
            raise RuntimeError(f"pyppbox tracker source differs: {name}; use commit {identity['git_commit']} with --pyppbox-root")
    _validated_module = module
    return module
