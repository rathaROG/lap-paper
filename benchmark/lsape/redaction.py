"""Keep local absolute paths out of saved records (standard library only).

Known roots become placeholders (<lsape>, <python>, <site-packages>, <tmp>, ~,
<cwd>); any other absolute path becomes <external>/<name>. System locations are redacted too. Hashes still identify every file.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import site
import sys
import sysconfig
import tempfile

REDACT_ROOT = (Path(__file__).resolve().parent, "<lsape>")
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
