"""Build original and specialized LSAPE together with MSVC, the running Python and NumPy.

The paper's measurements used this MSVC x64 build (flags below). The toolchain
is located through VCVARS64 (path to vcvars64.bat), vswhere, or a standard
Visual Studio 2022 installation. Windows x64 only; other platforms are not
provided because they would not reproduce the measured configuration.

No installation, package resolution, solver execution, tests, or timing occurs.
Upstream source and its CeCILL-C attribution are left unchanged.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import sysconfig
import shutil
from datetime import datetime, timezone

from redaction import redact


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_vcvars() -> Path:
    configured = os.environ.get("VCVARS64")
    if configured:
        return Path(configured)
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if vswhere.is_file():
        found = subprocess.run([str(vswhere), "-latest", "-products", "*", "-requires",
                                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                                "installationPath"], capture_output=True, text=True, check=False).stdout.strip()
        if found:
            candidate = Path(found.splitlines()[0]) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if candidate.is_file():
                return candidate
    for edition in ("Enterprise", "Professional", "Community", "BuildTools"):
        candidate = (Path(r"C:\Program Files\Microsoft Visual Studio\2022") / edition
                     / "VC" / "Auxiliary" / "Build" / "vcvars64.bat")
        if candidate.is_file():
            return candidate
    raise SystemExit("MSVC x64 toolchain not found; set VCVARS64 to the full path of vcvars64.bat")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Build directory; defaults to this experiment's build/")
    options = parser.parse_args()
    if os.name != "nt" or struct.calcsize("P") != 8:
        raise SystemExit("This bridge builds with MSVC for 64-bit Python on Windows only.")
    import numpy as np

    root = Path(__file__).resolve().parent
    source = root / "lsape_bridge.cpp"
    archive = root / "src" / "lsape-v4.zip"
    include = root / "src" / "upstream-v4" / "lsape" / "include"
    header = include / "hungarian-lsape.hh"
    specialized = root / "src" / "specialized" / "hungarian-lsape-uniform.hh"
    vcvars = find_vcvars()
    py_include = Path(sysconfig.get_path("include"))
    py_libs = Path(sys.base_prefix) / "libs"
    python_lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
    python_lib = py_libs / python_lib_name
    numpy_include = Path(np.get_include())
    for path in (source, archive, header, specialized, vcvars, py_include / "Python.h", python_lib,
                 numpy_include / "numpy" / "arrayobject.h"):
        if not path.is_file():
            raise SystemExit(redact(f"Required existing build input is missing: {path}"))

    build = (options.output_dir or root / "build").resolve()
    build.mkdir(parents=True, exist_ok=True)
    output = build / "_lsape_native.pyd"
    obj = build / "lsape_bridge.obj"
    flags = ["/nologo", "/O2", "/EHsc", "/std:c++17", "/LD", "/MD", "/DNDEBUG", "/fp:precise", "/Bv"]
    command = ["cl.exe", *flags, f"/I{py_include}", f"/I{numpy_include}",
               f"/I{include}", str(source), f"/Fo{obj}", f"/Fe{output}",
               "/link", f"/LIBPATH:{py_libs}", python_lib_name,
               f"/IMPLIB:{build / '_lsape_native.lib'}"]
    # Paths are fixed local inputs, quoted by Windows' argument quoting rules.
    # The batch file exists only to inherit vcvars64's current default toolchain.
    # No -vcvars_ver override selects an older compiler.
    command_text = subprocess.list2cmdline(command)
    batch = build / "compile_bridge.cmd"
    batch.write_text(
        "@echo off\n"
        f'call "{vcvars}"\n'
        "if errorlevel 1 exit /b %errorlevel%\n"
        f"{command_text}\n"
        "exit /b %errorlevel%\n",
        encoding="utf-8",
    )
    invocation = [os.environ.get("COMSPEC", r"C:\Windows\System32\cmd.exe"),
                  "/d", "/c", str(batch)]
    record = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "vcvars64": str(vcvars),
        "toolchain_selection": "vcvars64 default x64 compiler; no version override",
        "floating_point_contract": "IEC559 binary64 round-to-nearest; /fp:precise; no /fp:fast or /fp:contract",
        "compiler_flags": flags,
        "compiler_command": command,
        "compiler_command_text": command_text,
        "invocation": invocation,
        "input_sha256": {
            "lsape_bridge.cpp": sha256(source),
            "build_bridge.py": sha256(Path(__file__).resolve()),
            "lsape-v4.zip": sha256(archive),
            "hungarian-lsape.hh": sha256(header),
            "src/specialized/hungarian-lsape-uniform.hh": sha256(specialized),
        },
        "output": str(output),
        "build_log": str(build / "build.log"),
    }
    try:
        result = subprocess.run(invocation, cwd=build, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", check=False)
    except OSError as error:
        raise SystemExit(redact(f"Unable to launch compiler: {error}")) from None
    finally:
        # The temporary launcher contains real paths while running. Scrub it even
        # if launching the compiler fails, before leaving it as a build artifact.
        batch.write_text(redact(batch.read_text(encoding="utf-8")), encoding="utf-8")
    # Saved build records name locations by role (<lsape>, <python>, ...), never by local path.
    (build / "build.log").write_text(redact(result.stdout), encoding="utf-8")
    record["returncode"] = result.returncode
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["output_sha256"] = sha256(output) if result.returncode == 0 and output.is_file() else None
    (build / "provenance.json").write_text(json.dumps(redact(record), indent=2) + "\n", encoding="utf-8")
    snapshot = build / "history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    snapshot.mkdir(parents=True, exist_ok=False)
    for input_path in (source, Path(__file__).resolve(), header, specialized, archive,
                       root / "verify_implementation.py", root / "README.md"):
        if input_path.is_file():
            destination = snapshot / "sources" / input_path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, destination)
    for artifact in (build / "provenance.json", build / "build.log", batch):
        shutil.copy2(artifact, snapshot / artifact.name)
    if result.returncode == 0 and output.is_file():
        shutil.copy2(output, snapshot / output.name)
    print(redact(result.stdout))
    print(redact(f"Build provenance: {build / 'provenance.json'}"))
    print(redact(f"Immutable build snapshot: {snapshot}"))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
