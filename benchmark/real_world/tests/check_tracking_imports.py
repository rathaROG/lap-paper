"""Import the pinned tracker closure without constructing/updating/solving."""
from __future__ import annotations

import argparse
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[1]


def check(pyppbox_root=None, require_headless=False):
    class NoDetector(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "rfdetr", "torchvision"}:
                raise ModuleNotFoundError("Detector dependency deliberately unavailable", name=fullname)

    sys.meta_path.insert(0, NoDetector())
    import numpy
    import yaml
    import cv2
    import scipy.optimize
    import laprust

    calls = {"assignment": 0}

    def forbidden(*args, **kwargs):
        calls["assignment"] += 1
        raise AssertionError("Assignment calls are forbidden in this import-only check")

    scipy.optimize.linear_sum_assignment = forbidden
    laprust.lapjv = forbidden
    sys.path.insert(0, str(PACKAGE))
    from realbench_paths import configure_pyppbox, pyppbox_module
    configure_pyppbox(pyppbox_root)
    module = pyppbox_module()
    from pyppbox.utils.persontools import Person
    from pyppbox.modules.trackers.centroid import MyCentroid
    from pyppbox.modules.trackers.sort import MySORT
    flavors = {}
    for name in ("opencv-python", "opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"):
        try:
            flavors[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if require_headless and set(flavors) != {"opencv-python-headless"}:
        raise RuntimeError(f"Expected only the CPU headless OpenCV distribution, found {flavors}")
    assert calls["assignment"] == 0
    assert not ({"torch", "rfdetr", "torchvision"} & set(sys.modules))
    return {"passed": True, "imported_symbols": [Person.__name__, MyCentroid.__name__, MySORT.__name__],
            "pyppbox_version": module.__version__, "numpy_version": numpy.__version__,
            "PyYAML_version": yaml.__version__, "cv2_version": cv2.__version__,
            "opencv_distributions": flavors, "assignment_calls": calls["assignment"],
            "tracker_instances_constructed": 0, "frames_processed": 0, "GPU_detector_imports": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyppbox-root", type=Path)
    parser.add_argument("--require-headless", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.pyppbox_root, args.require_headless), indent=2))
