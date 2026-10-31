"""C++ counters for separately named diagnostic artifacts, never primary timing.

Apply AFTER native_build.patch_source to an isolated, pinned source archive
whose extension namespace is campaign_diag_cpp_<arm>. This helper edits only
solver.cpp and generates a C ABI header. The diagnostic builder supplies Cython
exports and records source/binary hashes. Reset/snapshot require all solver
calls to be quiescent; counters aggregate parallel workers with relaxed atomics.

Counts 0--5 cover the rectangular core (including square OPTIONAL input).
Required square solving is deliberately uncounted there. A row scan means one
initializer row-minimum scan or one completion-loop row-label relaxation; the
cached initial root labels do not add a second search scan. Initialization
chains start at a pass's next free row, excluding immediate displaced-row retry.

Capacity maxima are ACTUAL vector capacity * sizeof(T), not requested payload.
Cleanup events include stock release_large_buffers and W2 destruction. To
observe W2's after-cleanup capacity while its vector remains a valid object,
diagnostic scope exit explicitly swaps that orientation vector with an empty
vector, then records its capacity; ordinary destruction follows. W1 restores
orientation to the retained worker and drops only fresh solver state. This
scope-exit instrumentation is absent from every primary artifact.
"""
from __future__ import annotations

import difflib
from pathlib import Path
import re


FIELD_NAMES = (
    "initialization_row_scans", "search_row_scans", "initialization_displacements",
    "initialization_chain_starts", "augmentations", "rejection_endings",
    "orientation_capacity_before_solve_max_bytes", "orientation_capacity_after_solve_max_bytes",
    "orientation_capacity_before_cleanup_max_bytes", "orientation_capacity_after_cleanup_max_bytes",
    "orientation_cleanup_calls", "orientation_releases", "problems",
    "orientation_nonzero_before_solve_count", "orientation_nonzero_after_cleanup_count",
    "fresh_state_drops", "fresh_worker_drops",
)


HEADER = '''// Diagnostic-only ABI; never included in primary timing builds.
#pragma once
#ifdef __cplusplus
extern "C" {
#endif
void campaign_diag_reset(void);
void campaign_diag_snapshot(unsigned long long* out);
#ifdef __cplusplus
}
#endif
'''


COUNTERS = '''// Diagnostic-only atomics; callers reset/snapshot outside active solves.
namespace campaign_diag {
constexpr size_t FIELD_COUNT = 17;
static std::atomic<unsigned long long> fields[FIELD_COUNT]{};
inline void add(size_t field, unsigned long long value = 1) {
    fields[field].fetch_add(value, std::memory_order_relaxed);
}
inline void maximum(size_t field, unsigned long long value) {
    auto previous = fields[field].load(std::memory_order_relaxed);
    while (value > previous && !fields[field].compare_exchange_weak(
            previous, value, std::memory_order_relaxed)) {}
}
inline void cleanup_before(size_t bytes) {
    add(10);
    maximum(8, static_cast<unsigned long long>(bytes));
}
inline void cleanup_after(size_t before, size_t after) {
    maximum(9, static_cast<unsigned long long>(after));
    if (before && !after) add(11);
    if (after) add(14);
}
} // namespace campaign_diag

extern "C" void campaign_diag_reset(void) {
    for (size_t i = 0; i < campaign_diag::FIELD_COUNT; ++i)
        campaign_diag::fields[i].store(0, std::memory_order_relaxed);
}
extern "C" void campaign_diag_snapshot(unsigned long long* out) {
    for (size_t i = 0; i < campaign_diag::FIELD_COUNT; ++i)
        out[i] = campaign_diag::fields[i].load(std::memory_order_relaxed);
}

'''


SOLVE_SAMPLE = '''// Observe the effective W0/W1/W2 worker, not a dormant retained worker.
template<class T> struct DiagnosticSolveSample {
    Worker<T>& value;
    bool destroy_orientation;
    explicit DiagnosticSolveSample(Worker<T>& w, bool destroy = false)
        : value(w), destroy_orientation(destroy) {
        const size_t bytes = value.orientation.capacity()*sizeof(T);
        campaign_diag::add(12);
        campaign_diag::maximum(6, static_cast<unsigned long long>(bytes));
        if (bytes) campaign_diag::add(13);
    }
    ~DiagnosticSolveSample() {
        const size_t bytes = value.orientation.capacity()*sizeof(T);
        campaign_diag::maximum(7, static_cast<unsigned long long>(bytes));
        if (destroy_orientation) {
            // The W2 worker is about to be destroyed; observe its actual release.
            campaign_diag::cleanup_before(bytes);
            std::vector<T>().swap(value.orientation);
            campaign_diag::cleanup_after(bytes, value.orientation.capacity()*sizeof(T));
            campaign_diag::add(16);
        }
    }
};
'''


def _replace(text: str, before: str, after: str, count: int = 1) -> str:
    found = text.count(before)
    if found != count:
        raise ValueError(f"Diagnostic C++ source guard expected {count} occurrences, found {found}: {before[:90]!r}")
    return text.replace(before, after)


def instrument_text(text: str) -> str:
    """Pure source transformation, guarded before any file is written."""
    if "campaign_diag" in text:
        raise ValueError("C++ source is already instrumented")
    text = _replace(text, '#include "campaign_policy.h"',
                    '#include "campaign_policy.h"\n#include "campaign_diag.h"')
    text = _replace(text, "namespace {\n", COUNTERS + "namespace {\n")
    start = text.index("template<class T> struct Rectangular {")
    end = text.index("template<class T> struct Worker {")
    rectangular = text[start:end]
    rectangular = _replace(rectangular,
        "            size_t k = 0, reductions = 0;\n            while (k < previous) {",
        "            size_t k = 0, reductions = 0;\n"
        "            bool diagnostic_continuation = false;\n"
        "            while (k < previous) {\n"
        "                if (!diagnostic_continuation) campaign_diag::add(3);\n"
        "                diagnostic_continuation = false;\n"
        "                campaign_diag::add(0);")
    rectangular = _replace(rectangular,
        "if (Optional && j1 == FREE) { xx[i] = FREE; u[i] = penalty; stalled = 0; continue; }",
        "if (Optional && j1 == FREE) { campaign_diag::add(5); "
        "xx[i] = FREE; u[i] = penalty; stalled = 0; continue; }", count=2)
    rectangular = _replace(rectangular,
        "                if (displaced != FREE) {\n                    if (lowers) free[--k] = displaced;",
        "                if (displaced != FREE) {\n"
        "                    campaign_diag::add(2);\n"
        "                    if (lowers) { free[--k] = displaced; diagnostic_continuation = true; }")
    rectangular = _replace(rectangular,
        "        for (size_t f = 0; f < num_free; ++f) {",
        "        for (size_t f = 0; f < num_free; ++f) {\n            campaign_diag::add(4);")
    rectangular = _replace(rectangular, "            for (;;) {",
        "            for (;;) {\n                campaign_diag::add(1);")
    rectangular = _replace(rectangular,
        "                if (Optional && exact_finite(dummy_distance) && dummy_distance <= best) {",
        "                if (Optional && exact_finite(dummy_distance) && dummy_distance <= best) {\n"
        "                    campaign_diag::add(5);")
    text = text[:start] + rectangular + text[end:]
    text = _replace(text,
        "    void release_large_buffers() {\n        if (orientation.capacity()",
        "    void release_large_buffers() {\n"
        "        const size_t diagnostic_before = orientation.capacity()*sizeof(T);\n"
        "        campaign_diag::cleanup_before(diagnostic_before);\n"
        "        if (orientation.capacity()")
    text = _replace(text,
        "        if (rectangular.y.capacity() > 65536) rectangular = Rectangular<T>{};\n    }",
        "        if (rectangular.y.capacity() > 65536) rectangular = Rectangular<T>{};\n"
        "        campaign_diag::cleanup_after(diagnostic_before, orientation.capacity()*sizeof(T));\n    }")
    text = _replace(text, "template<class T> Worker<T>& worker()",
                    SOLVE_SAMPLE + "template<class T> Worker<T>& worker()")
    text = _replace(text,
        "            ~RestoreOrientation() { retained.swap(fresh); }",
        "            ~RestoreOrientation() { retained.swap(fresh); campaign_diag::add(15); }")
    text = _replace(text,
        "        } restore{retained.orientation, fresh.orientation};\n        solve_one_inner(fresh,",
        "        } restore{retained.orientation, fresh.orientation};\n"
        "        DiagnosticSolveSample<T> diagnostic_sample(fresh);\n        solve_one_inner(fresh,")
    text = _replace(text,
        "    } else if constexpr (campaign_policy::workspace == 2) {\n        Worker<T> fresh;\n        solve_one_inner(fresh,",
        "    } else if constexpr (campaign_policy::workspace == 2) {\n        Worker<T> fresh;\n"
        "        DiagnosticSolveSample<T> diagnostic_sample(fresh, true);\n        solve_one_inner(fresh,")
    text = _replace(text,
        "    } else {\n        solve_one_inner(retained,",
        "    } else {\n        DiagnosticSolveSample<T> diagnostic_sample(retained);\n        solve_one_inner(retained,")
    return text


def patch_cpp(root: Path) -> dict[str, str]:
    """Instrument an explicitly named diagnostic copy; reject primary namespaces."""
    root = Path(root)
    setup = (root / "setup.py").read_text(encoding="utf-8")
    namespaces = set(re.findall(r"(campaign_diag_cpp_[a-z0-9_]+)\._lapjv", setup))
    if len(namespaces) != 1 or not (root / next(iter(namespaces))).is_dir():
        raise ValueError("C++ diagnostics require an isolated campaign_diag_cpp_<arm> package")
    relative = "src/_lapjv/solver.cpp"
    path = root / relative
    before = path.read_text(encoding="utf-8")
    after = instrument_text(before)
    header = root / "src/_lapjv/campaign_diag.h"
    if header.exists():
        raise ValueError("Diagnostic header already exists; use a fresh isolated source copy")
    path.write_text(after, encoding="utf-8", newline="\n")
    header.write_text(HEADER, encoding="utf-8", newline="\n")
    diff = "".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                                     fromfile="controlled/"+relative, tofile="diagnostic/"+relative))
    return {relative: diff, "src/_lapjv/campaign_diag.h": HEADER}
