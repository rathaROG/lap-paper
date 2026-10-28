// Shared Python adapter for original LSAPE v4 and its uniform-penalty specialization.
// Upstream: Sebastien Bougleux and Luc Brun, GREYC; CeCILL-C licensed.
// Attribution and the original license remain in src/upstream-v4/lsape/.
// The original entry point always calls the unmodified upstream implementation.
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <climits>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <exception>
#include <new>
#include <vector>
#include <xmmintrin.h>
#include "hungarian-lsape.hh"
#include "src/specialized/hungarian-lsape-uniform.hh"

namespace {
enum class Variant { Original, OneSided, ImplicitScan, Specialized, SpecializedPadded };

const char* variant_name(Variant v) {
    switch (v) {
    case Variant::Original: return "original";
    case Variant::OneSided: return "one_sided";
    case Variant::ImplicitScan: return "implicit_scan";
    case Variant::Specialized: return "specialized";
    default: return "specialized_padded";
    }
}

struct PyRef {
    PyObject* value;
    explicit PyRef(PyObject* p = nullptr) : value(p) {}
    ~PyRef() { Py_XDECREF(value); }
    PyRef(const PyRef&) = delete;
    PyRef& operator=(const PyRef&) = delete;
    PyObject* release() { auto* p = value; value = nullptr; return p; }
};

struct AllowThreads {
    PyThreadState* state;
    AllowThreads() : state(PyEval_SaveThread()) {}
    ~AllowThreads() { PyEval_RestoreThread(state); }
    AllowThreads(const AllowThreads&) = delete;
    AllowThreads& operator=(const AllowThreads&) = delete;
};

PyObject* return_result(PyObject* pairs, bool with_dual, double dual) {
    if (!with_dual) return pairs;
    PyRef owned_pairs(pairs);
    PyRef value(PyFloat_FromDouble(dual));
    if (!value.value) return nullptr;
    return PyTuple_Pack(2, owned_pairs.value, value.value);
}

bool put(PyObject* dict, const char* key, PyObject* value) {
    PyRef owned(value);
    return value && PyDict_SetItemString(dict, key, value) == 0;
}

PyObject* int_array(const std::vector<int>& values) {
    npy_intp shape[1] = {static_cast<npy_intp>(values.size())};
    PyObject* out = PyArray_SimpleNew(1, shape, NPY_INT64);
    if (!out) return nullptr;
    auto* data = static_cast<npy_int64*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)));
    for (size_t i = 0; i < values.size(); ++i) data[i] = values[i];
    return out;
}

PyObject* double_array(const std::vector<double>& values) {
    npy_intp shape[1] = {static_cast<npy_intp>(values.size())};
    PyObject* out = PyArray_SimpleNew(1, shape, NPY_DOUBLE);
    if (out && !values.empty()) std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(out)),
                                          values.data(), values.size() * sizeof(double));
    return out;
}

PyObject* diagnostic_result(PyObject* pairs, const std::vector<int>& rho,
        const std::vector<int>& varrho, const std::vector<double>& u,
        const std::vector<double>& v, double dual, double half, double penalty,
        double magnitude_bound, int ld, Variant variant, bool empty,
        const lsape_experiment::Coverage& stats) {
    PyRef owned_pairs(pairs), out(PyDict_New()), counters(PyDict_New());
    if (!out.value || !counters.value) return nullptr;
    const bool specialized = variant == Variant::ImplicitScan || variant == Variant::Specialized ||
                             variant == Variant::SpecializedPadded;
    const bool no_search = empty || (specialized ? stats.searches == 0 :
        stats.preprocess_unassigned_rows == 0 && stats.preprocess_unassigned_cols == 0);
    if (!put(counters.value, "bridge_empty", PyBool_FromLong(empty)) ||
        !put(counters.value, "no_search", PyBool_FromLong(no_search)) ||
        !put(counters.value, "preprocess_unassigned_rows", PyLong_FromUnsignedLongLong(stats.preprocess_unassigned_rows)) ||
        !put(counters.value, "preprocess_unassigned_cols", PyLong_FromUnsignedLongLong(stats.preprocess_unassigned_cols))) return nullptr;
    const char* keys[] = {"searches", "dual_updates", "rows_reached", "free_column",
        "rejection_row_entry", "rejection_dual_update", "cache_fast_path", "cache_fallback", "scan_rejection"};
    const std::uint64_t counts[] = {stats.searches, stats.dual_updates, stats.rows_reached,
        stats.free_column, stats.rejection_row_entry, stats.rejection_dual_update,
        stats.cache_fast_path, stats.cache_fallback, stats.scan_rejection};
    for (size_t k = 0; k < 9; ++k) {
        PyObject* value;
        if (specialized || empty) value = PyLong_FromUnsignedLongLong(counts[k]);
        else { value = Py_None; Py_INCREF(value); }
        if (!put(counters.value, keys[k], value)) return nullptr;
    }
    if (PyDict_SetItemString(out.value, "pairs", owned_pairs.value) != 0 ||
        PyDict_SetItemString(out.value, "counters", counters.value) != 0 ||
        !put(out.value, "rho", int_array(rho)) || !put(out.value, "varrho", int_array(varrho)) ||
        !put(out.value, "u", double_array(u)) || !put(out.value, "v", double_array(v)) ||
        !put(out.value, "dual_F", PyFloat_FromDouble(dual)) ||
        !put(out.value, "half", PyFloat_FromDouble(half)) ||
        !put(out.value, "penalty", PyFloat_FromDouble(penalty)) ||
        !put(out.value, "magnitude_bound", PyFloat_FromDouble(magnitude_bound)) ||
        !put(out.value, "leading_dimension", PyLong_FromLong(ld)) ||
        !put(out.value, "mxcsr", PyLong_FromUnsignedLong(_mm_getcsr())) ||
        !put(out.value, "variant", PyUnicode_FromString(variant_name(variant)))) return nullptr;
    return out.release();
}

template<bool DIAGNOSTIC = false>
PyObject* solve_impl(PyObject* args, PyObject* kwargs, Variant variant) {
    PyObject* input = nullptr;
    double limit = 0;
    int with_dual = 0;
    static const char* names[] = {"C", "L", "return_dual", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|p:solve",
            const_cast<char**>(names), &input, &limit, &with_dual)) return nullptr;
    if (!std::isfinite(limit) || limit < 0) {
        PyErr_SetString(PyExc_ValueError, "L must be finite and nonnegative in this experiment");
        return nullptr;
    }
    if (std::fegetround() != FE_TONEAREST || (_mm_getcsr() & 0x8040u) != 0) {
        PyErr_SetString(PyExc_ValueError, "This experiment requires round-to-nearest and gradual underflow (FTZ/DAZ disabled)");
        return nullptr;
    }

    // Reject complex/object/string costs before a float64 conversion could hide them.
    PyRef raw(PyArray_FromAny(input, nullptr, 2, 2, NPY_ARRAY_ENSUREARRAY, nullptr));
    if (!raw.value) return nullptr;
    auto* raw_array = reinterpret_cast<PyArrayObject*>(raw.value);
    if (!PyArray_ISNUMBER(raw_array) || PyArray_ISCOMPLEX(raw_array)) {
        PyErr_SetString(PyExc_TypeError, "C must be a two-dimensional real numeric array");
        return nullptr;
    }
    PyRef converted(PyArray_FromAny(raw.value, PyArray_DescrFromType(NPY_DOUBLE), 2, 2,
        NPY_ARRAY_CARRAY_RO | NPY_ARRAY_FORCECAST | NPY_ARRAY_ENSUREARRAY, nullptr));
    if (!converted.value) return nullptr;
    auto* array = reinterpret_cast<PyArrayObject*>(converted.value);
    const npy_intp n_input = PyArray_DIM(array, 0);
    const npy_intp m_input = PyArray_DIM(array, 1);
    if (n_input > INT_MAX - 1 || m_input > INT_MAX - 1 ||
        static_cast<unsigned long long>(n_input + 1) *
            static_cast<unsigned long long>(m_input + 1) > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError, "Augmented dimensions exceed upstream signed-int indexing");
        return nullptr;
    }
    const int n = static_cast<int>(n_input), m = static_cast<int>(m_input);
    const int nr = n + 1, nc = m + 1;
    const double half = limit / 2.0;
    // The original implementation stores the rounded endpoint price, half.
    // Use its effective two-endpoint price even when L/2 is subnormal or zero.
    const double penalty = 2.0 * half;
    const double magnitude_bound = std::numeric_limits<double>::max() /
        (16.0 * (static_cast<double>(n) + m + 1.0));
    if (limit > magnitude_bound) {
        PyErr_SetString(PyExc_ValueError, "L exceeds DBL_MAX/[16*(n+m+1)], the common magnitude guard");
        return nullptr;
    }
    const bool compact = variant == Variant::ImplicitScan || variant == Variant::Specialized;
    const bool specialized = compact || variant == Variant::SpecializedPadded;
    const int stride = compact ? n : nr;
    try {
    if (n == 0 || m == 0) {
        npy_intp shape[2] = {0, 2};
        PyRef pairs(PyArray_SimpleNew(2, shape, NPY_INT64));
        if (!pairs.value) return nullptr;
        if constexpr (DIAGNOSTIC) {
            std::vector<int> rho(n, m), varrho(m, n);
            std::vector<double> u(nr, 0), v(nc, 0);
            for (int i = 0; i < n; ++i) u[i] = variant == Variant::Original ? half : penalty;
            for (int j = 0; j < m; ++j) v[j] = variant == Variant::Original ? half : 0;
            lsape_experiment::Coverage stats;
            return diagnostic_result(pairs.release(), rho, varrho, u, v,
                half * (static_cast<double>(n) + m), half, penalty, magnitude_bound,
                stride, variant, true, stats);
        }
        return return_result(pairs.release(), with_dual != 0, half * (static_cast<double>(n) + m));
    }

        const auto* costs = static_cast<const double*>(PyArray_DATA(array));
        // Identical validation for every arm, separate from representation creation.
        for (size_t index = 0; index < static_cast<size_t>(n) * m; ++index) {
            if (!std::isfinite(costs[index]) || costs[index] < 0 || costs[index] > magnitude_bound) {
                PyErr_SetString(PyExc_ValueError,
                    "C must be finite, nonnegative, and <= DBL_MAX/[16*(n+m+1)]");
                return nullptr;
            }
        }
        // Both algorithms receive column-major costs and perform the same layout
        // conversion here. Compact arms change both border storage AND stride.
        // The padded specialization uses exactly the one-sided upstream buffer.
        // Neither arm transposes the assignment problem or borrows the NumPy input.
        const int stored_cols = compact ? m : nc;
        std::vector<double> edit(static_cast<size_t>(stride) * stored_cols,
                                 variant == Variant::Original ? half : 0.0);
        for (int j = 0; j < m; ++j) {
            for (int i = 0; i < n; ++i) {
                edit[static_cast<size_t>(j) * stride + i] = costs[static_cast<size_t>(i) * m + j];
            }
        }
        if (!compact) {
            if (variant != Variant::Original) {
                for (int i = 0; i < n; ++i) edit[static_cast<size_t>(m) * nr + i] = penalty;
            }
            edit[static_cast<size_t>(m) * nr + n] = 0;
        }
        std::vector<int> rho(n), varrho(m);
        std::vector<double> u(nr), v(nc);
        lsape_experiment::Coverage stats;
        if constexpr (DIAGNOSTIC) {
            if (!specialized) {
                // Exact upstream preprocessing, diagnostic only. The real solve
                // below still calls the unmodified released entry point.
                int nass = 0, mass = 0;
                basicPreprocColsLSAPE<double, int>(edit.data(), nr, nc, rho.data(), varrho.data(),
                                                   u.data(), v.data(), nass, mass);
                stats.preprocess_unassigned_rows = n - nass;
                stats.preprocess_unassigned_cols = m - mass;
            }
        }
        {
            AllowThreads unlocked;
            if (specialized) {
                lsape_experiment::hungarianUniform<double, int, DIAGNOSTIC>(
                    edit.data(), n, m, stride, penalty, rho.data(), varrho.data(), u.data(), v.data(),
                    variant != Variant::ImplicitScan, DIAGNOSTIC ? &stats : nullptr);
            } else {
                hungarianLSAPE<double, int>(edit.data(), nr, nc, rho.data(), varrho.data(),
                                            u.data(), v.data(), 1, false);
            }
        }

        npy_intp count = 0;
        for (int i = 0; i < n; ++i) {
            const int j = rho[i];
            if (j < 0 || j > m || (j < m && varrho[j] != i)) {
                PyErr_SetString(PyExc_RuntimeError, "LSAPE returned an inconsistent row mapping");
                return nullptr;
            }
            if (j < m) ++count;
        }
        for (int j = 0; j < m; ++j) {
            const int i = varrho[j];
            if (i < 0 || i > n || (i < n && rho[i] != j)) {
                PyErr_SetString(PyExc_RuntimeError, "LSAPE returned an inconsistent column mapping");
                return nullptr;
            }
        }
        npy_intp shape[2] = {count, 2};
        PyRef pairs(PyArray_SimpleNew(2, shape, NPY_INT64));
        if (!pairs.value) return nullptr;
        auto* result = static_cast<npy_int64*>(PyArray_DATA(
            reinterpret_cast<PyArrayObject*>(pairs.value)));
        npy_intp k = 0;
        for (int i = 0; i < n; ++i) {
            if (rho[i] < m) {
                result[2 * k] = i;
                result[2 * k + 1] = rho[i];
                ++k;
            }
        }
        // Null potentials are zero. Restore F = P + half*(m-n) for one-sided arms.
        // Only null indices are removed; real equality-cost ties remain valid pairs.
        double dual = 0;
        if (with_dual || DIAGNOSTIC) {
            for (int i = 0; i < n; ++i) dual += u[i];
            for (int j = 0; j < m; ++j) dual += v[j];
            if (variant != Variant::Original) dual += half * (static_cast<double>(m) - n);
        }
        if constexpr (DIAGNOSTIC)
            return diagnostic_result(pairs.release(), rho, varrho, u, v, dual, half, penalty,
                                     magnitude_bound, stride, variant, false, stats);
        return return_result(pairs.release(), with_dual != 0, dual);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected native LSAPE exception");
        return nullptr;
    }
}

PyObject* solve(PyObject*, PyObject* args, PyObject* kwargs) {
    return solve_impl(args, kwargs, Variant::Original);
}
PyObject* solve_one_sided(PyObject*, PyObject* args, PyObject* kwargs) {
    return solve_impl(args, kwargs, Variant::OneSided);
}
PyObject* solve_implicit_scan(PyObject*, PyObject* args, PyObject* kwargs) {
    return solve_impl(args, kwargs, Variant::ImplicitScan);
}
PyObject* solve_specialized(PyObject*, PyObject* args, PyObject* kwargs) {
    return solve_impl(args, kwargs, Variant::Specialized);
}
PyObject* solve_specialized_padded(PyObject*, PyObject* args, PyObject* kwargs) {
    return solve_impl(args, kwargs, Variant::SpecializedPadded);
}
PyObject* diagnose(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* input = nullptr;
    double limit = 0;
    const char* name = "original";
    static const char* names[] = {"C", "L", "variant", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Od|s:diagnose", const_cast<char**>(names),
                                    &input, &limit, &name)) return nullptr;
    Variant selected;
    if (std::strcmp(name, "original") == 0) selected = Variant::Original;
    else if (std::strcmp(name, "one_sided") == 0) selected = Variant::OneSided;
    else if (std::strcmp(name, "implicit_scan") == 0) selected = Variant::ImplicitScan;
    else if (std::strcmp(name, "specialized") == 0) selected = Variant::Specialized;
    else if (std::strcmp(name, "specialized_padded") == 0) selected = Variant::SpecializedPadded;
    else { PyErr_SetString(PyExc_ValueError, "Unknown diagnostic variant"); return nullptr; }
    PyRef forwarded(Py_BuildValue("(Od)", input, limit));
    if (!forwarded.value) return nullptr;
    return solve_impl<true>(forwarded.value, nullptr, selected);
}

PyMethodDef methods[] = {
    {"solve", reinterpret_cast<PyCFunction>(solve), METH_VARARGS | METH_KEYWORDS,
     "solve(C, L, return_dual=False) -> int64[K,2] pairs, or (pairs, dual_F).\n"
     "Calls unmodified LSAPE v4 Hungarian code. Finite nonnegative costs/limit only.\n"
     "Each unmatched endpoint costs L/2. Required, forbidden, and signed cases are excluded.\n"
     "Input conversion, edit-matrix creation, workspace, and output allocation are included."},
    {"solve_one_sided", reinterpret_cast<PyCFunction>(solve_one_sided), METH_VARARGS | METH_KEYWORDS,
     "solve_one_sided(C, L, return_dual=False): unchanged upstream with deletion 2*(L/2), insertion 0.\n"
     "Intermediate control; same result contract and reconstructed dual_F as solve."},
    {"solve_implicit_scan", reinterpret_cast<PyCFunction>(solve_implicit_scan), METH_VARARGS | METH_KEYWORDS,
     "solve_implicit_scan(C, L, return_dual=False): uniform Hungarian with implicit penalty border.\n"
     "Intermediate control retaining the reached-row rejection scan; returns dual_F when requested."},
    {"solve_specialized", reinterpret_cast<PyCFunction>(solve_specialized), METH_VARARGS | METH_KEYWORDS,
     "solve_specialized(C, L, return_dual=False): implicit uniform Hungarian with cached rejection.\n"
     "Same finite nonnegative contract, pair output, and dual_F as solve; no JV routines."},
    {"solve_specialized_padded", reinterpret_cast<PyCFunction>(solve_specialized_padded), METH_VARARGS | METH_KEYWORDS,
     "Specialized cached kernel reading the identical one-sided materialized buffer at stride n+1."},
    {"diagnose", reinterpret_cast<PyCFunction>(diagnose), METH_VARARGS | METH_KEYWORDS,
     "diagnose(C,L,variant='original'): untimed state arrays, objective, and actual coverage counters.\n"
     "Unavailable upstream internal counters are None; no_search uses its exact preprocessing.\n"
     "All calls require C,L<=DBL_MAX/[16*(n+m+1)] and round-to-nearest mode."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_lsape_native",
    "Shared NumPy bridge to original LSAPE v4 and an experimental Hungarian specialization.",
    -1, methods};
} // namespace

PyMODINIT_FUNC PyInit__lsape_native() {
    import_array();
    return PyModule_Create(&module);
}
