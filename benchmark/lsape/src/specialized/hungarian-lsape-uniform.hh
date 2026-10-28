// Uniform one-sided specialization of the LSAPE v4 Hungarian algorithm.
//
// Adapted from include/hungarian-lsape.hh in upstream LSAPE v4 (July 2017),
// by Sebastien Bougleux and Luc Brun, Normandie Univ / GREYC.
// Reference: Bougleux and Brun, Linear Sum Assignment with Edition (2016).
// This adapted file is distributed under the upstream CeCILL-C license;
// see ../upstream-v4/lsape/Licence_CeCILL-C_V1-en.txt relative to this header.
// Upstream files are retained unchanged.
//
// Experimental changes:
//   * supply the uniform removal border and zero insertion border implicitly;
//   * specialize column preprocessing for finite nonnegative real costs;
//   * retain column-order greedy assignment and FIFO row-rooted Hungarian search;
//   * omit column-rooted completion: preprocessing assigns every real column
//     either to a real row or to the free-insertion null row;
//   * optionally cache the rejection predecessor during existing state updates.
// No JV reductions, JV shortest-path implementation, orientation, or batching.

#ifndef LSAPE_EXPERIMENT_HUNGARIAN_UNIFORM_HH
#define LSAPE_EXPERIMENT_HUNGARIAN_UNIFORM_HH

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace lsape_experiment {
// Instrumented instantiations are used only by the untimed diagnostic API.
// if constexpr removes every counter access from timed instantiations.
struct Coverage {
    std::uint64_t searches = 0, dual_updates = 0, rows_reached = 0;
    std::uint64_t free_column = 0, rejection_row_entry = 0, rejection_dual_update = 0;
    std::uint64_t cache_fast_path = 0, cache_fallback = 0, scan_rejection = 0;
    std::uint64_t preprocess_unassigned_rows = 0, preprocess_unassigned_cols = 0;
};
namespace detail {

template<class DT, typename IT>
inline DT realCost(const DT* C, IT leading_dimension, IT i, IT j) {
    return C[static_cast<std::size_t>(j) * static_cast<std::size_t>(leading_dimension)
             + static_cast<std::size_t>(i)];
}

// Row-rooted FIFO Hungarian tree, corresponding to augmentingPathRowLSAPE.
// V partitions columns into settled, tight-unsettled, and remaining columns.
// SU lists reached rows in insertion order and ends in a -1 sentinel.
// A real column is free for this search if varrho[j] == n (free insertion).
template<class DT, typename IT, bool DIAGNOSTIC>
void augmentRow(const IT k, const DT* C, const IT n, const IT m, const IT ld, const DT penalty,
                const IT* rho, const IT* varrho, IT* V, IT* SU, IT* pred,
                DT* u, DT* v, DT* pi, IT& zi, IT& zj, bool cache_rejection, Coverage* coverage) {
    if constexpr (DIAGNOSTIC) ++coverage->searches;
    const IT* suptr = nullptr;
    const IT* lvptr = nullptr;
    IT* const vend = V + m;
    IT* lvsvtop = V;
    IT i = k, j = 0, c = 0;
    IT* SUptr = SU;
    IT* vlvtop = V;
    IT* vlvptr = nullptr;
    DT delta = 0, cred = 0;
    bool last_column = false;
    IT best_rejection_row = -1;
    *SU = -1;
    zi = zj = -1;
    for (j = 0; j < m; ++j) {
        pi[j] = std::numeric_limits<DT>::max();
        V[j] = j;
    }

    while (true) {
        *SUptr = i;
        *(++SUptr) = -1;
        if constexpr (DIAGNOSTIC) ++coverage->rows_reached;
        // Reached rows all receive the same dual increment. A maximal u has
        // minimal penalty-u; >= retains the latest reached row on exact ties.
        if (cache_rejection &&
            (best_rejection_row < 0 || u[i] >= u[best_rejection_row]))
            best_rejection_row = i;

        if (rho[i] < m && penalty == u[i]) {
            if constexpr (DIAGNOSTIC) ++coverage->rejection_row_entry;
            zi = i;
            zj = m;
            return;
        }

        for (vlvptr = vlvtop; vlvptr != vend; ++vlvptr) {
            c = *vlvptr;
            cred = realCost(C, ld, i, c) - (u[i] + v[c]);
            if (cred < pi[c]) {
                pred[c] = i;
                pi[c] = cred;
                if (cred == 0) {
                    if (varrho[c] == -1 || varrho[c] == n) {
                        if constexpr (DIAGNOSTIC) ++coverage->free_column;
                        zi = -1;
                        zj = c;
                        return;
                    }
                    j = *vlvtop;
                    *vlvtop = c;
                    *vlvptr = j;
                    ++vlvtop;
                }
            }
        }

        if (lvsvtop == vlvtop) {
            if constexpr (DIAGNOSTIC) ++coverage->dual_updates;
            delta = std::numeric_limits<DT>::max();
            last_column = false;
            for (vlvptr = vlvtop; vlvptr != vend; ++vlvptr)
                if (pi[*vlvptr] < delta) delta = pi[*vlvptr];

            // Recompute the cached candidate from the current potential,
            // rather than repeatedly subtracting deltas from a cached slack.
            // Assumptions: IEC559 binary arithmetic, round-to-nearest, and no
            // fast-math/contraction/extended-expression precision. The build
            // uses /fp:precise. This local ordering argument is not a proof of
            // all Hungarian floating-point transitions or objective accuracy.
            // With p/2 < u* <= 2p, p-u* is exact by Sterbenz. Rows <=p/2
            // cannot tie that smaller slack (p/2 is representable for p=2h);
            // rows above p/2 have exact ordered differences. The latest
            // maximal-u row therefore agrees with the ordered <= scan.
            // Check the upper bound explicitly, rather than assuming that
            // every rounded dual update preserves an informal u<=2p bound.
            // At/below p/2 or outside this range, retain the original scan.
            const bool exact_cache_order = cache_rejection && best_rejection_row >= 0 &&
                std::isfinite(u[best_rejection_row]) &&
                u[best_rejection_row] > penalty / DT(2) &&
                u[best_rejection_row] <= penalty + penalty;
            if (exact_cache_order) {
                if constexpr (DIAGNOSTIC) ++coverage->cache_fast_path;
                cred = penalty - u[best_rejection_row];
                if (cred <= delta) {
                    delta = cred;
                    last_column = true;
                    zi = best_rejection_row;
                }
            } else {
                if constexpr (DIAGNOSTIC) {
                    if (cache_rejection) ++coverage->cache_fallback;
                    else ++coverage->scan_rejection;
                }
                for (suptr = SU; *suptr != -1; ++suptr) {
                    cred = penalty - u[*suptr];
                    if (cred <= delta) {
                        delta = cred;
                        last_column = true;
                        zi = *suptr;
                    }
                }
            }

            // This loop is already required by Hungarian dual updates.
            // Refreshing the maximum here also catches ties introduced when
            // adding the common delta rounds two distinct potentials equally.
            if (cache_rejection) best_rejection_row = -1;
            for (suptr = SU; *suptr != -1; ++suptr) {
                u[*suptr] += delta;
                if (cache_rejection &&
                    (best_rejection_row < 0 || u[*suptr] >= u[best_rejection_row]))
                    best_rejection_row = *suptr;
            }
            for (lvptr = V; lvptr != vlvtop; ++lvptr) v[*lvptr] -= delta;
            if (last_column) {
                if constexpr (DIAGNOSTIC) ++coverage->rejection_dual_update;
                zj = m;
                return;
            }
            for (vlvptr = vlvtop; vlvptr != vend; ++vlvptr) {
                pi[*vlvptr] -= delta;
                if (pi[*vlvptr] == 0) {
                    if (varrho[*vlvptr] == -1 || varrho[*vlvptr] == n) {
                        if constexpr (DIAGNOSTIC) ++coverage->free_column;
                        zi = -1;
                        zj = *vlvptr;
                        return;
                    }
                    c = *vlvtop;
                    *vlvtop = *vlvptr;
                    *vlvptr = c;
                    ++vlvtop;
                }
            }
        }
        j = *lvsvtop;
        ++lvsvtop;
        i = varrho[j];
    }
}
} // namespace detail

// Preconditions: finite nonnegative C and penalty; n,m >= 0; signed IT;
// column-major C with leading_dimension>=n; caller-allocated rho[n], varrho[m],
// u[n+1], v[m+1]. The real entries are C[j*leading_dimension+i].
// Dimensions, products, and cost magnitudes must fit the working/index types.
// The caller validates the numeric domain; this kernel adds no full-input scan.
//
// Equivalent edit matrix: [[C, penalty], [0, 0]]. Objective is
// sum(real matched costs) + penalty*(n-K), not the symmetric endpoint total.
// For penalty=2*(L/2), restoring (L/2)*(m-n) gives the endpoint objective
// in exact arithmetic. Pair choices/cardinality need not be unique under ties.
// rho[i]==m means removal; varrho[j]==n means insertion; null potentials are 0.
template<class DT, typename IT, bool DIAGNOSTIC = false>
void hungarianUniform(const DT* C, IT n, IT m, IT leading_dimension, DT penalty,
                      IT* rho, IT* varrho, DT* u, DT* v,
                      bool cache_rejection = true, Coverage* coverage = nullptr) {
    static_assert(std::is_integral<IT>::value && std::is_signed<IT>::value,
                  "LSAPE mappings require a signed integral index type");
    static_assert(std::numeric_limits<DT>::is_iec559,
                  "This experimental kernel requires IEC559 floating-point costs");
    if (n < 0 || m < 0 || leading_dimension < n || penalty < DT(0))
        throw std::invalid_argument("hungarianUniform requires nonnegative dimensions and penalty");
    if constexpr (DIAGNOSTIC) {
        if (!coverage) throw std::invalid_argument("Diagnostic coverage is required");
        *coverage = Coverage{};
    }
#ifndef NDEBUG
    const DT magnitude_bound = std::numeric_limits<DT>::max() /
        (DT(16) * (static_cast<DT>(n) + static_cast<DT>(m) + DT(1)));
    assert(std::isfinite(penalty) && penalty >= DT(0) && penalty <= magnitude_bound);
    for (IT j = 0; j < m; ++j) for (IT i = 0; i < n; ++i) {
        const DT value = detail::realCost(C, leading_dimension, i, j);
        assert(std::isfinite(value) && value >= DT(0) && value <= magnitude_bound);
    }
#endif
    u[n] = v[m] = DT(0);
    IT nass = 0;

    // Specialization of basicPreprocColsLSAPE. The insertion row is zero,
    // and all real costs are nonnegative, so every initial column minimum is 0.
    for (IT j = 0; j < m; ++j) {
        v[j] = DT(0);
        varrho[j] = -1;
    }
    for (IT i = 0; i < n; ++i) {
        DT minimum = std::numeric_limits<DT>::max();
        for (IT j = 0; j < m; ++j) {
            const DT value = detail::realCost(C, leading_dimension, i, j) - v[j];
            if (value < minimum) minimum = value;
        }
        const DT removal = penalty - v[m];
        if (removal < minimum) minimum = removal;
        u[i] = minimum;
        rho[i] = -1;
    }
    // Preserve upstream column-major greedy traversal and tight-edge tests.
    for (IT j = 0; j < m; ++j) {
        for (IT i = 0; i < n; ++i) {
            if (rho[i] < 0 && varrho[j] < 0 &&
                detail::realCost(C, leading_dimension, i, j) == u[i] + v[j]) {
                rho[i] = j;
                varrho[j] = i;
                ++nass;
                break;
            }
        }
        if (varrho[j] < 0) varrho[j] = n;
    }
    for (IT i = 0; i < n; ++i) {
        if (rho[i] < 0 && penalty == u[i]) {
            rho[i] = m;
            ++nass;
        }
    }
    if constexpr (DIAGNOSTIC) coverage->preprocess_unassigned_rows = n - nass;
    if (nass == n) return;

    // Match upstream's allocation timing and uninitialized scratch storage.
    // RAII changes ownership only; each search initializes the same entries.
    std::unique_ptr<IT[]> V(new IT[static_cast<std::size_t>(m) + 1]);
    std::unique_ptr<IT[]> SU(new IT[static_cast<std::size_t>(n) + 1]);
    std::unique_ptr<DT[]> pi(new DT[static_cast<std::size_t>(m)]);
    std::unique_ptr<IT[]> pred(new IT[static_cast<std::size_t>(m)]);
    for (IT k = 0; k < n; ++k) {
        if (rho[k] != -1) continue;
        IT i = -1, j = -1, c = -1;
        detail::augmentRow<DT, IT, DIAGNOSTIC>(k, C, n, m, leading_dimension, penalty, rho, varrho,
                           V.get(), SU.get(), pred.get(), u, v, pi.get(),
                           i, j, cache_rejection, coverage);
        if (j == m) {
            c = rho[i];
            rho[i] = j;
            j = c;
        } else {
            i = -1;
        }
        while (i != k) {
            i = pred[j];
            varrho[j] = i;
            c = rho[i];
            rho[i] = j;
            j = c;
        }
    }
}
} // namespace lsape_experiment

#endif
