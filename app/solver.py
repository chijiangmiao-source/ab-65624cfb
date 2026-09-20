"""Exact oligomer-combination solver.

All arithmetic is performed with arbitrary-precision Python integers
(micro-Dalton values and particle counts).  No floating-point value ever
participates in a feasibility or optimisation decision.

Problem
-------
Given component masses m_i (positive integers, micro-Dalton), per-component
count bounds 0 <= lo_i <= hi_i (<= 1_000_000), target mass T and tolerance
tol, find count vectors c (lo_i <= c_i <= hi_i) that are lexicographic
witnesses of:

  1. minimum absolute mass error |sum c_i*m_i - T| among vectors inside the
     tolerance window, then
  2. minimum total particle count sum c_i,

reporting the lexicographically smallest vector (component-id order) as the
canonical result and whether the two-stage optimum is unique.

The bounded counting ranges are never enumerated.  A component's excess
range [0, hi_i - lo_i] is binary-split into O(log hi_i) 0/1 blocks, and the
set of reachable masses is represented as an ordered list of disjoint
integer intervals (run compression).  Exact greedy predecessor/successor
walks over per-component suffix sets provide globally nearest reachable
masses; branch-and-bound depth-first searches over components (never over
individual count values) provide minimum counts, the canonical vector and
uniqueness witnesses.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Per-component count upper bound mandated by the service contract.
MAX_COUNT = 1_000_000


# ---------------------------------------------------------------------------
# Ordered disjoint inclusive integer intervals
# ---------------------------------------------------------------------------

def merge_intervals(intervals):
    """Merge an iterable of ``[lo, hi]`` intervals (input order irrelevant)."""
    import heapq

    heap = [(int(a), int(b)) for a, b in intervals if a <= b]
    if not heap:
        return []
    heapq.heapify(heap)
    out = []
    cur_lo, cur_hi = heapq.heappop(heap)
    while heap:
        lo, hi = heapq.heappop(heap)
        if lo <= cur_hi + 1:
            if hi > cur_hi:
                cur_hi = hi
        else:
            out.append([cur_lo, cur_hi])
            cur_lo, cur_hi = lo, hi
    out.append([cur_lo, cur_hi])
    return out


def shifted(intervals, delta):
    return [[lo + delta, hi + delta] for lo, hi in intervals]


def union_interval_sets(a, b):
    """Union of two ordered disjoint interval lists (streaming merge)."""
    out = []
    i = j = 0
    na, nb = len(a), len(b)
    while i < na or j < nb:
        if j >= nb or (i < na and a[i][0] <= b[j][0]):
            lo, hi = a[i]
            i += 1
        else:
            lo, hi = b[j]
            j += 1
        if out and lo <= out[-1][1] + 1:
            if hi > out[-1][1]:
                out[-1][1] = hi
        else:
            out.append([lo, hi])
    return out


def minkowski_add(a, b):
    """Minkowski sum of two run-compressed sets.

    Cost is proportional to the numbers of runs, never to interval widths
    or the underlying count ranges.
    """
    if not a or not b:
        return []
    pieces = [shifted(b, lo) for lo, _hi in a]
    return merge_intervals(iv for part in pieces for iv in part)


def contains(intervals, x):
    """Exact membership test by binary search."""
    lo, hi = 0, len(intervals) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        a, b = intervals[mid]
        if x < a:
            hi = mid - 1
        elif x > b:
            lo = mid + 1
        else:
            return True
    return False


# ---------------------------------------------------------------------------
# Binary 0/1 splitting of a bounded count range
# ---------------------------------------------------------------------------

def split_range(lo, hi):
    """Split integer range ``[lo, hi]`` into binary 0/1 blocks.

    Returns ``(base, weights)``: any value in [lo, hi] equals ``base`` plus a
    subset sum of ``weights``.  The subset sums of ``weights`` cover
    ``0 .. hi-lo`` with no gaps.
    """
    lo = int(lo)
    width = int(hi) - lo
    weights = []
    bit = 1
    remaining = width
    while bit <= remaining:
        weights.append(bit)
        remaining -= bit
        bit <<= 1
    if remaining > 0:
        weights.append(remaining)
    return lo, weights


def _ceil_div(a, b):
    """Integer ceil(a / b) for b > 0 (a may be negative)."""
    return -((-a) // b)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    identifier: str
    mass: int          # micro-Dalton, positive
    low: int           # inclusive count lower bound
    high: int          # inclusive count upper bound


@dataclass
class SolveInput:
    target: int
    tolerance: int
    components: list   # list[Component] in submission (= canonical id) order


@dataclass
class Attrib:
    counts: tuple
    total_mass: int
    error: int        # signed: total_mass - target
    total_count: int


@dataclass
class SolveResult:
    feasible: bool
    best_error: int | None = None
    min_total_count: int | None = None
    canonical: Attrib | None = None
    unique: bool | None = None
    witness: Attrib | None = None
    lower: Attrib | None = None   # unsatisfiable: nearest reachable <= target
    upper: Attrib | None = None   # unsatisfiable: nearest reachable >= target
    reachable_min: int = 0
    reachable_max: int = 0


# ---------------------------------------------------------------------------
# Reachability model
# ---------------------------------------------------------------------------

class Model:
    """Per-component suffix reachability sets.

    Every component contributes a mandatory base ``lo_i`` (folded into
    ``base_mass``) plus an excess count ``e_i in [0, hi_i - lo_i]``.
    ``suffix[i]`` is the run-compressed set of excess masses reachable by
    components i..n-1.
    """

    def __init__(self, components):
        self.components = components
        n = len(components)
        self.base_counts = [c.low for c in components]
        self.base_mass = sum(c.low * c.mass for c in components)
        self.widths = [c.high - c.low for c in components]

        suffix = [None] * (n + 1)
        suffix[n] = [[0, 0]]
        for i in range(n - 1, -1, -1):
            _, weights = split_range(components[i].low, components[i].high)
            reach = [[0, 0]]
            m = components[i].mass
            for w in weights:
                v = w * m
                reach = union_interval_sets(reach, shifted(reach, v))
            suffix[i] = minkowski_add(reach, suffix[i + 1])
        self.suffix = suffix

        self.reach_min = self.base_mass + suffix[0][0][0]
        self.reach_max = self.base_mass + suffix[0][-1][1]

        # Largest component mass among components i..n-1 (count lower bound
        # for the branch-and-bound search).
        max_mass = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            max_mass[i] = max(components[i].mass, max_mass[i + 1])
        self.max_mass_tail = max_mass

    # -- feasibility -------------------------------------------------------

    def excess_reachable(self, excess_mass):
        return contains(self.suffix[0], excess_mass)

    # -- candidate excess-count windows ------------------------------------

    def excess_count_ranges(self, i, need):
        """Return merged ``[e_lo, e_hi]`` ranges of excess counts for
        component ``i`` such that ``need - e*m_i`` is reachable by components
        i+1..n-1.  The count range [0, width_i] itself is never enumerated.
        """
        m = self.components[i].mass
        width = self.widths[i]
        tail = self.suffix[i + 1]
        raw = []
        for a, b in tail:
            # need - e*m in [a, b]  <=>  e in [(need-b)/m, (need-a)/m]
            e_lo = _ceil_div(need - b, m)
            e_hi = (need - a) // m
            if e_lo < 0:
                e_lo = 0
            if e_hi > width:
                e_hi = width
            if e_lo <= e_hi:
                raw.append((e_lo, e_hi))
        return merge_intervals(raw)

    # -- exact greedy predecessor / successor walks ------------------------

    def predecessor(self, target):
        """Greatest reachable mass <= target and a witnessing count vector.

        Returns ``(mass, counts)`` or ``None`` when target is below the
        smallest reachable mass.
        """
        need = target - self.base_mass
        if need < 0:
            return None
        chosen = self._walk(need, downward=True)
        counts = tuple(self.base_counts[i] + chosen[i]
                       for i in range(len(self.components)))
        mass = self.base_mass + sum(
            chosen[i] * self.components[i].mass
            for i in range(len(self.components)))
        return mass, counts

    def successor(self, target):
        """Smallest reachable mass >= target and a witnessing count vector.

        Returns ``(mass, counts)`` or ``None`` when target is above the
        largest reachable mass.
        """
        need = target - self.base_mass
        total_width_mass = self.suffix[0][-1][1]
        if need > total_width_mass:
            return None
        chosen = self._walk(need, downward=False)
        counts = tuple(self.base_counts[i] + chosen[i]
                       for i in range(len(self.components)))
        mass = self.base_mass + sum(
            chosen[i] * self.components[i].mass
            for i in range(len(self.components)))
        return mass, counts

    def _walk(self, need0, downward):
        """Greedy exact predecessor/successor attribution.

        At component i the tail set ``suffix[i+1]`` is exact, so the best
        feasible continuation can be chosen per interval run without
        expanding any count range.
        """
        n = len(self.components)
        chosen = [0] * n
        need = need0
        for i in range(n):
            m = self.components[i].mass
            width = self.widths[i]
            tail = self.suffix[i + 1]
            best = None  # (gap, total excess from i, e)
            for a, b in tail:
                if downward:
                    # Maximise e*m + s with s in [a, b], total <= need.
                    e_hi = (need - a) // m
                    if e_hi > width:
                        e_hi = width
                    if e_hi >= 0:
                        s = need - e_hi * m
                        if s <= b:
                            # Exact hit inside this run (s >= a by e_hi).
                            best = (0, e_hi * m + s, e_hi)
                            break
                        e_lo = _ceil_div(need - b, m)
                        if 0 <= e_lo <= width:
                            s = need - e_lo * m  # s <= b
                            total = e_lo * m + s
                            gap = need - total
                            if best is None or gap < best[0] or (
                                    gap == best[0] and e_lo < best[2]):
                                best = (gap, total, e_lo)
                else:
                    # Minimise e*m + s with s in [a, b], total >= need.
                    e_lo = _ceil_div(need - b, m)
                    if e_lo < 0:
                        e_lo = 0
                    if e_lo <= width:
                        s = need - e_lo * m
                        if s >= a:
                            # Exact hit inside this run (s <= b by e_lo).
                            best = (0, e_lo * m + s, e_lo)
                            break
                        e_hi = _ceil_div(need - a, m)
                        if 0 <= e_hi <= width:
                            s = need - e_hi * m  # s <= a
                            total = e_hi * m + s
                            gap = total - need
                            if best is None or gap < best[0] or (
                                    gap == best[0] and e_hi < best[2]):
                                best = (gap, total, e_hi)
            if best is None:
                # Should be unreachable given the outer existence check:
                # every reachable state has at least one continuation.
                raise RuntimeError("internal error: greedy walk stranded")
            _gap, total, e = best
            chosen[i] = e
            need -= e * m  # residual need passed to the next component
        # For a predecessor walk the residual is the (non-negative) shortfall
        # absorbed by the final component; total mass is tracked by caller.
        return chosen


# ---------------------------------------------------------------------------
# Depth-first searches at an exact target mass
# ---------------------------------------------------------------------------

def _excess_target(model, mass):
    return mass - model.base_mass


def min_count(model, mass):
    """Minimum total particle count attaining ``mass`` exactly.

    Returns ``(min_total, counts)`` or ``None``.  Components are visited in
    canonical order; feasible excess counts are obtained in merged range
    windows from the suffix oracle, so individual count values are only
    generated inside pruned windows.
    """
    n = len(model.components)
    base_total = sum(model.base_counts)
    E0 = _excess_target(model, mass)
    if not contains(model.suffix[0], E0):
        return None

    best = [None, None]  # [extra_count, chosen excess counts]

    def search(i, need, extra, chosen):
        if i == n:
            if need == 0 and (best[0] is None or extra < best[0]):
                best[0] = extra
                best[1] = list(chosen)
            return
        if best[0] is not None and extra >= best[0]:
            return
        # Count lower bound: remaining need needs at least
        # ceil(need / largest tail mass) further particles.
        if need > 0:
            lb = _ceil_div(need, model.max_mass_tail[i])
            if best[0] is not None and extra + lb >= best[0]:
                return
        elif need < 0:
            return
        ranges = model.excess_count_ranges(i, need)
        for el, eh in ranges:
            # ascending counts: good incumbent early
            e = el
            while e <= eh:
                if best[0] is not None and extra + e >= best[0]:
                    break
                chosen.append(e)
                search(i + 1, need - e * model.components[i].mass,
                       extra + e, chosen)
                chosen.pop()
                e += 1
            if best[0] is not None and extra + el >= best[0]:
                break

    search(0, E0, 0, [])
    if best[0] is None:
        return None
    counts = tuple(model.components[i].low + best[1][i] for i in range(n))
    return base_total + best[0], counts


def lex_first(model, mass, budget, forbidden=None):
    """Lexicographically smallest count vector (component-id order)
    attaining ``mass`` with total count <= ``budget``.

    If ``forbidden`` is given as a count tuple, vectors equal to it are
    skipped, yielding the next lexicographically smallest vector instead.
    """
    n = len(model.components)
    base_total = sum(model.base_counts)
    E0 = _excess_target(model, mass)
    if not contains(model.suffix[0], E0):
        return None

    result = []

    def search(i, need, extra, chosen, diverged):
        if result:
            return
        if i == n:
            if need == 0 and (diverged or forbidden is None):
                result.extend(chosen)
                return
            return
        if extra > budget - base_total:
            return
        if need < 0:
            return
        if need > 0:
            lb = _ceil_div(need, model.max_mass_tail[i])
            if extra + lb > budget - base_total:
                return
        ranges = model.excess_count_ranges(i, need)
        for el, eh in ranges:
            max_e = eh
            cap = budget - base_total - extra
            if max_e > cap:
                max_e = cap
            e = el
            while e <= max_e and not result:
                if not diverged and forbidden is not None:
                    ref_e = forbidden[i] - model.components[i].low
                    # Explore the reference-matching branch last so a true
                    # divergence, when it exists, is found without first
                    # descending the whole forbidden path.
                    if e == ref_e and e < max_e:
                        e += 1
                        continue
                nd = diverged or (
                    forbidden is not None
                    and e != forbidden[i] - model.components[i].low)
                chosen.append(e)
                search(i + 1, need - e * model.components[i].mass,
                       extra + e, chosen, nd)
                chosen.pop()
                e += 1

    search(0, E0, 0, [], forbidden is None)
    if not result:
        return None
    return tuple(model.components[i].low + result[i] for i in range(n))


def _attrib(model, counts, target):
    total = sum(c * comp.mass for c, comp in zip(counts, model.components))
    return Attrib(
        counts=tuple(counts),
        total_mass=total,
        error=total - target,
        total_count=sum(counts),
    )


# ---------------------------------------------------------------------------
# Two-stage inversion
# ---------------------------------------------------------------------------

def solve(data: SolveInput) -> SolveResult:
    model = Model(data.components)
    target = data.target
    tol = data.tolerance
    lo_bound = target - tol
    hi_bound = target + tol

    result = SolveResult(
        feasible=False,
        reachable_min=model.reach_min,
        reachable_max=model.reach_max,
    )

    full = model.suffix[0]
    in_window = any(b >= lo_bound and a <= hi_bound for a, b in full)

    if not in_window:
        pred = model.predecessor(target)
        succ = model.successor(target)
        if pred is not None and pred[0] <= target:
            result.lower = _attrib(model, pred[1], target)
        if succ is not None and succ[0] >= target:
            result.upper = _attrib(model, succ[1], target)
        return result

    # ---- Stage 1: reachable mass(es) at minimum absolute error -----------
    optimal_masses = []          # masses sharing the minimum |error|
    if lo_bound <= target <= hi_bound and model.excess_reachable(
            target - model.base_mass):
        optimal_masses = [target]
    else:
        pred = model.predecessor(target)
        succ = model.successor(target)
        candidates = []
        if pred is not None and pred[0] < target and pred[0] >= lo_bound:
            candidates.append((target - pred[0], pred[0], pred[1]))
        if succ is not None and succ[0] > target and succ[0] <= hi_bound:
            candidates.append((succ[0] - target, succ[0], succ[1]))
        # Exact equality is handled above; defensively keep it here too.
        if pred is not None and pred[0] == target:
            candidates.append((0, target, pred[1]))
        if succ is not None and succ[0] == target:
            candidates.append((0, target, succ[1]))
        if not candidates:
            # The window intersects a run but neither tight neighbour lies in
            # it: clamp to the window boundary instead.
            for a, b in full:
                if b >= lo_bound and a <= hi_bound:
                    edge = max(a, lo_bound)
                    sc = model.successor(edge)
                    if sc is not None and sc[0] <= hi_bound:
                        candidates.append((abs(sc[0] - target), sc[0], sc[1]))
                    edge2 = min(b, hi_bound)
                    pr = model.predecessor(edge2)
                    if pr is not None and pr[0] >= lo_bound:
                        candidates.append((abs(pr[0] - target), pr[0], pr[1]))
                    break
        best_err = min(err for err, _m, _c in candidates)
        seen = set()
        optimal_masses = []
        for err, mass, _counts in candidates:
            if err == best_err and mass not in seen:
                seen.add(mass)
                optimal_masses.append(mass)

    # ---- Stage 2: minimum particle count over all best-error masses ------
    per_mass = []  # (mass, min_total, some counts)
    global_min = None
    for mass in optimal_masses:
        found = min_count(model, mass)
        if found is None:
            continue
        total, counts = found
        per_mass.append((mass, total, counts))
        if global_min is None or total < global_min:
            global_min = total

    best_mass_entries = [(m, t, c) for m, t, c in per_mass if t == global_min]

    # Canonical vector = lexicographically smallest across all best masses.
    canonical_mass = None
    canonical_counts = None
    for mass, _t, _c in sorted(best_mass_entries, key=lambda x: x[0]):
        vec = lex_first(model, mass, global_min)
        if vec is not None and (canonical_counts is None
                                or vec < canonical_counts):
            canonical_counts = vec
            canonical_mass = mass

    result.feasible = True
    result.best_error = abs(canonical_mass - target)
    result.min_total_count = global_min
    result.canonical = _attrib(model, canonical_counts, target)

    # ---- Uniqueness: any other optimal vector at any best-error mass -----
    witness_counts = None
    witness_mass = None
    other = lex_first(model, canonical_mass, global_min,
                      forbidden=canonical_counts)
    if other is not None:
        witness_counts, witness_mass = other, canonical_mass
    else:
        for mass, _t, _c in best_mass_entries:
            if mass == canonical_mass:
                continue
            vec = lex_first(model, mass, global_min)
            if vec is not None:
                witness_counts, witness_mass = vec, mass
                break

    if witness_counts is not None:
        result.unique = False
        result.witness = _attrib(model, witness_counts, target)
        # Witness must point at the mass it actually attains.
        result.witness = Attrib(
            counts=result.witness.counts,
            total_mass=witness_mass,
            error=witness_mass - target,
            total_count=sum(witness_counts),
        )
    else:
        result.unique = True
    return result
