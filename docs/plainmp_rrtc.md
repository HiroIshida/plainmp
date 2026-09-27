# PlainmpRRTC

For the historical comparison against the removed `vamp_rrtc` planner, see
[OMPL, PlainmpRRTC and VampRRTC](compare_rrtc.md).

`plainmp_rrtc` reimplements the search used by plainmp's default OMPL
[RRTConnect 1.6](https://github.com/ompl/ompl/blob/1.6.0/src/ompl/geometric/planners/rrt/src/RRTConnect.cpp).
It uses exact Euclidean KD-tree queries and constructor-allocated pools.
The default algorithm remains `RRTConnect`; the default range remains **2.0**.

```python
from plainmp.ompl_solver import (
    Algorithm, OMPLSolver, OMPLSolverConfig, PlainmpRRTCSettings,
)

settings = PlainmpRRTCSettings()
settings.max_samples = 100_000  # combined capacity of both trees, including roots
solver = OMPLSolver(OMPLSolverConfig(
    algorithm=Algorithm.PlainmpRRTC,
    algorithm_range=2.0,
    plainmp_rrtc_settings=settings,
))
result = solver.solve(problem)
```

## Algorithm

The implementation preserves these RRTConnect choices:

- Uniform sampling with OMPL's state sampler.
- Alternate the active tree each outer iteration, starting with the start tree.
- EXTEND the active tree once; CONNECT the opposite tree toward that new node.
- Each CONNECT step performs a nearest-neighbor query and advances by `range`,
  with a shorter last step when the target is nearer.
- Validate goal-tree edges toward the parent, explicitly checking the new
  state first. Use the existing plainmp box/Euclidean motion validators.
- Remove one duplicated connection state using OMPL's path reconstruction rule.

VAMP's direct initial connection, dynamic domains, balanced tree selection,
Halton samples, and equal subdivisions during CONNECT are not used here.
OMPL normally selects GNAT for its nearest-neighbor backend; this implementation
uses an independently implemented exact batch KD-tree. Ties choose the earliest
inserted node, which need not be GNAT's tie order. The selected edge's distance
and interpolation use OMPL arithmetic to preserve discrete validation grids.

There are a few explicit interface differences:

- The combined node capacity is finite. Exhaustion returns failure without
  reallocating. Raise `max_samples` **at construction** for larger problems.
- Termination is checked during CONNECT as well as between iterations. The
  dedicated Python wrapper respects both the call budget and the timeout.
- Only exact solutions are returned; plainmp's existing wrapper also rejects
  approximate solutions. Approximate-path bookkeeping is omitted.
- Each fixed-goal solve clears the pools and starts from the start tree. The
  sampler is retained and its random sequence continues across solves.
- Only the standard Euclidean `RealVectorStateSpace` is supported. Custom
  metrics, state-space subclasses, and intermediate-state insertion are not.

## Memory and interfaces

The native constructor allocates:

1. One contiguous coordinate array, with coordinates of a configuration adjacent.
2. Separate `uint32_t` parent/root arrays.
3. Both KD-tree node pools (128 indices per leaf, median split on the widest axis).
4. Sampling/interpolation scratch space and a path-index buffer.
5. The uniform state sampler.

Insert, nearest query, splitting, reset, CONNECT, and path reconstruction reuse
this storage. Clearing a query resets counters; it does not free nodes or visit
all reserved entries. Unused pool pages are not eagerly initialized. For 8 DOF
and 100,000 nodes, the pools reserve roughly 10.5 MiB (resident memory depends
on nodes actually visited).

`PlainmpRRTCPlanner` provides the same Python `solve` arguments and result matrix
as the other wrappers, plus `get_node_count()`. It allocates its constraint input
vector and termination condition in its constructor. Reuse this object directly
when bounds and the collision constraint stay fixed. `OMPLSolver.solve()` still
constructs a planner per call, following the existing high-level interface.
Pose goals are converted by the existing IK flow; supplying an initial trajectory
still uses ERTConnect.

`get_ns_internal()` / `OMPLSolverResult.ns_internal` measures search, path
reconstruction, and requested refinement. Like the existing RRTConnect wrapper,
it stops before allocating and filling the output Eigen matrix. Constructor
allocation, Python conversion, and planner destruction are also excluded.

The fixed-goal path avoids allocating OMPL states or `PathGeometric` objects in
search. Python argument conversion and the returned NumPy/Eigen matrix allocate
at the API boundary. Requested shortcut/B-spline refinement uses OMPL and may
allocate. External constraint implementations may allocate internally; they
were not changed. Planners and their validity buffers are not thread safe.

The name `plainmp_rrtc` is also registered in the normal C++/`OMPLPlanner`
selector. This adapter supports sampleable goals and multiple roots through
OMPL, retaining its usual problem/goal/output allocations. Use the dedicated
wrapper for the allocation-conscious fixed-goal path.

## Verification

- Native comparison: 20 deterministic sample streams at each of ranges 0.2 and
  2.0. All state-validity calls, directed motion checks and resulting waypoints
  agree with OMPL RRTConnect (using exact linear NN for a common tie rule).
- Global C++ `new/delete` counters: no allocation or release during the first
  search, reuse, path reconstruction, or capacity exhaustion, including KD splits.
  Also checked with OMPL's default uniform sampler and both existing plainmp
  motion validators. Constructor and external robot constraints are outside
  that counter assertion.
- Native tests cover capacity limits, multiple goals, the OMPL adapter, clear,
  invalid starts, and immediate termination. The KD-tree has independent
  brute-force correctness tests in `test_batch_nearest`.
- AddressSanitizer, UndefinedBehaviorSanitizer and LeakSanitizer pass the native tests.
- Python tests cover robot collisions, range bounds, repeated solves, budgets,
  invalid input, fixed/IK goals, refinements, and both registration interfaces.
  Dense path audits use a fine validation grid: the existing discrete validators
  do not provide continuous collision guarantees between samples.

Build native tests with `-DPLAINMP_BUILD_TESTS=ON`, then run:

```sh
ctest --test-dir build/native --output-on-failure
.venv/bin/python -m pytest -q tests/test_plainmp_rrtc.py
.venv/bin/python example/bench/plainmp_rrtc.py --runs 1000 --output /tmp/plainmp_rrtc.json
```

## Benchmark (range 2.0)

AMD Ryzen 7 7840HS, GCC 9.4.0, OMPL 1.6, Release with `-march=native` and LTO.
Each combination uses 10 warmups and 1,000 measured solves, alternating execution
order, uniform sampling, seed 42, validation resolution 0.05, no refinement.
All **24,000 solves succeeded**. RNG streams differ between implementations;
these distribution comparisons include sampling variation, not just overhead.

`solver` measures the complete high-level solve, including planner construction
and result conversion. `reuse` measures the Python call on an existing C++
planner. Times are wall-clock medians in milliseconds.

| Scene / validator | Solver OMPL | Solver new | Ratio | Reuse OMPL | Reuse new | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| fetch / box | 0.631 | 0.548 | 1.15x | 0.595 | 0.565 | 1.05x |
| fetch / euclidean | 0.785 | 0.763 | 1.03x | 0.760 | 0.672 | 1.13x |
| panda_easy / box | 0.187 | 0.159 | 1.17x | 0.149 | 0.126 | 1.18x |
| panda_easy / euclidean | 0.239 | 0.198 | 1.21x | 0.188 | 0.173 | 1.09x |
| panda_difficult / box | 0.674 | 0.567 | 1.19x | 0.599 | 0.537 | 1.12x |
| panda_difficult / euclidean | 0.802 | 0.737 | 1.09x | 0.781 | 0.684 | 1.14x |

P95, means, success counts, and validity-call counts are recorded in
[plainmp_rrtc_benchmark.json](plainmp_rrtc_benchmark.json). Ratios are old/new;
a value above one means the new implementation is faster.

## Internal timing (output conversion excluded)

Fresh measurement after aligning the output-matrix timing boundary with the
existing wrapper. Same range 2.0, seed 42, 1,000 runs per combination, 10 warmups,
resolution 0.05, uniform sampling, and no refinement. All 24,000 solves succeeded
across the fresh-planner and reuse modes. The table uses the normal high-level
solver, which constructs a planner each call; constructor time is excluded by
`ns_internal`. Runtime allocations in the existing RRTConnect search remain
inside its measurement. Values are milliseconds, median / p95.

| Scene | Validator | RRTConnect | plainmp_rrtc | Median speedup |
|---|---|---:|---:|---:|
| panda_easy | box | 0.166 / 0.366 | 0.119 / 0.281 | 1.40x |
| panda_easy | euclidean | 0.217 / 0.469 | 0.160 / 0.412 | 1.36x |
| panda_difficult | box | 0.653 / 1.652 | 0.535 / 1.344 | 1.22x |
| panda_difficult | euclidean | 0.781 / 1.894 | 0.699 / 1.594 | 1.12x |
| fetch | box | 0.616 / 1.917 | 0.508 / 1.643 | 1.21x |
| fetch | euclidean | 0.780 / 2.462 | 0.722 / 2.134 | 1.08x |

The algorithms consume different random streams, so exploration variability
is included. Full summaries (including reuse) are saved in
[plainmp_rrtc_internal_benchmark.json](plainmp_rrtc_internal_benchmark.json).

Reproduce the internal measurement with:

```sh
.venv/bin/python example/bench/plainmp_rrtc.py \
    --runs 1000 --range 2.0 --seed 42 \
    --output /tmp/plainmp_rrtc_internal_range20.json
```

The script uses the Fetch table and Panda easy/difficult scenes defined in
[`make_problem`](../example/bench/rrtc_scenes.py), with fixed joint-space goals
and the same collision constraint for both planners. Box widths are 0.05 on
every axis; Euclidean resolution is 0.05. The validity-call budget is 1,000,000,
no timeout is specified, and the new planner's combined tree capacity is
100,000 nodes. Execution order alternates each trial. The JSON emitted by the
script contains individual samples as well as wall/internal timing summaries;
the checked-in JSON contains summaries and the machine/build conditions.
