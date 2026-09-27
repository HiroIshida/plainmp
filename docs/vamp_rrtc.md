# VAMP-inspired RRTC

```python
from plainmp.ompl_solver import Algorithm, OMPLSolver, OMPLSolverConfig, VampRRTCSettings

settings = VampRRTCSettings()
solver = OMPLSolver(OMPLSolverConfig(
    algorithm=Algorithm.VampRRTC,
    algorithm_range=2.0,
    vamp_rrtc_settings=settings,
))
result = solver.solve(problem)
```

The default range remains **2.0**, including when the native VampRRTC planner
is constructed without an explicit range. Existing planners keep their previous
defaults and implementations. Pose goals use the existing IK flow; `refine_seq`
uses OMPL's existing shortcut/spline routines. As before, supplying a trajectory
guess selects ERTConnect.

## Algorithm correspondence

The reference is the local VAMP checkout at
`f6de6a72e0725ba08e0829f0a34261a3f465aed7`:

- [`rrtc.hh`](https://github.com/KavrakiLab/vamp/blob/f6de6a72e0725ba08e0829f0a34261a3f465aed7/src/impl/vamp/planning/rrtc.hh)
- [`rrtc_settings.hh`](https://github.com/KavrakiLab/vamp/blob/f6de6a72e0725ba08e0829f0a34261a3f465aed7/src/impl/vamp/planning/rrtc_settings.hh)
- [`nn.hh`](https://github.com/KavrakiLab/vamp/blob/f6de6a72e0725ba08e0829f0a34261a3f465aed7/src/impl/vamp/planning/nn.hh)
- [`halton.hh`](https://github.com/KavrakiLab/vamp/blob/f6de6a72e0725ba08e0829f0a34261a3f465aed7/src/impl/vamp/random/halton.hh)

The implementation preserves these behaviors:

1. Check the direct start-to-goal edge before random exploration.
2. Swap trees when balancing is disabled, or when
   `abs(size_a - size_b) / size_a < tree_ratio`. This is the actual asymmetric
   VAMP rule, rather than always selecting the smaller tree.
3. Reject a sample outside its nearest node's dynamic domain. A failed initial
   extension sets an infinite radius to `radius`, or shrinks a finite radius
   by `1-alpha`, bounded below by `min_radius`. Successful initial extensions
   expand an existing finite radius by `1+alpha`.
4. After EXTEND succeeds, query the other tree once. Continue growing the
   **active tree** toward that fixed nearest node in equally divided segments
   of length at most `range`. Retain the valid prefix if CONNECT hits an obstacle.
   The CONNECT loop does not shrink dynamic domains.
5. Reconstruct the exact start-to-goal path with the appropriate orientation.

In [OMPL's RRTConnect](https://github.com/ompl/ompl/blob/main/src/ompl/geometric/planners/rrt/src/RRTConnect.cpp),
CONNECT grows the opposite tree toward the newly added state and invokes its
normal tree-growth routine repeatedly. Its standard implementation also owns
individual motion/state allocations. The new implementation uses the VAMP
exploration order and pooled storage.

## Storage and interface

`cpp/plainmp/ompl/vamp_rrtc.cpp` implements an OMPL `Planner` for ordinary
Euclidean `RealVectorStateSpace` and reversible geometric motion validation.
It is registered as `vamp_rrtc` in the C++ algorithm selector.

Coordinates, parent indices, dynamic-domain radii, scratch space, and both
nearest-neighbor indices are allocated in batches at setup. Unused pool entries
are not zero-filled. `clear()` reuses the pools. Tree insertion and nearest
queries perform no heap allocations, including leaf splitting; a native test
checks this with an allocation counter.

The independent exact KD-tree has 128-entry leaves, matching VAMP's Nigh batch
size. It splits along the widest coordinate at the median and compares squared
distances. Parent and nearest-neighbor references are integer indices. Stack
OMPL state views point into the stable coordinate pool. Only the returned path
is converted into individually owned OMPL states.

`VampRRTCPlanner`, exported from `plainmp.ompl_solver`, provides the Python
wrapper used by `OMPLSolver`. It reuses its own `Eigen::VectorXd` during validity
checks and enforces both collision-call and time limits. These changes are
confined to the new wrapper; the existing OMPL wrapper, constraint, and motion
validator implementations are unchanged. Reusing a native `VampRRTCPlanner`
instance also reuses its pools across `solve()` calls. The high-level solver
constructs a native planner per call, as it does for other algorithms.

The C++ planner supports multiple start/goal roots. The dedicated Python wrapper
takes one start and one fixed goal. High-level pose goals are resolved through
IK; continuous Python goal callbacks are not supported by this wrapper.

## Settings

`algorithm_range` belongs to `OMPLSolverConfig`; all other knobs below belong to
`VampRRTCSettings`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `algorithm_range` | `2.0` | Maximum extension length |
| `use_kdtree` | `True` | Use batch KD-tree; `False` selects exact linear scan |
| `use_halton` | `True` | Halton sampling with prime bases 3, 5, 7, ... |
| `halton_skip` | `0` | Starting sequence offset, up to 1,000,000,000 |
| `dynamic_domain` | `True` | Adapt node sampling radii |
| `radius` | `4.0` | Radius after the first collision |
| `alpha` | `0.0001` | Radius expansion/contraction factor |
| `min_radius` | `1.0` | Minimum radius |
| `balance` | `True` | Enable VAMP's tree size rule |
| `tree_ratio` | `1.0` | Threshold for that rule |
| `max_iterations` | `100000` | Maximum sampling attempts, including rejected samples |
| `max_samples` | `100000` | Combined capacity for both trees, including roots |
| `start_tree_first` | `True` | Initial tree ordering before the size rule |

Halton is deterministic and resets to `halton_skip` on each solve. Change the
offset for another deterministic search, or set `use_halton=False` for OMPL's
uniform RNG (controlled by `set_random_seed`). The integer Halton recurrence
avoids VAMP's float precision limit; its sequence does not reproduce VAMP's
eventual float reset/base rotation bit for bit.

Set `settings.use_kdtree = False` to scan every node in the selected tree.
This allocates only a flat node-index array for nearest-neighbor queries; no
KD-tree is built. Both modes use the same squared-distance calculation and
break distance ties by node index, so the same Halton inputs yield the same
path and collision-call count (when termination limits are not reached).
Insertion and queries allocate no heap memory in either mode. Linear queries
cost O(number of nodes × dimension); their lower indexing overhead can help
small trees but offers no general speed guarantee.

plainmp's existing OMPL RRTConnect uses `NearestNeighborsGNATNoThreadSafety`
in the installed OMPL 1.6 configuration. GNAT is also a tree index, using metric
distances rather than the axis partitions of a KD-tree. It is not a linear
scan. The existing planner is unchanged by `use_kdtree`.

To compare the two new backends with identical sampling:

```bash
.venv/bin/python example/bench/vamp_rrtc.py --compare-nn --runs 1000 --output /tmp/vamp_rrtc_nn.json
```

This is an algorithm/storage port, not VAMP's SIMD robot collision engine.
It retains double precision and plainmp's box/Euclidean motion resolution and
sample order. Like those existing validators, it checks discretized motions,
not continuous collision freedom. VAMP's SIMD rake validation, generated robot
kinematics, float vectorization, and Nigh implementation are not dependencies.
Capacity, zero-distance connections, exact terminal coordinates, and finite
parameter validation are handled explicitly rather than copying upstream edge
cases. Apache attribution and license are in `third/vamp`.

## Validation and measurement

```bash
cmake -S . -B build/native \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DPLAINMP_BUILD_TESTS=ON
cmake --build build/native --target _plainmp test_vamp_rrtc --parallel 4
ctest --test-dir build/native --output-on-failure
.venv/bin/python -m pytest tests/test_vamp_rrtc.py tests/test_ompl_solver.py
.venv/bin/python example/bench/vamp_rrtc.py --runs 1000 --output /tmp/vamp_rrtc.json
```

Follow `AGENTS.md` to link the built extension into the virtual environment.
Native tests compare the KD-tree against brute force in 1, 2, 7, 8, and 14
dimensions, exercise duplicate coordinates and pool reuse, and plan through
a synthetic wall with multiple goals. Python tests cover real Fetch collision
geometry, both validators, reversed trees, disabled heuristics, repeated solves,
limits, invalid inputs, deterministic sampling, IK, and refinement.

The benchmark alternates planner order, uses 10 warmup rounds and 1,000 measured
rounds per condition, and records external latency (including construction and
path conversion), internal time, collision calls, path length, and failures.
Every planner uses range 2.0 and the same collision checker and resolution.
Halton offsets advance by 1,000 per round so measurements cover different
searches. Uniform VampRRTC is also measured. No path refinement is enabled.

Linux `perf record -e cycles:u -F 999 --call-graph dwarf,16384` was also run on
20,000 Panda difficult solves, using an unstripped copy of the same Release
extension. The initial profile attributed 32.1% of user cycles to collision
validity, 16.9% to joint updates, 13.2% to group sphere caches, and 11.1% to
kinematic cache construction. KD-tree search accounted for 4.4%; the planner
solve routine itself for 1.8% (self samples). Existing collision and kinematics
code was left unchanged. The profile exposed unnecessary RNG/simplifier
construction in the new implementation; those are now initialized only when
uniform sampling/refinement is requested.

### Measured results

AMD Ryzen 7 7840HS, GCC 9.4.0, OMPL 1.6.0, Release with native CPU flags and LTO.

Median external latency in milliseconds; 1,000/1,000 successes in every row.

| Scene | Validator | RRTConnect | VampRRTC (Halton) | Speedup |
| --- | --- | ---: | ---: | ---: |
| fetch | box | 0.631 | 0.481 | 1.31x |
| fetch | euclidean | 0.819 | 0.589 | 1.39x |
| panda_easy | box | 0.190 | 0.184 | 1.03x |
| panda_easy | euclidean | 0.236 | 0.236 | 1.00x |
| panda_difficult | box | 0.672 | 0.773 | 0.87x |
| panda_difficult | euclidean | 0.814 | 1.014 | 0.80x |

A speedup below 1 means the new planner is slower. The VAMP exploration order
reduces collision checks on Fetch but increases them on these Panda scenes.
Unrefined paths are also longer here: for example Panda difficult/box has
median length 23.37 versus 19.92. These results do not establish superiority
across other scenes, robots, or resolutions. The current default algorithm
remains RRTConnect.

See [the benchmark summary](vamp_rrtc_benchmark.json) for p95, means,
collision-call counts, path lengths, and uniform-sampler measurements.

### KD-tree versus linear scan

With the same machine/settings and 1,000 paired trials in each of the six
scene/validator combinations, both backends succeeded on every trial.
Collision-call counts and path lengths matched for every pair. Median
external latency differed by at most 1.34%, with no consistent winner.
These scenes do not demonstrate a material planning-time benefit from the
KD-tree; this comparison does not cover very large search trees. See the
[paired benchmark summary](vamp_rrtc_nn_benchmark.json).

### KD-tree versus linear scan at range 0.5

Repeated the paired comparison at range 0.5, keeping the production default
at 2.0. Other settings, scenes, and Halton offsets are unchanged. Every
condition has 1,000 measured trials after 10 warmup rounds. All 12,000
solves succeeded; collision-call counts and path lengths matched in all
6,000 pairs. Times below are external planning latency in milliseconds.

| Scene | Validator | KD median | Linear median | Median speedup | KD p95 | Linear p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fetch | box | 5.139 | 5.719 | 1.113x | 13.336 | 19.428 |
| fetch | euclidean | 6.429 | 7.137 | 1.110x | 16.245 | 21.907 |
| panda_easy | box | 1.073 | 1.074 | 1.000x | 2.130 | 2.188 |
| panda_easy | euclidean | 1.276 | 1.273 | 0.998x | 2.556 | 2.543 |
| panda_difficult | box | 6.730 | 7.182 | 1.067x | 13.319 | 15.426 |
| panda_difficult | euclidean | 8.003 | 8.482 | 1.060x | 15.301 | 17.500 |

KD-tree wins on Fetch and Panda difficult at this smaller extension range;
Panda easy remains effectively tied. The advantage is larger in the slow
tail on Fetch. This is consistent with nearest-neighbor indexing becoming
more useful with smaller extensions, although node counts were not recorded.
Absolute planning time is higher than in the range-2.0 measurements.

Reproduce with:

```bash
.venv/bin/python example/bench/vamp_rrtc.py --compare-nn --range 0.5 --runs 1000 --output /tmp/vamp_rrtc_nn_range05.json
```

[Full summary](vamp_rrtc_nn_range05_benchmark.json).

### Halton versus uniform random sampling (range 2.0, KD-tree)

Direct comparison within VampRRTC; both samplers explicitly use KD-tree,
range 2.0, resolution 0.05, and no refinement. The two solvers alternate
execution order. Each of the six conditions has 10 warmup rounds and
1,000 measured rounds per sampler; all 12,000 measured solves succeeded.
Halton offsets advance by 1,000 per round. Uniform random sampling uses
OMPL RNG instances under global seed 42. These are different searches,
so collision counts and path lengths need not match.

External planning latency in milliseconds:

| Scene | Validator | Halton median | Random median | Halton p95 | Random p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| fetch | box | 0.491 | 0.491 | 1.406 | 1.327 |
| fetch | euclidean | 0.585 | 0.571 | 1.849 | 1.839 |
| panda_easy | box | 0.185 | 0.198 | 0.389 | 0.451 |
| panda_easy | euclidean | 0.248 | 0.260 | 0.527 | 0.607 |
| panda_difficult | box | 0.794 | 0.745 | 1.675 | 1.694 |
| panda_difficult | euclidean | 1.010 | 0.977 | 2.161 | 2.282 |

Median collision-call counts and unrefined Euclidean path lengths:

| Scene | Validator | Halton calls | Random calls | Halton length | Random length |
| --- | --- | ---: | ---: | ---: | ---: |
| fetch | box | 891.5 | 861.0 | 22.66 | 22.26 |
| fetch | euclidean | 1218.5 | 1100.5 | 22.69 | 21.67 |
| panda_easy | box | 544.5 | 585.5 | 14.38 | 15.92 |
| panda_easy | euclidean | 780.5 | 831.5 | 14.45 | 15.72 |
| panda_difficult | box | 2457.0 | 2267.5 | 23.37 | 24.27 |
| panda_difficult | euclidean | 3434.5 | 3266.5 | 23.40 | 24.18 |

No sampler wins every condition. Halton is faster on Panda easy and yields
shorter median Panda paths. Random has lower median latency on Panda
difficult, while Halton has a slightly lower p95 there. Fetch is tied in
box median latency and slightly favors random in Euclidean median latency.
These findings describe these repeated scenes/subsequences, not a general
guarantee about either sampling strategy.

```bash
.venv/bin/python example/bench/vamp_rrtc.py --compare-samplers --range 2.0 --runs 1000 --seed 42 --output /tmp/vamp_rrtc_samplers_range20.json
```

[Full summary](vamp_rrtc_sampler_benchmark.json).
