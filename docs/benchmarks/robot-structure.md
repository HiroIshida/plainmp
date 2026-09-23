# Robot structure preprocessing without JIT

Compared with the pushed four-state implementation (`5fb9380`), this change measured **1.027–1.068×** planner speedup across seven analytical-SDF scenes (roughly 3–6% less time). Point-cloud checks retain their existing scalar path and showed no improvement.

The runtime uses ordinary C++ and requires no generated code, compiler subprocess, dynamic library loader or new Python API.

## Changes

- Build a flat list of required FK links, following the existing external/self-collision group visit order. During each batch, evaluate only the necessary prefix. This removes repeated ancestor traversal and per-link readiness bookkeeping while retaining lazy evaluation and early exits. Rebuild on `set_sdf()` or a link-count change.
- Maintain the “all links include rotation” property when the model is loaded or a link is added. Classify supported primitive dynamic types when the flattened SDF list is rebuilt, instead of scanning all links and SDF types on every batch.
- Restore the visible scalar model by selecting the previously calculated SIMD lane. Do not repeat joint trigonometry or origin composition. Clear the same downstream caches as before.

The first two changes use model/constraint structure; the third reuses values within the current query. None learns collision labels or reuses a previous query's geometry.

## Measurements

| Scene | Planner speedup | Whole Python solve |
|---|---:|---:|
| panda | 1.063× | 1.058× |
| panda_hard | 1.060× | 1.052× |
| fetch_table | 1.044× | 1.041× |
| fetch_spheres4 | 1.028× | 1.024× |
| fetch_spheres9 | 1.068× | 1.059× |
| panda_boxes | 1.039× | 1.041× |
| fetch_table_tilted | 1.027× | 1.027× |
| fetch_table_cloud | 0.995× | 0.991× |
| fetch_table_cloud100k | 0.990× | 0.990× |

Each cell is the median of three seed-level ratios of median execution times. The baseline is the unchanged Release binary from `5fb9380`; the candidate uses the same compiler/flags and CPU. These are incremental improvements over the already accelerated batch implementation, not over the original scalar implementation.

AMD Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3, AVX2 kernel with `-ffp-contract=off` and no LTO, one thread pinned to CPU 2. RRTConnect, range 2, Euclidean resolution 1/32, self collision enabled, no IK or simplification in the timing runs. Per scene/seed/variant: 30 warmup plans and six alternating blocks of 100 measured plans. Seeds: 67867967, 67867979, 67867987. The RNG continues across the same scene order in both variants.

32,400 measured plans, all successful; all 16,200 matched pairs have identical paths and logical validity-call counts. CPU frequency and background load were not locked. Gains are small; the table is evidence for these workloads and machine, not a universal speedup claim. The point-cloud controls varied around 1% slower and did not use the modified batch path.

A screening run of the flat FK schedule alone was approximately unchanged (0.953–1.024× by scene). Adding state restoration produced 1.017–1.059× in that screening seed. The final table measures the combined implementation; it does not attribute the entire gain to topology preprocessing.

## Correctness

- 2,776,703 collision labels from recorded planner traces and independent/contact-adjacent configurations: zero differences from scalar references.
- 772 targeted API scenarios / 2,356 input states: batch tails, visible full/prefix state, changing uncontrolled joints and base pose, SDF replacement/translation, added links, custom primitives, point clouds and moving-base fallbacks.
- 1,200 planning-mode runs (600 baseline/candidate pairs): long/short Euclidean motion checks, box motion checks, generic constraints, shortcutting and call-budget exhaustion; all results/paths/counts match.
- Standalone C++ regression includes a branched robot, arbitrary joint axes/origins and interleaved external/self-only groups: 8,400 batches / 21,000 states, with restored link transforms checked. Both AVX2-enabled and disabled builds pass. The same standalone check also passes a Debug AddressSanitizer/UndefinedBehaviorSanitizer build (leak detection disabled).

The supported model mutation APIs (`add_new_link`, joint/base setters, `set_sdf`) preserve the cached facts. Direct edits to internal structural arrays must keep their cached summaries consistent. No general Python test suite was run.

## Reproduction

The standalone C++ check is enabled with `PLAINMP_BUILD_BATCH_COLLISION_CHECK=ON`. Optional `PLAINMP_BUILD_BATCH_QUERY_CHECK=ON` adds diagnostic bindings used for large query replay; it is off by default and does not change the public Python API.

Machine-readable results: [robot-structure-summary.json](robot-structure-summary.json). The local experiment directory `/home/h-ishida/tmp/plainmp_robot_specialization` contains the driver, full per-plan timings/hashes, reference query links and check logs. Its `build/baseline` and `build/specialized` symlinks select the previous and new native modules. Run `taskset -c 2 python3 scripts/screen.py --variants baseline specialized --tag simple-final --seeds 67867967 67867979 67867987 --blocks 6 --n 100 --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k`, then `python3 scripts/summarize_simple.py`.
