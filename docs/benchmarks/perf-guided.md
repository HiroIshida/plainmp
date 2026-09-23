# Optimization guided by Linux perf

This round starts from the pushed robot-specialized implementation, `3d9965a` (code change `7f37675`). It retains exact collision predicates, the original KD-tree construction and nearest-point tie order, and the existing scalar/SIMD dispatch rules.

## Measured causes and changes

### SIMD store forwarding

User-space cycle sampling put 33% of Fetch-table cycles in `BatchCollisionWorkspace::group_center`, including its inlined FK. Annotated assembly showed GCC's generic tuning splitting 256-bit aggregate stores into two 128-bit stores, followed by full-width loads. An explicit `wide::V` copy assignment keeps those stores at vector width; no arithmetic expression or compiler floating-point setting changes.

A fixed Release workload of 10,000 Fetch-table plans / 32,986,037 logical validity calls gave:

| Counter | Before | Copy fix |
|---|---:|---:|
| cycles | 35,899,235,322 | 33,518,319,220 |
| instructions | 107,239,979,494 | 106,613,637,282 |
| non-forwardable store/load conflicts (`ls_bad_status2.stli_other`) | 1,253,054,325 | 234,375,257 |
| successful store-to-load forwarding (`ls_stlf`) | 3,325,113,810 | 4,581,884,119 |

Conflicts fell 81.3%, cycles 6.6%, while the instruction count changed only 0.6%. This supports the forwarding-stall diagnosis beyond sampled instruction locations, which are subject to skid. All counters ran without multiplexing. The fixed-counter run isolates the copy fix, not the final combined change.

### Exact KD-tree pruning

In the 100k-point table workload, `KDTree::nearest` accounted for 58.5% of sampled cycles. `sqdist()` formerly tracked a nearest point that its caller never used and pruned using only the current split plane.

The new distance-only traversal keeps the best distance by value and tail-iterates the far child. For queries outside the cloud's bounds, it carries three coordinate distance lower bounds, initialized from the cloud's bounding box and tightened at far-child splits. Their squared norm can exclude branches that the single-plane test cannot. It computes that norm in the same operation order as the point distance, rather than updating a squared-distance sum with subtraction. There is no approximation tolerance.

Queries inside the cloud's box use the lighter single-plane traversal. An unconditional region-bound implementation regressed on uniformly filled volume clouds; this dispatch removes that regression. A standalone screen of 1k/10k/100k uniformly filled clouds measured 1.05–1.09× for interior queries versus the original distance routine, with identical distance checksums.

The cloud's bounds add 48 bytes per tree and one O(n) scan during construction, not a bounding box per node. Tree building and robot/scene initialization are outside the planning timings. `query()` and the tree partitioning remain unchanged, including the selected point when several points tie.

## Final planning comparison

| Scene | Incremental planner | vs original master | Whole Python solve, incremental | Whole Python solve, vs master |
|---|---:|---:|---:|---:|
| panda | 1.062× | 1.924× | 1.055× | 1.838× |
| panda_hard | 1.081× | 1.930× | 1.076× | 1.892× |
| fetch_table | 1.073× | 1.882× | 1.071× | 1.864× |
| fetch_spheres4 | 1.058× | 1.818× | 1.054× | 1.772× |
| fetch_spheres9 | 1.072× | 1.778× | 1.065× | 1.725× |
| panda_boxes | 1.050× | 1.934× | 1.041× | 1.822× |
| fetch_table_tilted | 1.074× | 1.956× | 1.065× | 1.906× |
| fetch_table_cloud | 1.191× | 1.389× | 1.189× | 1.380× |
| fetch_table_cloud100k | 2.004× | 2.816× | 1.993× | 2.792× |

Analytical-scene geometric means: **1.067×** incremental and **1.888×** versus original master. All 48,600 measured plans succeeded; each baseline/candidate comparison has 16,200 identical path and logical-call pairs. Independent collision-label mismatches: zero.

Each cell is the median of three seed-level ratios of median times. The original baseline is the unmodified `7019b1d` build; the incremental baseline is the previously pushed `3d9965a` build. These are direct comparisons within the same alternating benchmark run, not products of earlier ratios.

AMD Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3, CPU 2, one thread. The AVX2 translation unit uses `-ffp-contract=off -fno-lto`. RRTConnect range 2, Euclidean resolution 1/32, self collision enabled. Three seeds (67867967, 67867979, 67867987), nine scenes, 30 warmups per scene, six alternating blocks of 100 plans per scene/seed/variant. Clearance certificates are reset before each plan. CPU frequency and background load are not locked, so small differences remain subject to noise. Results describe these workloads and this machine.

## Correctness and remaining profile

- 4,159,717 recorded/independent collision inputs (2,776,703 analytical and 1,383,014 cloud inputs), checked through scalar and batch dispatch: zero label mismatches.
- 772 API/mutation/fallback scenarios / 2,356 input states: masks, first-invalid prefixes and visible state match.
- 1,200 planning-mode runs / 600 baseline-candidate pairs: paths, logical calls and success/failure agree, including call-budget exhaustion.
- Native batched collision regression: 8,400 batches / 21,000 states, including restored link transforms.
- Native KD-tree regression: 10,860 distances equal brute force exactly, including empty/small trees, volumes, thin slabs, planes, lines, duplicate points, nextafter-adjacent points and underflow/overflow-scale coordinates.
- Both native checks pass AddressSanitizer and UndefinedBehaviorSanitizer (`-O1`, leak detection disabled).

A fixed Release counter run of 3,000 cloud100k plans / 10,000,323 logical validity calls reduced internal time from 11.429 s to 5.599 s. Cycles fell from 51,715,490,948 to 25,516,203,403; instructions from 157,513,720,567 to 85,049,197,395; branch misses from 270,288,501 to 75,077,625. All counters ran without multiplexing.

In the final warmed cloud profile, nearest-distance traversal takes about 20.2% of cycles, down from the earlier nearest-point traversal's 58.5%. Remaining exclusive shares include scalar collision checking (23.7%), FK cache construction (17.1%), group-center construction (8.6%) and joint-angle updates (7.3%). These sample percentages are descriptive and have different total-time denominators; the fixed-work counter run above measures the actual reduction.

For the analytical Fetch table, the final profile still spends about 33.7% in the batched collision kernel and 22.4% in group-center/FK work. This round has not eliminated the cost of those computations and does not establish a hardware-independent performance ceiling.

## Reproduction

The local study directory `/home/h-ishida/tmp/plainmp_perf_study` contains profiling/benchmark drivers, saved intermediate binaries, source snapshots of rejected experiments, raw perf data, counter logs and per-plan timings/hashes. `source` points to the working tree and `build/{master,baseline,candidate}` select the three Release binaries.

```sh
# In the study directory:
taskset -c 2 python3 scripts/screen.py --variants master baseline candidate \
  --tag final --seeds 67867967 67867979 67867987 --blocks 6 --n 100 \
  --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 \
    panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k
python3 scripts/summarize.py
python3 scripts/profile.py --variant profile_final --scene fetch_table_cloud100k \
  --plans 3000 --tag final-cloud100k
```

The profiler starts disabled and uses perf's FIFO control interface to enable events only after imports, model construction and warmups. It records `cycles:u` at 999 Hz with DWARF call stacks. The symbolized profile builds use O3, DWARF4 and frame pointers, with stripping disabled. Final timings use ordinary Release binaries, not profiling or sanitizer builds.

Native regression checks are opt-in CMake targets: `PLAINMP_BUILD_KDTREE_CHECK=ON` / `check_kdtree` and `PLAINMP_BUILD_BATCH_COLLISION_CHECK=ON` / `check_batched_collision`. `PLAINMP_BUILD_BATCH_QUERY_CHECK=ON` adds diagnostic bindings used by the replay/API checks. No general Python test suite is run.

Machine-readable final results: [perf-guided-summary.json](perf-guided-summary.json).
