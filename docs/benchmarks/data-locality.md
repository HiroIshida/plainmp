# Collision data locality

This change builds on `2f6fb07`, the pushed perf-guided implementation. It reduces the live collision working set and packs KD-tree nodes. No floating-point formulas, collision resolution, nearest-neighbor approximation, tree partitioning or planning parameters change.

## Changes and measurements

The four-state FK workspace now stores local poses only for controlled joints and world poses in traversal order, with compact steps identifying the original link, parent slot and controlled-joint slot. Uncontrolled-joint transforms are still read from the current model. The workspace rebuilds on SDF replacement, a link-count change or changes to the public C++ control-joint list (including reordering). A small contiguous byte array holds group readiness instead of clearing flags scattered through group geometry.

The group center uses the same quaternion-to-matrix calculation, but the nine SIMD rotation vectors are no longer retained. A group that reaches narrow-phase sphere checks reconstructs the matrix from its already computed world pose. This trades some repeated arithmetic for fewer persistent writes/loads and a smaller hot group record: **416 → 128 bytes** on this build. The broad-phase predicates and their order are unchanged.

KD-tree nodes are **40 → 32 bytes**, aligned to 32 bytes so no node straddles a 64-byte cache line. In preorder, a non-leaf's left child is always the following node; it needs no stored index. The right-child field distinguishes a leaf from a node with only a left child. Both distance traversals and nearest-point lookup stop immediately at leaves. Partitioning and visit order remain the same, including strict-less-than tie selection. The old public `KDNode` type is retained for source compatibility; the tree uses a private compact record. Persistent node storage falls 20%.

The node change is not solely a cache experiment: it also removes leaf split/child work. Standalone distance-query screens of 1k/10k/100k uniform-volume and thin-slab clouds measured approximately 1.1–1.3× against `2f6fb07`, with identical checksums. Planning gains are smaller because KD traversal is only part of the workload.

The final Fetch-table counter comparison runs the identical 10,000 plans / 32,986,037 logical validity calls, in baseline/new/new/baseline order. Median counts of the two runs per variant are:

| Counter | Pushed baseline | New |
|---|---:|---:|
| cycles:u | 35,690,940,106 | 33,711,191,025 |
| instructions:u | 106,612,015,126 | 106,928,057,013 |
| L1-dcache-loads | 36,028,939,199 | 34,788,156,552 |
| L1-dcache-load-misses | 1,197,796,994 | 770,742,298 |

L1 misses fall **35.7%**, with the perf-reported miss ratio falling from **3.32% to 2.22%**. Instructions increase only 0.30%; cycles fall 5.5%. This is consistent with reducing data traffic and cache pressure rather than eliminating arithmetic. All events run without multiplexing. These are user-space whole planning-loop counters, not just the collision kernel.

In the 100k cloud fixed-work run (5,000 plans / 16,634,462 logical calls), cycles fall 5.3%, while L1 misses fall only 3.0% and the L1 miss ratio stays near 0.74%. The compact-node benefit cannot be attributed solely to a higher L1 hit rate; node size/addressing and leaf work also change.


## Planning comparison

| Scene | Incremental planner | vs original master | Whole Python solve, incremental | Whole Python solve, vs master |
|---|---:|---:|---:|---:|
| panda | 1.048× | 1.979× | 1.044× | 1.872× |
| panda_hard | 1.026× | 1.978× | 1.022× | 1.947× |
| fetch_table | 1.064× | 1.993× | 1.062× | 1.949× |
| fetch_spheres4 | 1.072× | 1.935× | 1.071× | 1.869× |
| fetch_spheres9 | 1.050× | 1.845× | 1.048× | 1.788× |
| panda_boxes | 1.045× | 1.956× | 1.030× | 1.841× |
| fetch_table_tilted | 1.049× | 1.977× | 1.047× | 1.929× |
| fetch_table_cloud | 1.060× | 1.472× | 1.059× | 1.460× |
| fetch_table_cloud100k | 1.082× | 3.094× | 1.080× | 3.058× |

Analytical-scene geometric means: **1.051×** incremental and **1.951×** versus original master. All 48,600 measured plans succeed; each baseline/candidate comparison has 16,200 identical path and logical-call pairs. Independent collision-label differences: zero.

Each cell is the median of three seed-level ratios of median plan times. The incremental baseline is the unchanged Release binary of `2f6fb07`; the original-master baseline is `7019b1d`. All variants run in alternating blocks within one benchmark, so the cumulative results are direct measurements, not products of previous speedups.

Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3, one thread on CPU 2. The AVX2 translation unit retains `-ffp-contract=off -fno-lto`. RRTConnect range 2, Euclidean resolution 1/32, self collision enabled. Nine scenes, three seeds (67867967, 67867979, 67867987), 30 warmups and six blocks of 100 measured plans per scene/seed/variant. Clearance certificates are reset before each plan. CPU frequency and background load are not locked; small differences remain subject to noise.

## Alternatives screened

- Compaction of FK/readiness alone reduced Fetch-table L1 miss counts in a fixed-work screen but gave roughly unchanged median plan times across the seven analytical scenes. The rotation-matrix storage reduction supplied the more consistent gain.
- Packing immutable joint parameters into a new record gave no consistent improvement and was removed. Numeric joint parameters remain read from the existing model arrays.
- Removing the identity flag/padding from world transforms gave mixed results and was removed.
- Allocating an OMPL state header and its coordinates together added only 0.8% to the analytical-scene geometric mean in the three-seed prototype screen; the two point-cloud cases were about unchanged. This prototype is archived, but not part of the change.

## Correctness

- 4,159,717 recorded/independent collision inputs, through scalar and batch dispatch: zero label mismatches.
- 772 API/mutation/fallback cases / 2,356 states: masks, first-invalid prefixes and visible state match.
- 600 baseline/candidate planning-mode pairs: paths, success/failure and logical call counts match (long/short Euclidean, box, generic constraints, shortcutting and call-budget exhaustion).
- Native collision check: 8,490 batches / 21,270 states, including branched models, arbitrary axes/origins, restored poses and 90 batches changing the C++ control-joint list without resetting the SDF.
- Native KD-tree check: 10,862 exact distances match brute force, including empty/small trees, volumes, planes, lines, duplicates, boundary-adjacent and extreme-scale coordinates; explicit nearest-point tie-selection checks also pass.
- Both native checks pass AddressSanitizer and UndefinedBehaviorSanitizer (`-O1`, leak detection disabled).

## Reproduction and artifacts

Local study: `/home/h-ishida/tmp/plainmp_locality_study`. It contains scripts, original query references, raw per-plan timings/hashes, perf counter logs and sampled profiles, intermediate binaries and rejected prototypes. `source` points to the working tree; `build/{master,baseline,candidate}` select the unchanged original master, pushed perf implementation and this change. The archived `packed_state` variant was screened separately before adding the control-list invalidation guard.

```sh
# In the study directory:
taskset -c 2 python3 scripts/screen.py \
  --variants master baseline candidate --tag verified \
  --seeds 67867967 67867979 67867987 --blocks 6 --n 100 \
  --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 \
    panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k
python3 scripts/summarize.py
python3 scripts/profile.py --variant candidate --scene fetch_table --plans 10000 \
  --kind stat \
  --events cycles:u,instructions:u,L1-dcache-loads:u,L1-dcache-load-misses:u,cache-misses:u
```

Profiling is enabled through perf's FIFO control interface only after imports, model initialization and warmups. Counter comparisons use ordinary Release binaries. L1-miss sampling uses a symbolized O3 build and `L1-dcache-load-misses:u`, with a period of 100,000; sampling locations can skid and are not an exact attribution to individual instructions.

Native checks use the opt-in `PLAINMP_BUILD_KDTREE_CHECK` and `PLAINMP_BUILD_BATCH_COLLISION_CHECK` targets. Optional `PLAINMP_BUILD_BATCH_QUERY_CHECK` bindings support recorded-query/API validation. No general Python test suite is run.

Machine-readable results: [data-locality-summary.json](data-locality-summary.json).
