# Prune unneeded SIMD lanes during ordered collision queries

Starting at `62508b7`, the final implementation improves the seven analytical planning scenes by **1.026×** (geometric mean), or **2.421×** versus original master `7019b1d`. Whole Python solve time improves by 1.023× incrementally. This is a small additional gain after the previous optimizations. All three ordinary Release binaries were measured directly in the same experiment.

## Change and scope

`first_invalid_batch()` tells the eight-lane SIMD kernel that it needs an ordered prefix. When lane 5 collides, lanes 5–7 can be discarded immediately; lanes 0–4 still finish their checks. If a later obstacle or self-collision check rejects lane 2, the answer becomes 2. A collision in a later lane must never terminate inspection of earlier lanes.

The kernel expands each nonzero rejection mask upward from its lowest set bit using unsigned arithmetic. Both the active mask and the current narrow-phase mask lose those lanes. This can end sphere, obstacle, group and FK traversal earlier. Full-mask queries keep every lane's result. Visible kinematic state, sample order, logical call counts, collision arithmetic and tolerances are preserved.

A second change updates the masks only when a collision actually occurs. Common non-collision iterations avoid unconditional loop-carried mask updates. Assembly confirms a conditional branch around those updates. An early prototype that added pruning without this guard retired fewer instructions but took more cycles in its fixed-work experiment; its counters are retained in the JSON summary.

**The final optimization is restricted to AVX-512 eight-lane batches.** Applying the same extra branch to AVX2 caused a substantial regression in one screened scene. The four-lane arithmetic/control flow was restored; its kernel/restoration object's 7,980-byte `.text` section matches the pushed baseline byte for byte (hash in the JSON). It accepts the shared private mode argument but does not prune a suffix. The original scalar and point-cloud fallbacks remain in use.

## Perf evidence

The warmed, symbolized baseline Fetch-table profile attributed 36.90% of cycle samples to the eight-lane collision kernel and 11.00% to its FK pose helper. The same kernel accounted for 46.08% of sampled L1 load misses. Annotation highlighted sphere transformations and comparison/mask sequences, motivating experiments in memory layout and control flow. Sampling locations can skid; they are not measurements of individual instruction latency.

The final fixed-work experiment runs the same **10,000 plans / 32,986,037 logical validity calls** per run. Values below are medians of four runs per variant, in baseline/final/final/baseline order repeated twice. Perf's FIFO control starts counting after imports, setup and 30 warmups. All five events ran at 100%, without multiplexing. All runs, including timing variability, are retained.

| Counter | Baseline `62508b7` | Final |
|---|---:|---:|
| cycles:u | 27,670,696,890 | 26,503,005,082 |
| instructions:u | 73,055,112,360 | 71,537,681,504 |
| branch-misses:u | 82,435,364 | 78,924,188 |
| L1-dcache-loads | 26,401,706,123 | 26,122,739,655 |
| L1-dcache-load-misses | 1,283,913,078 | 1,159,111,822 |

The final version reduces cycles by **4.2%**, instructions by **2.1%**, branch misses by **4.3%**, and L1 load misses by **9.7%**. The final change does not repack the workspace; the miss reduction accompanies less work and changed generated control flow. These counters do not establish a universal cache improvement on other CPUs or scenes.

## Planning results

| Scene | Final / `62508b7` speedup | Final / original master speedup |
|---|---:|---:|
| panda | 1.011× | 2.491× |
| panda_hard | 1.042× | 2.402× |
| fetch_table | 1.041× | 2.451× |
| fetch_spheres4 | 1.015× | 2.387× |
| fetch_spheres9 | 1.024× | 2.322× |
| panda_boxes | 1.037× | 2.497× |
| fetch_table_tilted | 1.015× | 2.402× |
| fetch_table_cloud | 1.009× | 1.477× |
| fetch_table_cloud100k | 1.010× | 2.959× |

All **48,600** measured plans succeed. Each comparison contains **16,200** pairs with identical trajectory hashes and logical validity-call counts; independent query labels also match. Point clouds use the existing scalar/certificate path, and their incremental differences are noise-scale, not evidence of cloud acceleration.

Each cell is the median of three seed-level ratios of median plan times. Seeds: 67867967, 67867979 and 67867987. Each scene/seed/variant uses 30 warmups and six alternating blocks of 100 measured plans. RNG state continues through the same scene order. Clearance certificates reset before each plan.

Hardware/software: Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3, one thread pinned to CPU 2. RRTConnect range 2, Euclidean resolution 1/32, self collision enabled. CPU frequency and background activity are not locked. Small differences between individual variants should not be overinterpreted. Earlier timing runs that applied the optimization to both widths are archived separately and are not the final table above.

## Correctness

- **4,159,717** recorded/independent inputs through scalar and batch dispatch: zero label mismatches.
- **772** API/mutation/fallback cases / **4,388** states: full masks, first-invalid prefixes and visible state match.
- **600** planning-mode pairs for each final AVX2/AVX-512 build: long/short Euclidean, box, generic constraints, shortcutting and budget exhaustion match the baseline.
- **9,041** native batches / **42,077** states pass at dispatch widths 8, 4 and 1. Width 8 also passes ASan/UBSan with leak detection disabled and UBSAN halt-on-error.
- The native regression now includes all 502 nonempty rejection masks across lengths 1–8, with obstacles visited in reverse lane order, plus external/self-collision ordering cases. Earlier tests cover arbitrary axes/origins, changed controls, visible transforms and nextafter contact cases.
- No general Python test suite was run.

## Rejected experiments

The local study preserves unsuccessful prototypes and per-plan results. Native mask-register combinations, masked AABB chains, streaming sphere transformation, shape-first predicates, type-specialized narrow loops, flat sphere buffers, early AABB-axis exits, equal-radius runs, input transposition, forced shape inlining, direct ground predicates, small self-collision bounding blocks, deferred scalar-cache invalidation, whole-row mask reduction, packed geometry specifications and hoisted loop metadata were screened. They regressed representative scenes or did not show a consistent benefit sufficient to retain extra code/state. The final patch contains ordered lane pruning and conditional mask updates for the wider kernel.

## Reproduction

Artifacts, immutable baseline/prototype binaries, datasets, perf records and scripts are in `/home/h-ishida/tmp/plainmp_working_set_study`. Build selection uses `build/{master,baseline,candidate}`. The original master and pushed baseline are unchanged. Final symbolized sampling uses the immutable `build/final_profile` module, linked to OMPL 1.6. Exact binary hashes, raw counters, work counts and validation results are in [prefix-pruning-summary.json](prefix-pruning-summary.json).

```sh
# From the study directory:
taskset -c 2 python3 scripts/screen.py --variants master baseline candidate \
  --tag final2 --seeds 67867967 67867979 67867987 --blocks 6 --n 100 \
  --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 \
    panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k
python3 scripts/summarize_prefix.py
python3 scripts/profile.py --variant candidate --scene fetch_table --plans 10000 \
  --kind stat \
  --events cycles:u,instructions:u,branch-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u
```
