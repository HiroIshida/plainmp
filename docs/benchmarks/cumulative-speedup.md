# Cumulative speedup against the original origin/master

Direct comparison of the unmodified original `origin/master` commit **7019b1d0544cbf3ab59ea8acae34dd7f36ea6a6d** against the optimized implementation **7f376759af8e64aa787bd6b1e1e3f58fb636cc8e**. The reference is the original locally recorded remote-tracking commit, not a freshly fetched later revision.

Across seven analytical-SDF scenes, planner speedup is **1.669–1.834×**, with an equally weighted geometric mean across scenes of **1.771×** (about **44% less execution time**). The corresponding whole Python `solve` speedup is **1.621–1.790×**, geometric mean **1.729×**. Point-cloud table scenes improve **1.171×** at 10,000 points and **1.399×** at 100,000 points.

These are direct measurements, not products of speedup ratios from earlier experiments. They include the earlier point-cloud clearance certificates, allocation removal, four-state SIMD collision checks, and the latest structure preprocessing/state-restoration changes. No JIT compilation or multithreaded collision checking is used.

| Scene | Planner speedup | Whole Python solve |
|---|---:|---:|
| panda | 1.834× | 1.759× |
| panda_hard | 1.815× | 1.790× |
| fetch_table | 1.717× | 1.693× |
| fetch_spheres4 | 1.732× | 1.707× |
| fetch_spheres9 | 1.669× | 1.621× |
| panda_boxes | 1.819× | 1.750× |
| fetch_table_tilted | 1.819× | 1.790× |
| fetch_table_cloud | 1.171× | 1.164× |
| fetch_table_cloud100k | 1.399× | 1.399× |

## Method and equivalence

- Same machine: AMD Ryzen 7 7840HS; GCC 9.4, OMPL 1.6, Release/O3, C++17, Eigen internal vectorization disabled, LTO for the normal translation units. The new AVX2 translation unit uses `-mavx2 -ffp-contract=off -fno-lto`, as in the implementation. No global `-march=native` or fast-math flags.
- The master worktree was checked out at the exact reference commit and built without source edits. Both modules use the same Python planning/scene code; those files are unchanged between the commits.
- One thread, pinned to CPU 2. RRTConnect, range 2, Euclidean resolution 1/32, self collision enabled, no IK or path simplification.
- Seeds 67867967, 67867979, 67867987; nine scenes in the table's order. Per seed/scene/variant: 30 warmup plans, then six alternating blocks of 100 measured plans. RNG state continues through scenes in identical order in each process.
- **32,400 measured plans, all successful. All 16,200 matched pairs have identical trajectory hashes and logical collision-check counts.** Independent configuration-label checks also have zero mismatches in both variants.
- Each table entry is the median of three seed-level ratios of median times. The geometric mean gives each analytical scene equal weight; it is not a workload-weighted average. Full per-seed medians and ratios are in the linked JSON.
- Model creation and warmup are excluded. Point-cloud certificates are reset before every solve in the new build, so the result does not depend on previous planning requests. This reset occurs before the wall timer; Python wall times cover the `solve` call itself.
- CPU frequency/background load were not fixed. Only two robot models and these synthetic scenes were tested; results should not be generalized to every robot/environment. Analytical SIMD gains require a supported fixed-base robot and AVX2 CPU; point clouds use the existing scalar/certificate path.

## Artifacts

[Machine-readable results, commits, binary hashes and compiler flags](cumulative-speedup-summary.json).

The local experiment is `/home/h-ishida/tmp/plainmp_master_comparison`. It contains `scripts/screen.py`, `scripts/benchmark.py`, `scripts/summarize.py`, all per-plan timings/hashes (`results/total_*.json`), and build metadata. `build/master` selects the unmodified reference binary, and `build/current` selects the optimized binary. Reproduce the interleaved experiment there with:

```sh
taskset -c 2 python3 scripts/screen.py --variants master current --tag total --seeds 67867967 67867979 67867987 --blocks 6 --n 100 --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k
python3 scripts/summarize.py
```

For an independent single-scene comparison, the repository's `example/bench/collision_data_path.py` also accepts explicit `--native`, `--scene`, `--seed`, `--plans`, `--output`, and `--reference` arguments. Its sequential-run timing protocol differs from the interleaved table above.
