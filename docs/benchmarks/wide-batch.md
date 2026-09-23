# Wider collision batches and compact joint values

This change starts at `d50698c` and changes the execution granularity of analytical collision checks. On the Ryzen 7 7840HS, the seven analytical planning scenes improve by **1.216×** over that commit and **2.367×** over the original `7019b1d` master. These are directly measured geometric means. Whole Python solve time improves by 1.195× incrementally. The AVX2-only build improves by 1.019× incrementally.

## What perf showed and what changed

The symbolized, warmed Fetch-table profile of `d50698c` puts 32.90% of samples in the four-state collision kernel, 19.31% in group-center/FK calculation and 3.67% in scalar-state restoration. `perf annotate` concentrates samples around FK arithmetic and the comparison/mask/branch sequences of broad-phase checks. These sampling locations can skid; they are not exact instruction latency measurements.

1. **Eight states per collision batch on AVX-512F/DQ CPUs.** The obstacle/group traversal, branching and scalar state restoration are shared across eight states. The same FK, SDF and self-collision arithmetic is compiled separately for AVX2 and AVX-512 from one internal implementation header. Both ISA-specific translation units retain `-ffp-contract=off -fno-lto`. Portable callers check CPU support before dispatch. Batches of 2–4 still use AVX2; unsupported CPUs retain AVX2 or scalar execution. The C++ batch API accepts 1–8 states, splitting an eight-state request on AVX2-only builds.
2. **The exact motion endpoint joins the first batch.** It is still the first logical predicate, followed by the original interpolation sequence. This removes a separate scalar collision/FK pass. The returned first-invalid prefix determines the consumed call count and restored state. Later SIMD lanes can be computed speculatively and discarded, as before.
3. **Store joint-dependent values only.** For a revolute joint, keep the quaternion and read its fixed translation from the current model when needed. A prismatic joint keeps its displacement in the first vector slot. The local record falls from 512 to 256 bytes for eight lanes, and from 256 to 128 for four lanes. State restoration extracts four quaternion components instead of also extracting three replicated translation components. Origins, axes, uncontrolled transforms, base pose and control-list changes retain their existing mutation behavior; there is no freezing of numerical robot parameters.

The wider-only version first measured 1.163× over `d50698c` on the analytical suite, but increased L1 misses. The compact joint representation was added in response to those counters. Removing world-frame padding was also screened and discarded because it did not give a consistent further gain.

## Fixed-work hardware counters

Each entry below is the median of two runs of the same 10,000 Fetch-table plans / 32,986,037 logical validity calls. Order: baseline, wider-only, final, final, wider-only, baseline. Counters run only after imports, initialization and 30 warmups, using perf's FIFO control interface. Five events run at 100% without multiplexing. The earlier six-event experiment multiplexed at about 83% and is excluded from this table.

| Counter | `d50698c` | Wider-only | Final |
|---|---:|---:|---:|
| cycles:u | 33,514,376,352 | 28,632,740,664 | 27,259,326,638 |
| instructions:u | 106,937,384,808 | 76,537,967,268 | 73,057,207,584 |
| branch-misses:u | 88,306,300 | 82,069,779 | 82,467,530 |
| L1-dcache-loads | 34,798,803,922 | 28,291,140,498 | 26,412,467,054 |
| L1-dcache-load-misses | 707,243,372 | 1,487,891,166 | 1,275,975,314 |

The final implementation reduces instructions by **31.7%** and cycles by **18.7%** versus `d50698c`. L1 misses are 1.80× the baseline and 0.86× the wider-only version. The larger batch working set remains a cost: the speedup is chiefly from sharing traversal/control/restoration work, not a claim of universally improved cache hit rate. Raw counters and per-run work counts are included in the JSON summary.

## Planning results

| Scene | Final vs `d50698c` | Final vs original master | AVX2-only vs `d50698c` | Whole solve, final vs `d50698c` |
|---|---:|---:|---:|---:|
| panda | 1.242× | 2.484× | 1.033× | 1.211× |
| panda_hard | 1.229× | 2.365× | 1.038× | 1.210× |
| fetch_table | 1.210× | 2.373× | 1.016× | 1.196× |
| fetch_spheres4 | 1.225× | 2.351× | 0.991× | 1.212× |
| fetch_spheres9 | 1.207× | 2.250× | 1.011× | 1.186× |
| panda_boxes | 1.247× | 2.396× | 1.042× | 1.221× |
| fetch_table_tilted | 1.150× | 2.357× | 1.000× | 1.133× |
| fetch_table_cloud | 0.991× | 1.484× | 0.996× | 0.992× |
| fetch_table_cloud100k | 1.000× | 2.970× | 0.991× | 0.998× |

All **64,800** measured plans succeed. There are **16,200** pairs per comparison, with identical trajectory hashes and logical validity-call counts. Point clouds continue to use the existing scalar/certificate path; the near-unity incremental ratios are noise-scale differences, not a cloud acceleration result.

Each cell is the median of three seed-level ratios of median plan times. Four ordinary Release binaries alternate in blocks: original master `7019b1d`, pushed baseline `d50698c`, final source with AVX-512 disabled, and final default source. This is not a product of speedups measured in separate sessions.

Hardware/software: Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3, one thread pinned to CPU 2. RRTConnect range 2, Euclidean resolution 1/32, self collision enabled. Seeds 67867967, 67867979 and 67867987; 30 warmups; six blocks of 100 measured plans per scene, seed and variant. Clearance certificates reset before every plan. CPU frequency and background activity are not locked; small differences remain subject to noise. AVX-512 gains are specific to this measured CPU/workload and should be remeasured on other CPUs. `-DPLAINMP_ENABLE_AVX512_COLLISION=OFF` selects the four-state implementation.

## Correctness and rejected experiments

- 4,159,717 recorded/independent collision inputs through scalar and batch dispatch: zero label mismatches.
- 772 API/mutation/fallback cases / 4,388 states: full masks, first-invalid prefixes and visible state match.
- 600 planning-mode pairs for each final AVX2/AVX-512 variant: long/short Euclidean, box, generic constraints, shortcutting and budget exhaustion all match the pushed baseline.
- 8,537 native batches / 38,511 states pass at dispatch widths 8, 4 and 1, including arbitrary origins/axes, changed control lists, all rejection positions, tails and nextafter contact cases. Width 8 also passes ASan/UBSan (`-O1`, leak detection disabled, UBSAN halt-on-error).
- Lazy per-joint FK initialization regressed Fetch scenes; explicit mask-register logical intrinsics regressed the screen by roughly 6–16%; conditional sin/cos folding and compact world frames gave no consistent gain. These prototypes are archived and excluded.
- No general Python test suite was run.

## Reproduction

Local artifacts are under `/home/h-ishida/tmp/plainmp_fused_study`, including scripts, unchanged reference binaries, query datasets, rejected prototypes, per-plan results and perf records. `build/{master,baseline,avx2,candidate}` selects the final comparison. `build/batch8` preserves the wider-only implementation. Symbolized profiles use O3 debug builds, explicitly linked to the same OMPL 1.6 library; an initial profile that loaded an older OMPL is discarded. Counters and timing tables use ordinary Release binaries.

```sh
# From the study directory:
taskset -c 2 python3 scripts/screen.py --variants master baseline avx2 candidate \
  --tag final --seeds 67867967 67867979 67867987 --blocks 6 --n 100 \
  --scenes panda panda_hard fetch_table fetch_spheres4 fetch_spheres9 \
    panda_boxes fetch_table_tilted fetch_table_cloud fetch_table_cloud100k
python3 scripts/summarize.py
python3 scripts/profile.py --variant candidate --scene fetch_table --plans 10000 \
  --kind stat \
  --events cycles:u,instructions:u,branch-misses:u,L1-dcache-loads:u,L1-dcache-load-misses:u
```

Native checks use `PLAINMP_BUILD_BATCH_COLLISION_CHECK`; optional diagnostic query bindings use `PLAINMP_BUILD_BATCH_QUERY_CHECK`. Machine-readable results and exact binary hashes: [wide-batch-summary.json](wide-batch-summary.json).
