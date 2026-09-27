# OMPL RRTConnect, PlainmpRRTC and VampRRTC

**Historical measurement before removal of `vamp_rrtc`.** The implementation,
Python/C++ registrations, VAMP-specific tests and `third/vamp` have been removed.
The benchmark results below are preserved unchanged as the basis for that decision.

This compares implementations inside plainmp using the same robot collision
checker and motion validators. It does not measure upstream VAMP SIMD collision
checking. All results below come from one rerun requested after the user reported
a YouTube background workload during the preceding run. Earlier measurements
are not used to calculate these ratios.

## Conditions

- AMD Ryzen 7 7840HS, GCC 9.4.0, OMPL 1.6.0, Release, `-march=native`, LTO.
- Same Fetch table, Panda easy and Panda difficult scenes as the preceding
  benchmarks, with fixed joint-space goals.
- Range **2.0**. OMPL uses its default GNAT nearest-neighbor implementation;
  PlainmpRRTC and VampRRTC use exact KD-trees and 100,000-node combined capacity.
- Box widths 0.05 on every axis or Euclidean resolution 0.05.
- No refinement or timeout; validity-call budget 1,000,000.
- 10 warmups + 1,000 measurements per configuration; seed 42, rotating order.
- A fresh planner is created by every high-level solve. Internal timing excludes
  constructors and output Eigen matrix allocation/copy. VAMP initial setup/pool
  allocation runs before its timer; plainmp pools are allocated in its constructor.
  OMPL solve-time setup and allocations remain included in its existing timer.
- Search, collision checking and native path reconstruction are included.
- OMPL and PlainmpRRTC use uniform sampling. VampRRTC uses its default dynamic
  domains and balancing; both default Halton and uniform sampling are measured.
  Halton skip is `trial * 1000`, including warmups.
- Uniform streams differ between implementations; exploration variability is
  included in these comparisons. All **24,000 solves succeeded**.

## Internal medians and speedups

Milliseconds. Parentheses show **OMPL time / implementation time**, calculated
from unrounded medians. Above 1 means faster than OMPL; below 1 means slower.

| Scene | Validator | OMPL | plainmp_rrtc | VAMP Halton | VAMP uniform |
|---|---|---:|---:|---:|---:|
| panda_easy | box | 0.177 | 0.131 (1.35x) | 0.144 (1.23x) | 0.161 (1.10x) |
| panda_easy | euclidean | 0.222 | 0.171 (1.30x) | 0.198 (1.12x) | 0.209 (1.06x) |
| panda_difficult | box | 0.656 | 0.542 (1.21x) | 0.747 (0.88x) | 0.722 (0.91x) |
| panda_difficult | euclidean | 0.815 | 0.709 (1.15x) | 1.000 (0.81x) | 0.923 (0.88x) |
| fetch | box | 0.722 | 0.547 (1.32x) | 0.430 (1.68x) | 0.429 (1.68x) |
| fetch | euclidean | 0.764 | 0.690 (1.11x) | 0.525 (1.46x) | 0.516 (1.48x) |

## Internal p95

Milliseconds, with OMPL-relative p95 ratios.

| Scene | Validator | OMPL | plainmp_rrtc | VAMP Halton | VAMP uniform |
|---|---|---:|---:|---:|---:|
| panda_easy | box | 0.374 | 0.308 (1.22x) | 0.352 (1.06x) | 0.411 (0.91x) |
| panda_easy | euclidean | 0.475 | 0.375 (1.26x) | 0.479 (0.99x) | 0.533 (0.89x) |
| panda_difficult | box | 1.629 | 1.331 (1.22x) | 1.591 (1.02x) | 1.741 (0.94x) |
| panda_difficult | euclidean | 2.050 | 1.741 (1.18x) | 2.158 (0.95x) | 2.263 (0.91x) |
| fetch | box | 2.067 | 1.596 (1.29x) | 1.451 (1.42x) | 1.422 (1.45x) |
| fetch | euclidean | 2.409 | 2.081 (1.16x) | 1.791 (1.34x) | 1.648 (1.46x) |

## Search effort and path length

Medians of validity-call counts and unrefined joint-space Euclidean path lengths.

| Scene | Validator | OMPL calls / length | plainmp calls / length | VAMP Halton calls / length | VAMP uniform calls / length |
|---|---|---:|---:|---:|---:|
| panda_easy | box | 476 / 13.16 | 493 / 13.24 | 544 / 14.38 | 598 / 16.07 |
| panda_easy | euclidean | 642 / 12.82 | 670 / 12.93 | 780 / 14.45 | 816 / 15.48 |
| panda_difficult | box | 1752 / 20.31 | 1774 / 20.25 | 2457 / 23.37 | 2372 / 24.39 |
| panda_difficult | euclidean | 2368 / 19.94 | 2421 / 20.10 | 3434 / 23.40 | 3212 / 24.19 |
| fetch | box | 1467 / 21.52 | 1313 / 20.89 | 892 / 22.66 | 882 / 22.36 |
| fetch | euclidean | 1730 / 20.82 | 1788 / 20.98 | 1218 / 22.69 | 1166 / 22.47 |

## Decision and benchmark provenance

PlainmpRRTC improved internal median time over OMPL in all six tested settings
(1.11–1.35x). VampRRTC was stronger on Fetch (1.46–1.68x), but slower than OMPL
on Panda difficult. PlainmpRRTC is retained as the optional pooled RRTConnect
implementation. The existing default remains OMPL RRTConnect with range 2.0.

The measured source was commit `b33d5f2`, plus a timing-only change to the VAMP
wrapper: run `setup_->setup()` before starting the internal timer, and record
elapsed time before allocating/filling the output Eigen matrix. The comparison
cycled OMPL, plainmp, VAMP Halton and VAMP uniform in rotating order, with the
conditions listed above. The VAMP algorithm and settings were unchanged.

VAMP's historical settings were dynamic domains enabled, radius 4.0,
alpha 0.0001, minimum radius 1.0, balancing enabled, tree ratio 1.0,
100,000 iterations, and start tree first. KD-tree was enabled in both variants.

[Summary JSON](compare_rrtc_benchmark.json) retains timings, OMPL-relative
ratios, call counts, path lengths and the machine/build conditions from the
rerun. These results are not regenerated by the removal commit. Repeating the
VAMP measurements requires restoring its historical implementation and timing
changes; the current build no longer exposes that planner.

The unchanged scene definitions have moved to
[rrtc_scenes.py](../example/bench/rrtc_scenes.py). The retained OMPL/plainmp
benchmark runs against the current build:

```sh
.venv/bin/python example/bench/plainmp_rrtc.py \
    --runs 1000 --range 2.0 --seed 42 \
    --output /tmp/plainmp_rrtc.json
```
