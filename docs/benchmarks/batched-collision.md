# Batch collision checks across four configurations

The motion validator evaluates up to four intermediate configurations at once on
AVX2 CPUs. One SIMD lane holds one complete configuration. This batches joint
trigonometry, FK, sphere transforms, analytic SDF predicates, and self collision.
Sampling positions, traversal order, double precision, and collision tolerances
are unchanged. Later lanes may already have been evaluated when an earlier one
collides; those results are discarded. `n_call` counts the ordered prefix used by
the validator, not the number of SIMD lanes physically computed.

Runtime dispatch enables the kernel for fixed-base SphereCollisionCst instances
using the built-in analytic shapes. Point clouds, moving bases, custom SDF
subclasses, other constraint types, and unsupported CPUs use the scalar path.
The AVX2 translation unit is compiled separately without FMA contraction or LTO;
the rest of the module retains its portable instruction set. The public
kinematic state ends at the first rejected state, or at the final accepted one.

## Measurements

AMD Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, O3. Three held-out seeds ×600 RRTConnect
plans per scene and build, alternating every 100 plans. The baseline includes
the earlier Eigen::Ref and point-cloud changes. A second control disables batch
checking in this same source. Ratios are medians of seed-level median-time ratios.

|環境|前回版 ms|バッチ版 ms|前回版比|バッチ無効版比|Python solve全体：前回版比|
|---|---:|---:|---:|---:|---:|
|Panda：円柱|0.316|0.191|1.651×|1.647×|1.587×|
|Panda：円柱＋天井|1.200|0.726|1.647×|1.663×|1.623×|
|Fetch：箱テーブル|1.078|0.666|1.625×|1.636×|1.599×|
|Fetch：球4個|0.440|0.284|1.569×|1.602×|1.527×|
|Fetch：球9個|0.485|0.320|1.527×|1.550×|1.479×|
|Panda：傾いた箱|0.318|0.191|1.676×|1.669×|1.600×|
|Fetch：傾いた箱テーブル|0.952|0.557|1.737×|1.690×|1.710×|
|Fetch：点群1万点|1.738|1.745|0.996×|0.990×|0.992×|
|Fetch：点群10万点|3.198|3.198|0.998×|0.993×|1.002×|

48,600 measured plans; 32,400 matched comparisons with identical path hashes and
logical validity counts. The SIMD kernel matched 2,776,703 prior query labels,
including independently generated and near-contact states. Additional mutation,
fallback, batch-tail, visible-state, validator-mode, shortcut, and budget checks
passed. The standalone native regression also passed with batch dispatch enabled
and disabled (8,400 batches / 21,000 states each), and under ASan/UBSan with leak
detection disabled. Some point-cloud runs fluctuate by a few percent; that path remains
scalar. CPU frequency was not fixed. The first tilted-table seed had a slow old
baseline process, so the disabled-feature control is included as a cross-check.

## Build and check

The default build uses runtime AVX2 dispatch where available. To disable it:

```bash
cmake -S . -B build-no-avx -DPLAINMP_ENABLE_AVX2_COLLISION=OFF
cmake --build build-no-avx -j4
```

A standalone C++ regression program compares scalar and batched decisions and
checks the resulting kinematic state. It needs no downloaded robot models:

```bash
cmake -S . -B build -DPLAINMP_BUILD_BATCH_COLLISION_CHECK=ON
cmake --build build -j4
./build/check_batched_collision
```

For a paired planning benchmark, run `example/bench/collision_data_path.py` with
`--native`, `--scene`, `--seed`, `--plans`, and `--output` for each build; give the
second run `--reference` pointing at the first JSON. For example:

```bash
python3 example/bench/collision_data_path.py --native /path/to/base/_plainmp.so \
  --scene fetch_table --seed 49979687 --plans 600 --output base.json
python3 example/bench/collision_data_path.py --native /path/to/batch/_plainmp.so \
  --scene fetch_table --seed 49979687 --plans 600 --output batch.json --reference base.json
```

Use the actual platform-specific module filenames. Full timings and seed ranges
are in `batched-collision-summary.json`. Rebuild C++ consumers for the changed
class layout and methods.
