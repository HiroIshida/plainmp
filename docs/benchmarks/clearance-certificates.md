# Online point-cloud clearance certificates

`SphereCollisionCst` reuses the last clearance measured for each robot sphere
and point cloud. If the sphere center moves less than that clearance, another
KD-tree query is unnecessary. Lookups use squared distance and subtract a
1e-9 m guard from the reuse radius.

The enclosing sphere also caches **negative** clearance: when overlap of the
enclosing sphere is guaranteed to persist, the checker proceeds directly to
the individual spheres. That coarse overlap never rejects a configuration by
itself. Only the ordinary narrow phase can report a collision.

Cheap AABB tests precede lookups. Analytic SDFs and self collision keep their
existing predicates. The cache is populated online, without extra KD-tree
queries to construct certificates. Only cloud pairs allocate entries: the
Fetch scene below uses 2,560 bytes of certificates.

The point clouds must remain fixed between resets. `set_sdf()` invalidates all
certificates even if passed the same pointer. `reset_clearance_cache()` clears
the history explicitly; it is available in C++ and Python. Normal configuration
changes retain the history. As with the existing mutable kinematic caches, a
constraint instance must not be queried concurrently.

## Initial experiment (2026-09-23)

Six builds, seven scenes, three seeds, 500 RRTConnect solves per seed and scene:
63,000 successful plans, with 52,500 paired path hashes and validity call counts
matching the unmodified checker. Caches started empty at every solve. More
than 3.49 million recorded configurations and another 220,500 random,
interpolated, and near-contact configurations were checked for identical results.

| Fetch table geometry | Planner speedup | KD-tree queries avoided |
| --- | ---: | ---: |
| 10,000 points | 1.129x | 41.0% |
| 100,000 points | 1.371x | 41.1% |

Speedups are medians of three seed-level median ratios. The corresponding
Python solve wall-time speedups, including cache resets, were 1.126x and 1.372x.
The two clouds sample the same artificial table volume, with fixed geometry
seed 424242 and point radius 0.002 m. These are preliminary results, not a
general claim about real sensor data. Timing excludes model loading and KD-tree
construction. Machine: Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, Release/O3/LTO,
one pinned CPU, sequential measurements with rotated build order.

Applying certificates indiscriminately to analytic shapes was slower, despite
84–91% hit rates for enclosing spheres. The cache often replaced only a few
AABB comparisons. Restricting reuse to expensive cloud queries made the
tradeoff favorable. Shuffling the query stream removed the speedup, supporting
temporal coherence as the source of the benefit.

The archived [summary](clearance-summary.csv) and [figure](clearance-speedup.svg)
describe the research prototype before this integration. Its base was commit
`6d264f4` plus local Jacobian/CMake edits; those unrelated edits and experimental
instrumentation are not part of this branch. Recheck performance for this
integrated build using the standalone benchmark below. `baseline` in the CSV
means a control with instrumentation APIs but reuse disabled; the ratios are
relative to the original source.

After integration, another 3,713,922 recorded and boundary configurations
matched the reference labels, and SDF replacement checks passed. One additional
500-plan block per cloud size retained identical paths and call counts and
measured 1.115x / 1.468x speedups. These single-block checks are kept separate
from the three-seed prototype results above.

## Reproduce a paired comparison

Build this branch and the parent revision in separate directories. Use the
same script to load each native module explicitly, avoiding editable-install
ambiguity. For example, with paths to the two compiled modules:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 example/bench/cloud_clearance.py \
  --native /path/to/baseline/_plainmp.cpython-38-x86_64-linux-gnu.so \
  --points 100000 --output /tmp/baseline.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 example/bench/cloud_clearance.py \
  --native /path/to/branch/_plainmp.cpython-38-x86_64-linux-gnu.so \
  --points 100000 --output /tmp/branch.json --reference /tmp/baseline.json
```

Repeat for both point counts and seeds 179424673, 179425003, and 179425039,
rotating execution order. The script checks identical paths and validity call
counts. The normal robot-model dependencies of `FetchSpec` are required.

The underlying idea is established prior work:
[Bialkowski et al., IJRR 2016, Safety Certificates](https://ottelab.com/html_stuff/pdf_files/Bialkowski.Otte.ea.IJRR16.pdf).
The implementation currently selects CloudSDF by a fixed rule; it does not
learn which predicates are worth caching.
