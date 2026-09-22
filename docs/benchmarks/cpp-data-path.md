# Avoid configuration allocations in collision checks

OMPL passes a mapped state vector to `IneqConstraintBase::is_valid`. The former
`const Eigen::VectorXd&` parameter materialized that map, allocating and copying
one vector per query. Passing `Eigen::Ref<const Vector>` through the constraint
and kinematics input path removes that temporary for contiguous inputs.

Collision predicates, FK arithmetic, sample sequences, and planner settings are
unchanged. Python read-only arrays, strided arrays, lists, and column vectors are
still accepted (non-contiguous input may require a conversion). C++ consumers
must rebuild for the changed function signatures.

## Measurements

AMD Ryzen 7 7840HS, GCC 9.4, OMPL 1.6, O3/LTO, Eigen vectorization disabled,
fixed CPU 2, RRTConnect, Euclidean resolution 1/32, range 2, self collision on.
Three seeds ×600 plans per condition, alternating processes every 100 plans.
Ratios are medians of the three seed-level median-time ratios; greater than one
is faster. Each query stream and resulting path matches the original.

|環境|対照 ms|割り当て除去|＋native命令生成|＋alignment|＋メンバ再配置|
|---|---:|---:|---:|---:|---:|
|Panda：円柱|0.341|1.052×|1.083×|1.061×|1.059×|
|Panda：円柱＋天井|1.241|1.043×|1.076×|1.048×|1.039×|
|Fetch：箱テーブル|1.110|1.035×|1.053×|1.047×|1.033×|
|Fetch：球4個|0.471|1.035×|1.038×|1.020×|1.030×|
|Fetch：球9個|0.511|1.046×|1.066×|1.037×|1.058×|
|Panda：傾いた箱|0.331|1.059×|1.090×|1.065×|1.068×|
|Fetch：傾いた箱テーブル|0.929|1.023×|1.044×|1.004×|1.025×|
|Fetch：点群1万点|1.967|1.031×|1.032×|1.020×|1.027×|
|Fetch：点群10万点|4.697|1.014×|1.023×|1.011×|1.005×|

Only allocation removal is implemented in this branch. Native code generation,
alignment, and member layout were measured as separate experimental variants.
See `cpp-data-path-summary.csv` for seed ranges and comparisons to both the
untouched build and the API-only control. CPU frequency and memory layout were
not fixed; some individual seeds regressed. A diagnostic malloc counter observed
20,000 allocations for 20,000 mapped queries before the input change and zero
afterward. Allocation counting was disabled during timing.

The six-variant experiment completed 97,200 plans, with 81,000 matched path/call
count comparisons. The minimal port on this branch matched 4,159,717 collision
queries (including independent and near-contact configurations), 9,600 FK poses
under mixed updates, 2,000 Python input-conversion calls, and 4,500 planning pairs
against the previous point-cloud branch. These checks were targeted experiments,
not a run of the repository's entire test suite.

## Reproduce a paired planning run

Build both revisions normally. Run the script twice with explicit module paths;
it avoids silently selecting a different editable installation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
python3 example/bench/collision_data_path.py \
  --native /path/to/base/build/_plainmp.cpython-38-x86_64-linux-gnu.so \
  --scene fetch_table --seed 982451653 --plans 500 --output base.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 \
python3 example/bench/collision_data_path.py \
  --native /path/to/feature/build/_plainmp.cpython-38-x86_64-linux-gnu.so \
  --scene fetch_table --seed 982451653 --plans 500 --output feature.json \
  --reference base.json
```

The script asserts matching paths and validity-call counts. It also supports
Panda, rotated boxes, four/nine spheres, and 10k/100k point-cloud tables. Any
point-cloud clearance state is reset before every solve on both builds.

For a local build on the deployment CPU, the experimental native configuration
can be requested through the existing CMake flags variable:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_INIT="-march=native -ffp-contract=off"
cmake --build build-native -j4
```

This is optional and CPU-specific; the default portable build is retained.
The native experiment used no fast-math flags, contained no FMA instructions,
and matched the tested FK outputs bitwise.
