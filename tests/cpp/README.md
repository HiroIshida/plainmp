# C++ tests

Build and run the tests with:

```sh
cmake -S . -B build-tests -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DPLAINMP_BUILD_PYTHON=OFF
cmake --build build-tests --parallel
ctest --test-dir build-tests --output-on-failure
```

CMake FetchContent downloads a pinned GoogleTest revision on the first
configuration. GoogleTest is only needed when `BUILD_TESTING=ON`; normal package
builds leave it OFF. The inline test robots require no external model assets.

The certificate tests check conservative intervals against ordinary point
queries, contact boundaries, self collision, invalidation, and fallback.
Motion-validator tests require actual skipping and compare results, logical
query counts, and final joint state.

Interval pruning is an experimental **runtime** option, OFF by default:

```python
config = OMPLSolverConfig(enable_interval_pruning=True)
```

The lower-level C++/Python `ValidatorConfig` has the same field.
`simplify_path(..., enable_interval_pruning=True)` enables it for standalone
simplification. Settings are read when constructing a planner/validator.
The previous `PLAINMP_ENABLE_INTERVAL_PRUNING` CMake option has been removed.

`tests/test_motion_certificate.py` compares paths, query counts, and final
joint state with the runtime option ON/OFF in the same build, using identical
random seeds and query budgets. These are regression tests, not a formal proof
of equivalence for all inputs; time-based termination can change outcomes.
