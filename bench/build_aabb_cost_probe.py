"""Build a diagnostic module that repeats cache work without changing its result.

Use an existing Release CMake build with compile_commands.json and the Unix
Makefiles generator. PLAINMP_COST_REPETITIONS controls the extra repetitions;
PLAINMP_COST_POSITIONS_ONLY=1 repeats only the position calculation. Compare
the two slopes to estimate the extra cost of constructing the AABB.
"""

import argparse
import concurrent.futures
import json
import shlex
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ref", required=True, help="Git revision of the reference implementation")
    args = parser.parse_args()
    build = args.build_dir.resolve()
    out = args.output_dir.resolve()
    relative = "cpp/plainmp/constraints/primitive_sphere_collision"
    source = subprocess.check_output(["git", "show", f"{args.ref}:{relative}.cpp"], text=True)
    header = subprocess.check_output(["git", "show", f"{args.ref}:{relative}.hpp"], text=True)
    source = source.replace("#include <cmath>", "#include <cmath>\n#include <cstdlib>", 1)
    source = source.replace(
        "namespace plainmp::constraint {",
        """namespace plainmp::constraint {
static const int cost_repetitions = [] {
  const char* s = std::getenv("PLAINMP_COST_REPETITIONS");
  return s ? std::atoi(s) : 0;
}();
static const bool cost_positions_only = [] {
  const char* s = std::getenv("PLAINMP_COST_POSITIONS_ONLY");
  return s && std::atoi(s) != 0;
}();
""",
        1,
    )
    marker = "        group.create_aabb_cache(kin_);"
    if source.count(marker) != 1:
        raise RuntimeError("Reference source must have exactly one external AABB creation site")
    source = source.replace(
        marker,
        marker
        + """
        for (int repeat = 0; repeat < cost_repetitions; ++repeat) {
          group.is_sphere_positions_dirty = true;
          if (cost_positions_only) {
            group.create_sphere_position_cache(kin_);
          } else {
            group.is_aabb_dirty = true;
            group.create_aabb_cache(kin_);
          }
          // Prevent elimination of repeated stores under LTO.
          asm volatile("" : : "g"(&group) : "memory");
        }""",
    )
    include = out / "include/plainmp/constraints"
    include.mkdir(parents=True, exist_ok=True)
    (include / "primitive_sphere_collision.hpp").write_text(header)
    generated = out / "primitive_sphere_collision.cpp"
    generated.write_text(source)
    replacements = {}
    commands = []
    for entry in json.loads((build / "compile_commands.json").read_text()):
        command = shlex.split(entry["command"])
        if "-o" not in command:
            continue
        original_object = command[command.index("-o") + 1]
        if "CMakeFiles/_plainmp.dir/" not in original_object:
            continue
        depfile = build / (original_object + ".d")
        changed_source = entry["file"].endswith("constraints/primitive_sphere_collision.cpp")
        # Recompile dependent translation units with the matching header too,
        # including inline methods and the vector element layout in bindings.
        if not (
            changed_source
            or not depfile.exists()
            or "primitive_sphere_collision.hpp" in depfile.read_text()
        ):
            continue
        obj = out / Path(original_object).name
        command[command.index("-o") + 1] = str(obj)
        if changed_source:
            command[command.index("-c") + 1] = str(generated)
        command.insert(1, "-I" + str(out / "include"))
        replacements[original_object] = str(obj)
        commands.append(command)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda cmd: subprocess.run(cmd, cwd=build, check=True), commands))
    link = shlex.split((build / "CMakeFiles/_plainmp.dir/link.txt").read_text())
    module = out / Path(link[link.index("-o") + 1]).name
    link[link.index("-o") + 1] = str(module)
    link = [replacements.get(arg, arg) for arg in link if arg not in ("-s", "-Wl,-s")]
    subprocess.run(link, cwd=build, check=True)
    print(module)


if __name__ == "__main__":
    main()
