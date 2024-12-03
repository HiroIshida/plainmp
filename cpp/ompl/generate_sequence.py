# plainmp - library for fast motion planning
#
# Copyright (C) 2024 Hirokazu Ishida
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

def find_farthest_sequence_fast(N):
    result = [0] * N
    used = [False] * N
    used[0] = True
    current_size = 1
    
    for i in range(1, N):
        max_min_distance = -1
        best_num = -1
        for num in range(N):
            if used[num]:
                continue
            min_distance = float('inf')
            for j in range(current_size):
                distance = abs(num - result[j])
                if distance < min_distance:
                    min_distance = distance
            if min_distance > max_min_distance or (min_distance == max_min_distance and num > best_num):
                max_min_distance = min_distance
                best_num = num
        result[i] = best_num
        used[best_num] = True
        current_size += 1
    return result

if __name__ == "__main__":
    # NOTE: tried compile-time computation, but it was too slow ...
    n_precopmute = 128
    cpp_code = """ 
#include <array>
#include <cstddef>
using SequenceTable = std::array<std::array<size_t, {N}>, {N}>;
constexpr SequenceTable SEQUENCE_TABLE =
    """.format(N=n_precopmute)
    cpp_code += "{{\n"

    for i in range(n_precopmute):
        ret = find_farthest_sequence_fast(i + 1)
        remain = n_precopmute - len(ret)
        ret = list(ret) + [0] * remain
        cpp_code += "    {{ {} }},\n".format(", ".join(str(x) for x in ret))
    cpp_code += "}};\n"

    this_dir_path = Path(__file__).parent
    output_path = this_dir_path / "sequence_table.hpp"
    with open(output_path, "w") as f:
        f.write(cpp_code)
