# plainmp - library for fast motion planning
#
# Copyright (C) 2024 Hirokazu Ishida
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

def find_farthest_sequence_fast(N):
    """
    Generate a sequence of N unique numbers where each new number maximizes its minimum distance
    from all previously placed numbers in the sequence.

    The algorithm works by iteratively selecting numbers (0 to N-1) such that at each step,
    the chosen number has the maximum possible minimum distance to all numbers already in 
    the sequence. In case of ties, the larger number is selected.

    * On bisection validation trick:
    The resulting sequence is intended to be used from motion validator in sampling-based
    motion planning. The applied heuristic here is that if nearing indices say (i, j)are
    more likely to be have the same collision status. So, to detect a path is in collision
    or not, it is better to sample the points in the sequence that are farthest from each other.
    OMPL's https://github.com/ompl/ompl/blob/main/src/ompl/base/src/DiscreteMotionValidator.cpp
    uses a similar method.

    Args:
        N (int): The length of the desired sequence. Must be a positive integer.
            The function will generate a permutation of numbers from 0 to N-1.

    Returns:
        list: A list of integers containing a permutation of numbers from 0 to N-1,
            arranged to maximize the minimum distance between each newly added number
            and all previously placed numbers.

    Example:
        input: N = 20
        output: [0, 19, 10, 5, 15, 17, 13, 8, 3, 18, 16, 14, 12, 11, 9, 7, 6, 4, 2, 1]
    """
    result = [0] * N
    used = [False] * N
    used[0] = True
    current_size = 1
    
    for i in range(1, N):  # first number is always 0, thus we start from 1
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


def test_find_farthest_sequence_fast():
    for n in range(1, 129):
        seq = find_farthest_sequence_fast(n)
        assert len(seq) == n
        assert len(set(seq)) == n
        assert seq[0] == 0
        if n > 1:
            assert seq[1] == n - 1


if __name__ == "__main__":
    # NOTE: tried compile-time computation, but it was too slow ...
    n_precopmute = 128
    cpp_code = """ 
// This file is generated by generate_sequence.py
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
