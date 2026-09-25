#!/bin/bash
set -euo pipefail
clang_format=${CLANG_FORMAT:-clang-format}
if [[ "$("$clang_format" --version)" != *"version 14."* ]]; then
    echo "format.sh requires clang-format 14 (selected: $clang_format)" >&2
    exit 1
fi
find cpp tests/cpp -type f \( -name "*.cpp" -o -name "*.hpp" \) | xargs "$clang_format" -i -style=Chromium

find src/plainmp tests example -type f -name "*.py" -print0 | xargs -0 python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
for module in "src/plainmp tests example"; do
    python3 -m isort $module --profile black
    python3 -m black --line-length 100 --target-version py38 --required-version 22.6.0 $module
    python3 -m flake8 $module
done

codespell --ignore-regex '(\#.*$|//.*$)' cpp src/plainmp tests example
