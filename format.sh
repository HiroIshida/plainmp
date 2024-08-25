#!/bin/bash
find . -maxdepth 2 -type f \( -name "*.cpp" -o -name "*.hpp" \) | xargs clang-format -i -style=Chromium
find cpp/collision -maxdepth 2 -type f \( -name "*.cpp" -o -name "*.hpp" \) | xargs clang-format -i -style=Chromium

find . -name "*py"|xargs python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
for module in "python example"; do
    python3 -m isort $module --profile black
    python3 -m black --line-length 100 --target-version py38 --required-version 22.6.0 $module
    python3 -m flake8 $module
done
