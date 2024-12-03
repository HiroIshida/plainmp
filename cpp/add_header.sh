#!/bin/bash

cat > header.txt << 'EOL'
/*
 * plainmp - library for fast motion planning
 *
 * Copyright (C) 2024 Hirokazu Ishida
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

EOL

has_header() {
    grep -q "plainmp - library for fast motion planning" "$1"
    return $?
}

find . -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) -print0 | while IFS= read -r -d '' file; do
    if ! has_header "$file"; then
        echo "Adding header to $file"
        temp_file=$(mktemp)
        cat header.txt "$file" > "$temp_file"
        mv "$temp_file" "$file"
    else
        echo "Header already exists in $file, skipping"
    fi
done

rm header.txt
