#!/bin/bash

# Array of search and replace patterns
patterns=(
    's/MatrixXd/MatrixXf/g'
    's/VectorXd/VectorXf/g'
    's/Matrix3Xd/Matrix3Xf/g'
    's/Matrix2Xd/Matrix2Xf/g'
    's/Quaterniond/Quaternionf/g'
    's/Matrix3d/Matrix3f/g'
    's/Matrix4d/Matrix4f/g'
    's/Vector3d/Vector3f/g'
    's/Vector4d/Vector4f/g'
    's/AngleAxisd/AngleAxisf/g'
    's/Affine3d/Affine3f/g'
)

# Function to apply sed patterns to a file
apply_patterns() {
    local file=$1
    for pattern in "${patterns[@]}"; do
        sed -i "$pattern" "$file"
    done
}

# Process .cpp and .hpp files in the cpp directory
for ext in cpp hpp; do
    while IFS= read -r -d '' file; do
        apply_patterns "$file"
    done < <(find cpp -name "*.$ext" -print0)
done

# Process .cpp, .hpp, and .h files in the third/urdf_parser directory
for ext in cpp hpp h; do
    while IFS= read -r -d '' file; do
        apply_patterns "$file"
    done < <(find third/urdf_parser -name "*.$ext" -print0)
done

echo "Conversion complete."
