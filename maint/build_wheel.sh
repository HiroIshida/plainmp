set -e
PYTHON_VERSIONS=("system" "3.9.19" "3.10.10" "3.11.9" "3.12.5")
CURRENT_DIR=$(cd $(dirname $0); pwd)
DIST_DIR=${CURRENT_DIR}/dist

for PYTHON_VERSION in "${PYTHON_VERSIONS[@]}"; do
    TMP_DIR=/tmp/plainmp-${PYTHON_VERSION}
    rm -rf ${TMP_DIR}
    mkdir -p ${TMP_DIR} && cd ${TMP_DIR}

    git clone git@github.com:HiroIshida/plainmp.git && cd plainmp
    git submodule update --init --recursive

    pyenv local ${PYTHON_VERSION}
    pip3 install scikit-build -v

    echo "Python version: $(python --version)"
    python3 setup.py bdist_wheel

    cp dist/*.whl ${DIST_DIR}
done
