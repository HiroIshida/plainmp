set -e
GPP_VERSION=$(g++ -dumpversion | cut -d. -f1)

# NOTE: this GPP version reqiurement is actually not mandatory for just building the wheel.
# However, recent g++ versions makes sizable performance improvements.
if [ "$GPP_VERSION" -lt 13 ]; then
  echo "Error: g++ version must be 13 or greater. Current version is $GPP_VERSION."
  exit 1
else
  echo "g++ version is $GPP_VERSION. Proceeding..."
fi

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
