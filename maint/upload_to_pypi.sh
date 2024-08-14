set -e
cd /tmp
if [ -d plainmp ]; then
    rm -rf plainmp
fi
git clone https://github.com/HiroIshida/plainmp.git
cd plainmp
tag=$(git describe --exact-match --tags HEAD 2>/dev/null)

if [[ $tag =~ ^v ]]; then
    version="${tag#v}"
    setup_version=$(grep -oP "(?<=version=['\"])[^'\"]*(?=['\"])" setup.py)

    if [[ "$version" == "$setup_version" ]]; then
        echo "Tag version '$version' matches setup.py version '$setup_version'. continuing..."
        git submodule update --init --recursive
        python3 setup.py sdist
        twine upload --skip-existing -u __token__ -p $PYPI_TOKEN dist/*
        exit 0
    else
        echo "Tag version '$version' does not match setup.py version '$setup_version'."
        exit 1
    fi
else
    echo "No tag found at HEAD, or the tag doesn't start with 'v'."
    exit 1
fi
