## License
At least for now, this project is licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/).

## installation
install scikit-robot
```bash
sudo apt-get install libspatialindex-dev freeglut3-dev libsuitesparse-dev libblas-dev liblapack-dev
pip install scikit-robot
```
install this
```bash
sudo apt install libeigen3-dev
pip install scikit-build
pip install plainmp
```

## for development
Do this before 
```bash
sudo apt-get install valgrind
pip install -e . -v
mkdir build
```
Then, do this every time you change the cpp files
```bash
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DUSE_VALGRIND=ON ..
make --
ln -sf $PWD/compile_commands.json ..
ln -sf $PWD/_plainmp.cpython-38-x86_64-linux-gnu.so ../python/plainmp
```
For performance tuning, e.g.
```bash
valgrind --tool=callgrind --instr-atstart=no python3 example/fetch_plan.py
```
with sandwitching the target code section with
```python
from plainmp import start_profiling, stop_profiling
start_profiling()
# target code section
stop_profiling()
```
