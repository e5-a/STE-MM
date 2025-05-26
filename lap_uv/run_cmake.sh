#!/usr/bin/zsh

# Please specify the path to your pybind11 installation.
# For example:
#   export pybind11_DIR=/home/.../venv/lib/python3.12/site-packages/pybind11/
export pybind11_DIR=/home/share/akira/tmp/STE-MM/venv/lib/python3.12/site-packages/pybind11/

mkdir build
cd build
cmake ../ --preset=default --fresh
ninja
cp ./lap_uv.cpython-*.so ../../srcs/
