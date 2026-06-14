#!/usr/bin/env bash
set -euo pipefail

cmake --version
ninja --version
/opt/venv/bin/python --version
clang --version
mlir-opt --version
rocminfo > /tmp/rocminfo.txt
grep -m1 'Marketing Name:.*AMD Instinct MI300' /tmp/rocminfo.txt

export CMAKE_ARGS="-DAVE_LANG_BACKEND=rocm -DWITH_PYTHON=ON \
  -DCMAKE_C_COMPILER=/usr/local/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/clang++ \
  -DCMAKE_PREFIX_PATH=/usr/local;/opt/rocm"

uv pip install --python /opt/venv/bin/python -e ".[dev]"
/opt/venv/bin/python -m pytest test/
