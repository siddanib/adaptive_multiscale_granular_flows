#!/bin/bash

venv_name="pyamrex-gpu-nersc"
source ./perlmutter_gpu.profile
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
python3 -m venv $venv_name
source $venv_name/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m pip cache purge
python3 -m venv $venv_name
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
python3 -m pip install --upgrade packaging
python3 -m pip install --upgrade wheel
python3 -m pip install --upgrade setuptools
python3 -m pip install --upgrade cython
python3 -m pip install --upgrade numpy
python3 -m pip install --upgrade pandas
python3 -m pip install --upgrade scipy
MPICC="cc -target-accel=nvidia80 -shared" python3 -m pip install --upgrade mpi4py --no-cache-dir --no-build-isolation --no-binary mpi4py
python3 -m pip install --upgrade openpmd-api
python3 -m pip install --upgrade matplotlib
python3 -m pip install --upgrade yt
python3 -m pip install --upgrade cupy-cuda12x
python3 -m pip install pyblock
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
