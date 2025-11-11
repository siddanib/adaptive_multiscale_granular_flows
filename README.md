# An adaptive, data-driven multiscale approach for dense granular flows

This repository has the code used to carry out the work in the manuscript
[[doi]](https://www.sciencedirect.com/science/article/pii/S0045782525005663) [[arXiv]](https://arxiv.org/abs/2505.13458v2).

## Dependencies

[AMReX](https://github.com/AMReX-Codes/amrex);
[pyAMReX](https://github.com/AMReX-Codes/pyamrex);
[AMReX-Hydro](https://github.com/AMReX-Fluids/AMReX-Hydro);
[incflo](https://github.com/siddanib/incflo/tree/tracer_two_fluid);
[PyLAMMPS](https://github.com/lammps/lammps);
[PyTorch](https://pytorch.org/);
[CuPy](https://cupy.dev/);
[pyblock](https://pyblock.readthedocs.io/en/latest/index.html);
[CMake](https://cmake.org/)

Please note that the linked **incflo** code is a forked version: [https://github.com/siddanib/incflo/tree/tracer_two_fluid](https://github.com/siddanib/incflo/tree/tracer_two_fluid)

## Python virtual environment

This work was carried out using GPUs on NERSC Perlmutter.
Sample script to create the environment is available [here](/nersc_perlmutter_specific/build_py_venv.sh)

## Installing PyLAMMPS into the environment using CMake

Assuming that we are inside ``lammps`` directory and the virtual environment is **active**.

```
   # Create a sub-folder named build
   mkdir build
   # Move into it
   cd build
   cmake -D BUILD_MPI=yes -D PKG_GRANULAR=on -D PKG_EXTRA-FIX=on -D BUILD_SHARED_LIBS=yes \
    -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV ../cmake
   # Create the shared library version
   make -j 10
   # Python installation
   make install-python
```
LAMMPS commit: ``591d20b00dfbafc92bb8e450952a5868f5eaae15``;
[Resource](https://docs.lammps.org/Python_module.html#)

## Installing pyAMReX into the environment using CMake

Assuming that we are inside ``pyAMReX`` directory and the virtual environment is **active**.

```
   cmake -S . -B build -DAMReX_SPACEDIM="1;2;3" -DAMReX_MPI=ON \
   -DAMReX_GPU_BACKEND=CUDA -DpyAMReX_amrex_src=/path/to/amrex
   # compile & install, here we use four threads
   cmake --build build -j 4 --target pip_install
```
[pyAMReX API](https://pyamrex.readthedocs.io/en/latest/install/cmake.html#compile);
[AMReX-MPMD Tutorial](https://amrex-codes.github.io/amrex/tutorials_html/MPMD_Tutorials.html#compiling-and-running-on-a-local-system)

## Compiling and running instructions for incflo

Use the GNUMake file in a setup folder to compile ``incflo`` (continuum solver).

**Ensure to build both incflo and pyAMReX using the same AMReX version.**

## Citation

```
@article{SIDDANI2025118294,
title = {An adaptive, data-driven multiscale approach for dense granular flows},
journal = {Computer Methods in Applied Mechanics and Engineering},
volume = {446},
pages = {118294},
year = {2025},
issn = {0045-7825},
doi = {https://doi.org/10.1016/j.cma.2025.118294},
url = {https://www.sciencedirect.com/science/article/pii/S0045782525005663},
author = {B. Siddani and Weiqun Zhang and Andrew Nonaka and John Bell and Ishan Srivastava}
}
```
## Acknowledgments

This work was supported by the U.S. Department of Energy (DOE), Office of Science,
Office of Advanced Scientific Computing Research, Applied Mathematics Program under contract No. DE-AC02-05CH11231.
This research used resources of the National Energy Research Scientific Computing Center (NERSC),
DOE Office of Science User Facility supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231 using NERSC award ASCR-ERCAP0026881.
