module purge
# required dependencies
module load PrgEnv-gnu/8.5.0
module load cray-mpich/8.1.30
module load craype/2.7.32
module load craype-x86-milan
module load gpu/1.0
module load craype-accel-nvidia80
module load cudatoolkit/12.4
module load cmake/3.30.2
# Required for pyAMReX
module load cray-python/3.11.5

# necessary to use CUDA-Aware MPI and run a job
export CRAY_ACCEL_TARGET=nvidia80

# optimize CUDA compilation for A100
export AMREX_CUDA_ARCH=8.0

# optimize CPU microarchitecture for AMD EPYC 3rd Gen (Milan/Zen3)
# note: the cc/CC/ftn wrappers below add those
export CXXFLAGS="-march=znver3"
export CFLAGS="-march=znver3"
# compiler environment hints
export CC=cc
export CXX=CC
export FC=ftn
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=CC
