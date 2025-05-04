#!/bin/bash
#SBATCH -N 2 # Total number of nodes
#SBATCH -n 8 # Total number of tasks
#SBATCH -c 4 # number of processors per MPI task
#SBATCH -C gpu
#SBATCH -G 8 # Total number of GPUs
#SBATCH -q regular
#SBATCH -J mpmd_test
#SBATCH -t 24:00:00
#SBATCH -A mpXXX

##### PyTorch DDP related ##############################################
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12340
########################################################################

#export NCCL_DEBUG=INFO
module purge
source ./perlmutter_gpu.profile
# Activate the virtual environment
source /path/to/pyamrex-gpu-nersc/bin/activate

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
# Taken from WarpX
export MPICH_OFI_NIC_POLICY=GPU
GPU_AWARE_MPI="amrex.use_gpu_aware_mpi=0 amrex.the_arena_is_managed=1"

#run the application:
srun --multi-prog --cpu_bind=cores --gpu-bind=none ./myrun.conf

deactivate
