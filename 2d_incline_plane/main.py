import os
import sys
import time
from mpi4py import MPI
import numpy as np
import cupy
import amrex.space2d as amr
import torch
import torch.distributed as dist
import math
### Local imports
from particle_solver import multiple_lammps_runs
from torch_dist_comm import init_process_group, get_global_new_points
from model_framework import Model_Framework

torch.set_default_dtype(torch.float64)
# Initialize amrex::MPMD
amr.MPMD_Initialize_without_split([])
# Leverage MPI from mpi4py to perform communication split
app_comm = MPI.COMM_WORLD.Split(amr.MPMD_AppNum(), amr.MPMD_MyProc())
app_world_size = app_comm.Get_size()
app_rank = app_comm.Get_rank()

# IT IS IMPORTANT TO INITIALIZE AMREX BEFORE DDP FOR SETTING THE CORRECT CUDA.CURRENT_DEVICE

# Leverage regularization or not
regulrz = True

# Initialize AMReX
amr.initialize_when_MPMD([], app_comm)

# Important incflo setup inputs
min_conc = 0.0 # Minimum concentration of second fluid
# Create a ParmParse to collect this information
parser_amr = amr.ParmParse()
parser_amr.addfile("inputs_column_gravity")
min_conc = parser_amr.get_real("incflo.second_fluid.min_conc")

# DDP Initialize
init_process_group(app_rank, app_world_size,backend='nccl')
print(app_rank,dist.get_rank(),torch.cuda.current_device(), torch.cuda.device_count())
device_id = torch.cuda.current_device()
torch.set_default_device('cuda:'+str(device_id))
sys.stdout.flush()

# Create an MPMD Copier that gets the BoxArray information from (C++) app
copr_old = amr.MPMD_Copier(True)
new_ba = copr_old.box_array()
# Break the BoxArray further
new_ba.max_size(32)
new_dm = amr.DistributionMapping(new_ba)
# Create one-more Copier that sends the information
copr = amr.MPMD_Copier(new_ba,new_dm)

inputs_ncomp = 2
inputs_mf = amr.MultiFab(copr.box_array(), copr.distribution_map(),
                        inputs_ncomp, 0)
outputs_ncomp = 1
outputs_mf = amr.MultiFab(copr.box_array(), copr.distribution_map(),
                        outputs_ncomp, 0)

# The initial data used to train the PRE-TRAINED model
# NOTE: THESE VALUES ARE BASED ON LAMMPS SETUP
p_lammps = 1e-5
rho_lammps = 1.0
diam_lammps = 1.0
# Max Inertial Number for LAMMPS
I_max = 0.3162277638912201
# Min Inertial Number for LAMMPS
I_min = 0.0001581138785695657

I_train = torch.tensor([0.0001581138785695657, 0.0003162277571391314,
                        0.0015811388147994876, 0.003162277629598975])
mu_train = torch.tensor([0.33132310855, 0.32158857550000003,
                         0.32433879895, 0.3359988484])
mu_std_train = torch.tensor([7.44979293657024e-05, 9.891419038046772e-05,
                             0.00012580914522527538, 0.00033117288226813287])

# Initial value for neutral Inertial number
I_neutral = torch.min(I_train,0,keepdim=True)

mdls_frmwrk = Model_Framework(device_id = device_id,
		n_ensembles=10, lr=1e-3, new_pts_fctr=5.0,
                n_layers=2,layer_width=100, dropout_val=0.)

model_num_prev = 0
# Keep track of which model number
model_num = 0
# Check if a pre-trained model exists
tar_files = [fl for fl in os.listdir('.') if os.path.isfile(fl)]
tar_files = [fl for fl in tar_files if ".tar" in fl]
# Get the maximum number that is available
if len(tar_files) > 0:
    for fl in tar_files:
        fl_num = ""
        for char in fl:
            if char.isdigit():
                fl_num += char
        model_num = max(model_num,int(fl_num))

if model_num == 0:
    mdls_frmwrk.train(I_train,mu_train,mu_std_train,max_iter=1e+4,n_freq=5e+3)
    model_num += 1
    if app_rank == 0:
        mdls_frmwrk.save_current_ddp_state(model_num)
else:
    mdls_frmwrk.load_pretrained_ddp_state(model_num,map_location="cuda")
    # Also expand the data training dataset
    aa = np.loadtxt("lammps_run_data.txt", max_rows=model_num-1)
    I_train_1 = torch.tensor(aa[:,0])
    mu_train_1 = torch.tensor(aa[:,1])
    mu_std_train_1 = torch.tensor(aa[:,2])
    I_train = torch.cat([I_train, I_train_1],dim=0)
    mu_train = torch.cat([mu_train, mu_train_1],dim=0)
    mu_std_train = torch.cat([mu_std_train, mu_std_train_1],dim=0)


dist.barrier()
sys.stdout.flush()

rank_offset = amr.MPMD_MyProc() - amr.ParallelDescriptor.MyProc()

if (rank_offset == 0):
    mpmd_other_root = amr.ParallelDescriptor.NProcs()
else:
    mpmd_other_root = 0

end = -10
i = -1 # This variable is for call number
while (True):
    # Receive last_call info from the other root to only current root
    if (app_rank == 0):
        last_call = np.empty(1, dtype='i')
        MPI.COMM_WORLD.Recv(last_call,source=mpmd_other_root,tag=94)
        sys.stdout.flush()
        end = last_call[0]
    # Broadcast to every rank in python app
    end = app_comm.bcast(end,root=0)

    if end == 1:
        break
    # Update call value
    if end == 0:
        i += 1

    # "end" variable is also used to indicate
    # the number of components being transferred
    if end == -2:
        inputs_ncomp = 1
    else:
        inputs_ncomp = 2

    # Receive inputs_mf
    copr.recv(inputs_mf,0,inputs_ncomp)
    # Keep track of wall time in each computational step
    if (app_rank == 0) and (end == 0):
        start_time = time.time()

    new_pts_glbl = 10 # Random initial non-zero value
    while (new_pts_glbl != 0) and (end == 0):
        new_train_x_lcl_lst = []
        for mfi in inputs_mf:
            inputs_mf_array = inputs_mf.array(mfi).to_xp(
                                copy=False,order='C')
            # Create corresponding input tensor
            inputs_tensor = torch.as_tensor(inputs_mf_array[0:1,...])
            # The second component of MultiFab corresponds to concentration
            conc_second_tensor = torch.as_tensor(inputs_mf_array[1:2,...])
            # Clamp the Inertial Numbers to specified range
            inputs_tensor = inputs_tensor.clamp(min=I_min,max=I_max)
            # Reshaping inputs_tensor
            inputs_tensor = torch.reshape(inputs_tensor,(-1,))
            conc_second_tensor = torch.reshape(conc_second_tensor, (-1,))
            # Check how many points DO NOT meet the criterion
            # for this box
            new_train_x_lcl_mfi = mdls_frmwrk.get_local_new_points(
                                    inputs_tensor,I_train,
                                    conc_second_tensor, min_conc)
            new_train_x_lcl_lst.append(new_train_x_lcl_mfi)
        # Create a single new_train_x_lcl from the list
        new_train_x_lcl = torch.cat(new_train_x_lcl_lst,dim=0)
        new_train_x_lcl = torch.unique(new_train_x_lcl,dim=0)
        # Get new GLOBAL training points
        new_train_x_glbl = get_global_new_points(mdls_frmwrk,
                                new_train_x_lcl,
				I_train,
				app_rank,
                                app_world_size,max_new_pts=1)

        new_pts_glbl = new_train_x_glbl.size(dim=0)
        ##########################################################
        ### Check if there exists a point among training data that
        ### satisfies C <= 0 (C is condition in Barker, Gray (JFM, 2017))
        ##########################################################
        # Get here only if new_pts_glbl == 0
        if new_pts_glbl == 0 and regulrz and (model_num > model_num_prev):
            new_pts_glbl, I_neutral = mdls_frmwrk.check_inertial_num_stability(
                                                   I_train,I_neutral)
            # new_pts_glbl == 1 means none of the training points satisfy C(I)<= 0
            if ((new_pts_glbl == 1)
                or ((I_neutral == torch.max(I_train)) and
                     (I_neutral < I_max))):
                # Only single new training point
                new_pts_glbl = 1
                new_train_x_glbl,_ = torch.max(I_train,0,keepdim=True)
                new_train_x_glbl *= 2
                new_train_x_glbl = torch.minimum(new_train_x_glbl,
                                                 torch.tensor([I_max,]))
        ########################################################################
        # Generating data for these new points
        if new_pts_glbl > 0:
            new_train_y, new_train_y_std = multiple_lammps_runs(
                                            new_train_x_glbl,
                                            app_comm,"in.nve_jam_flow",
                                            p_lammps=p_lammps,
                                            rho_lammps=rho_lammps,
                                            diam_lammps=diam_lammps)
            # Update training datasets
            I_train = torch.cat([I_train,new_train_x_glbl],dim=0)
            mu_train = torch.cat([mu_train,new_train_y],dim=0)
            mu_std_train = torch.cat([mu_std_train,new_train_y_std],dim=0)
            # Re-train the models with new training points as well
            mdls_frmwrk.train(I_train,mu_train,mu_std_train,
                    max_iter=1e+4,n_freq=5e+3)
            model_num += 1
            if app_rank == 0:
                mdls_frmwrk.save_current_ddp_state(model_num)
            dist.barrier()
    #### The Neutral Inertial Number at hand, is one of the training points
    #### Let us refine to find a better approximation of the value
    #### This step is an iterative process
    if ((I_neutral != torch.min(I_train)) and (end == 0)
        and regulrz and (model_num > model_num_prev)):
        model_num_prev = model_num
        for _ in range(5):
            refine_range = torch.linspace(torch.min(I_train), I_neutral[0],10)
            _, I_neutral = mdls_frmwrk.check_inertial_num_stability(
                                       refine_range, I_neutral)
    ##########################################################################
    for mfi in inputs_mf:
        inputs_mf_array = inputs_mf.array(mfi).to_xp(
                                copy=False,order='C')
        # Create corresponding input tensor
        inputs_tensor = torch.as_tensor(inputs_mf_array[0:1,...])
        # Inertial Numbers should NOT be clamped if we are leveraging
        # regularized mu(I) model
        if not regulrz:
            inputs_tensor = inputs_tensor.clamp(min=I_min,max=I_max)
        # Reshaping inputs_tensor
        inputs_tensor = torch.reshape(inputs_tensor,(-1,))
        outputs_mf_array = outputs_mf.array(mfi).to_xp(
                                copy=False,order='C')
        # Create corresponding output tensor
        outputs_tensor = torch.as_tensor(outputs_mf_array)
        # Reshaping outputs_tensor
        outputs_tensor = torch.reshape(outputs_tensor,(-1,))

        if regulrz:
            # Create two different tensors
            outputs_tensor_1 = torch.zeros_like(outputs_tensor)
            outputs_tensor_2 = torch.zeros_like(outputs_tensor)
            # First branch will have the regularized version
            outputs_tensor_1 = mdls_frmwrk.low_inertial_num_branch(
                                             inputs_tensor,I_neutral)
            # Second branch will have LAMMPS based solution
            outputs_tensor_2 = mdls_frmwrk.final_inference(inputs_tensor)
            # Decision based population of outputs_tensor
            outputs_tensor = torch.where(inputs_tensor < I_neutral,
                                            outputs_tensor_1,
                                            outputs_tensor_2)
            # Placing a max clamp based on Barker, Gray (2017)
            outputs_tensor = outputs_tensor.clamp(max=1.414)
        else:
            outputs_tensor = mdls_frmwrk.final_inference(inputs_tensor)

        # Reshape outputs_tensor to outputs_mf_array's shape
        outputs_tensor = outputs_tensor.reshape(outputs_mf_array.shape)
        # Write the final data from outputs_tensor to outputs_mf_array
        outputs_mf_array[...] = cupy.from_dlpack(outputs_tensor)

    copr.send(outputs_mf,0,outputs_ncomp)
    if (app_rank == 0) and (end == 0):
        end_time = time.time()
        time_in_step = end_time-start_time
        time_step_fl = open("time_per_step.txt","a")
        time_step_fl.write(f"{i}\t{time_in_step}\n")
        time_step_fl.close()
        neutral_inertial_fl = open("neutral_inertial_num.txt","a")
        neutral_inertial_fl.write(f"{i}\t{I_neutral.cpu()[0]}\n")
        neutral_inertial_fl.close()

    # Save the models on rank 0 at regular intervals
    if ((i+1)%100 == 0) and (app_rank == 0):
        mdls_frmwrk.save_current_ddp_state(model_num)
    # Precautionary barrier
    dist.barrier()
    sys.stdout.flush()
    # Clearing GPU cache
    torch.cuda.empty_cache()

# Final trained models
if app_rank == 0:
    mdls_frmwrk.save_current_ddp_state(model_num)
dist.barrier()
# Explicitly destroy DDP process group
dist.destroy_process_group()
# Finalize AMReX
amr.finalize()
# Finalize AMReX::MPMD
amr.MPMD_Finalize()
