import numpy as np
import torch
from lammps import lammps, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR
import pyblock
import math

def incflo_lammps_converter(incflo_inrtl_num,p_lammps=1e-5,
        rho_lammps=1.0,diam_lammps=1.0,
        min_sr_lammps=1e-6,max_sr_lammps=1e-3):
    # Convert incflo Inertial Number to LAMMPS strain rate
    incflo_inrtl_num /= diam_lammps
    incflo_inrtl_num *= math.sqrt(p_lammps/rho_lammps)
    return incflo_inrtl_num.clamp(min=min_sr_lammps,max=max_sr_lammps)

def multiple_lammps_runs(I_tensor, app_comm, input_file,
                         p_lammps=1e-5,rho_lammps=1.0,
                         diam_lammps=1.0):
    # I_tensor contains the new Inertial numbers
    n_sr = I_tensor.size(dim=0)
    mu_tensor = torch.zeros((n_sr,))
    mu_std_tensor = torch.zeros_like(mu_tensor)
    for i in range(n_sr):
        mu_tensor[i],mu_std_tensor[i] = single_lammps_run(I_tensor.cpu()[i],app_comm,
                                               input_file, p_lammps=p_lammps,
                                               rho_lammps=rho_lammps,
                                               diam_lammps=diam_lammps)

    return mu_tensor, mu_std_tensor

def single_lammps_run(I_value,app_comm, input_file,
                      p_lammps=1e-5,rho_lammps=1.0,
                      diam_lammps=1.0):
    # Convert Inertial Number to LAMMPS strain rate
    sr_value = I_value/diam_lammps
    sr_value *= math.sqrt(p_lammps/rho_lammps)

    lmp = lammps(comm=app_comm)
    # Read the required file thhat common info of all runs
    lmp.file(input_file)
    # Let us first apply only compression
    lmp.command("fix rheo all deform/pressure 1 "+
                "x pressure/mean ${Press} ${Pgain} y pressure/mean ${Press} ${Pgain} "+
                "z pressure/mean ${Press} ${Pgain} normalize/pressure yes remap x")
    lmp.command("thermo 0")
    lmp.command("thermo_style custom step cpu cpuremain ke v_vfac v_pKE v_muKE")
    lmp.command("thermo_modify lost warn norm no flush yes")
    lmp.command("compute_modify thermo_temp dynamic yes")
    # Run initial steps with timestep 0.2
    n_steps = 100000
    time_step = 0.2
    lmp.command(f"timestep {time_step}")
    lmp.command(f"run {n_steps} upto start 0")
    # Run specific information
    lmp.command("unfix rheo")
    # gamma_dot needs to be 2 times strain rate
    lmp.command(f"variable gamma_dot equal {2.0*sr_value}")
    lmp.command("fix rheo all deform/pressure 1 xy erate/rescale ${gamma_dot} "+
                "x pressure/mean ${Press} ${Pgain} y pressure/mean ${Press} ${Pgain} "+
                "z pressure/mean ${Press} ${Pgain} normalize/pressure yes remap x flip yes")

    max_steps = int(0.5/(sr_value*time_step))
    max_steps = max(max_steps,int((10**5)/time_step))
    max_steps += n_steps

    lmp.command(f"timestep {time_step}")
    lmp.command(f"run {max_steps} upto start 0")
    n_steps = max_steps

    # Run until gamma = 1.0 using timestep 0.02
    # Gamma left to reach 1.0, if not reached
    gamma_left = 1.0-(sr_value*n_steps*time_step)
    # New time_step
    time_step = 0.02
    if gamma_left > 0.:
        # New steps
        max_steps = int(gamma_left/(sr_value*time_step))
        max_steps += n_steps

        lmp.command(f"timestep {time_step}")
        lmp.command(f"run {max_steps} upto start 0")
        n_steps = max_steps

    # Need to start sampling after this point
    steps_at_a_time = 200000
    max_steps += steps_at_a_time
    Nfreq = 10
    Nevery = Nfreq
    Nrepeat = 1

    lmp.command(f"fix muKEcalc all ave/time {Nevery} {Nrepeat} {Nfreq}\t"
        +"v_vfac v_pKE v_muKE\t"
        +"mode scalar ave one\t"
        +f"start {n_steps} file stress.evol")

    lmp.command(f"timestep {time_step}")
    lmp.command(f"run {max_steps} upto start 0")
    n_steps = max_steps
    # Read the data
    mu_data = np.loadtxt("stress.evol")

    reblock_mu = pyblock.blocking.reblock(mu_data[:,3])
    opt = pyblock.blocking.find_optimal_block(len(reblock_mu),reblock_mu)

    while np.isnan(opt[0]):
        lmp.command(f"fix muKEcalc all ave/time {Nevery} {Nrepeat} {Nfreq}\t"
                    +"v_vfac v_pKE v_muKE\t"
                    +"mode scalar ave one\t"
                    +f"start {n_steps} append stress.evol")
        max_steps += steps_at_a_time
        lmp.command(f"timestep {time_step}")
        lmp.command(f"run {max_steps} upto start 0")
        n_steps = max_steps

        mu_data = np.loadtxt("stress.evol")
        reblock_mu = pyblock.blocking.reblock(mu_data[:,3])
        opt = pyblock.blocking.find_optimal_block(len(reblock_mu),reblock_mu)

    final_mu_mean = reblock_mu[opt[0]].mean
    final_mu_std = reblock_mu[opt[0]].std_err
    # Write the final data to a file ONLY on rank 0
    if app_comm.Get_rank() == 0:
        lammps_fl = open("lammps_run_data.txt","a")
        lammps_fl.write(f"{I_value}\t{final_mu_mean}\t{final_mu_std}\n")
        lammps_fl.close()
    # Close this lammps instance
    lmp.close()

    return final_mu_mean.item(), final_mu_std.item()

if __name__ == "__main__":
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    # Set separate GPU for each rank
    torch.set_default_device('cuda:'+str(rank))

    #final_mu, final_mu_std = single_lammps_run(3e-4,comm,"in.nve_jam_flow")
    #print(final_mu,final_mu_std)

    I_tensor = torch.tensor([1e-6/math.sqrt(1e-5),1e-5/math.sqrt(1e-5),1e-3/math.sqrt(1e-5)])
    mu_tensor, mu_std_tensor = multiple_lammps_runs(
                                I_tensor,comm,"in.nve_jam_flow",
                                p_lammps=1e-5,
                                rho_lammps=1.0,
                                diam_lammps=1.0)

    print(mu_tensor,mu_std_tensor)
