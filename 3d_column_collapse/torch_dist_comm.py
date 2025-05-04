import torch
import torch.distributed as dist
## Local imports
from sampling_optimization import uniform_subsampling

"""
Initialize the torch.distributed package.
Recommended to use backend="gloo" for CPUs
and backend="nccl" for GPUs
hostname CANNOT be localhost for multi-node runs
"""
def init_process_group(rank,world_size,backend="gloo"):
    # Let us use this TCPStore in dist.init_process_group
    dist.init_process_group(backend=backend,world_size=world_size,
            rank=rank)

    if rank == 0:
        print(f"Is DDP initialized? : {dist.is_initialized()}")

"""
This function collects local new training points and returns
global new training points.

Distributed communication in this function needs to be optimized.
"""
def get_global_new_points(models_frmwrk,new_train_x_lcl, old_train_x,
                          app_rank,app_world_size,max_new_pts=5):
    # Obtain the shape of input tensor
    inpt_shape = new_train_x_lcl.size()
    new_pts_lcl = new_train_x_lcl.size(dim=0)
    # Figure out how many global points exist
    # The optimal thing would be using asynchronous send and recv
    # for individual elements of this tensor
    # Using all_reduce for now
    lcl_reduce_tensor = torch.zeros((app_world_size,),dtype=torch.int)
    # Populate only your local value
    lcl_reduce_tensor[app_rank] = new_pts_lcl
    dist.all_reduce(lcl_reduce_tensor,op=dist.ReduceOp.SUM)
    # Check if global sum is equal to 0
    if (torch.sum(lcl_reduce_tensor).item() == 0):
        return torch.empty(0,*inpt_shape[1:])
    # It should get here only if globally new points exist
    if app_rank == 0:
        new_train_x_lcl_lst = []
        req_lst = []
        j = 0
        if new_pts_lcl > 0:
            new_train_x_lcl_lst.append(new_train_x_lcl)
            j += 1
        for i in range(1,app_world_size):
            if lcl_reduce_tensor[i] > 0:
                new_train_x_lcl_lst.append(
                        torch.zeros((lcl_reduce_tensor[i],*inpt_shape[1:])))
                req = dist.irecv(new_train_x_lcl_lst[j],src=i)
                req_lst.append(req)
                j += 1
        for req in req_lst:
            req.wait()
    else:
        if new_pts_lcl > 0:
            dist.send(new_train_x_lcl,dst=0)

    new_pts_glbl_tnsr = torch.zeros((1,),dtype=torch.int)

    if app_rank == 0:
        # Get the unique sampling points
        new_train_x_glbl_nonunq = torch.cat(new_train_x_lcl_lst,dim=0)
        new_train_x_glbl = torch.unique(new_train_x_glbl_nonunq,dim=0)
        # Look for points that are significantly far out of bounds
        eps = 1e-6
        low_new_points = new_train_x_glbl[
				torch.where(new_train_x_glbl < 
                                torch.min(old_train_x)-eps)]

        high_new_points = new_train_x_glbl[
				torch.where(new_train_x_glbl >
                                torch.max(old_train_x)+eps)]

        interior_new_points = new_train_x_glbl[torch.where(
                                (new_train_x_glbl >= torch.min(old_train_x)) &
                                (new_train_x_glbl <= torch.max(old_train_x)))]
        # Sort interior_new_points based on descending uncertainty
        if torch.numel(interior_new_points) > 1:
            interior_std = models_frmwrk.get_output_std(interior_new_points)
            interior_mean = models_frmwrk.final_inference(interior_new_points)
            _ , interior_order = torch.sort(interior_std/interior_mean,dim=0,
                                            descending=True)
            interior_new_points = interior_new_points[interior_order]
        # Ascending order for left bound points
        if torch.numel(low_new_points) > 1:
            low_new_points,_ = torch.sort(low_new_points,dim=0,
	    			descending=False)
        # Descending ordering for right bound points
        if torch.numel(high_new_points) > 1:
            high_new_points,_ = torch.sort(high_new_points,dim=0,
	    			descending=True)

        new_train_x_glbl_1 = torch.cat([low_new_points,high_new_points,
                                        interior_new_points],dim=0)

        if torch.numel(new_train_x_glbl_1) > 0:
            new_train_x_glbl = uniform_subsampling(new_train_x_glbl_1,
                                        max_new_pts=max_new_pts)
            new_pts_glbl_tnsr[0] = new_train_x_glbl.size(dim=0)
        else:
            new_train_x_glbl = torch.empty(0,*inpt_shape[1:])

        dist.broadcast(new_pts_glbl_tnsr,src=0)
        # Send only if non-zero elements exist
        if new_pts_glbl_tnsr[0] > 0:
            dist.broadcast(new_train_x_glbl,src=0)
    else:
        dist.broadcast(new_pts_glbl_tnsr,src=0)
        # Receive only if non-zero elements exist
        if new_pts_glbl_tnsr[0] > 0:
            new_train_x_glbl = torch.zeros((new_pts_glbl_tnsr[0],*inpt_shape[1:]))
            dist.broadcast(new_train_x_glbl,src=0)
        else:
            new_train_x_glbl = torch.empty(0,*inpt_shape[1:])

    return new_train_x_glbl
