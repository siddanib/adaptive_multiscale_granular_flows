import torch
import math

def uniform_subsampling(x,max_new_pts=5):
    if max_new_pts > 1:
        if x.size(dim=0) <= max_new_pts:
            return x
        n = math.ceil(x.size(dim=0)/max_new_pts)
        if (x.size(dim=0)-1) % (n+1) == 0:
            return x[::n+1]
        else:
            return torch.cat([x[::n+1],torch.narrow(x,0,-1,1)],dim=0)
    else:
        return torch.narrow(x,0,0,1)
