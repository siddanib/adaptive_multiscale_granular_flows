import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
# Local imports
from monotonic_visc import Monotonic_Convex_Viscosity_Network

class Model_Framework:
    def __init__(self, device_id, n_ensembles=10, lr=1e-3, new_pts_fctr=5.0,
		 n_layers=1,layer_width=10, dropout_val=0.5):

        self.models = [Monotonic_Convex_Viscosity_Network(
                            input_size=1, output_size=1,
                            n_layers=n_layers,layer_width=layer_width,
                            dropout_val=dropout_val).cuda()
			for _ in range(n_ensembles)]

        self.models_ddp = [DDP(mdl,device_ids=[device_id])
				 for mdl in self.models]

        # Stack parameters of ensembles
        self.params, self.buffers = torch.func.stack_module_state(
                                            self.models_ddp)

        self.n_ensembles = n_ensembles
        self.lr = lr
        self.optimizer = torch.optim.Adam(list(self.params.values()),
                                            lr=lr)
        self.loss = torch.nn.MSELoss()
        self.new_pts_fctr = new_pts_fctr
        # Confidence interval value
        self.confidence_val = 1.96 # 95% interval
        self.gamma = 3e-3

    def trnsfrm_inputs(self, x):
        return torch.log(x).unsqueeze(-1)

    def trnsfrm_outputs(self, x):
        return x.unsqueeze(-1)

    def inv_trnsfrm_outputs(self,x):
        return x.squeeze(-1)

    def train(self,input_data,output_data,output_std,max_iter=1e+6,n_freq=1e+3):
        sorted_input_data,_ = torch.sort(input_data)
        # Set in training mode
        [self.models_ddp[i].train(mode=True) for i in range(self.n_ensembles)]

        base_model = self.models_ddp[0]
        def ensemble_wrapper(params,buffers,data):
            return torch.func.functional_call(base_model,
                        (params,buffers),data)

        # Get the transformed inputs and outputs
        inpt = self.trnsfrm_inputs(input_data)
        for i in range(int(max_iter)):
            outpt = output_data.unsqueeze(0).expand(self.n_ensembles,-1)
            std_outpt = output_std.unsqueeze(0).expand(self.n_ensembles,-1)
            outpt = self.trnsfrm_outputs(
                    torch.clamp(outpt + torch.normal(mean=0,std=std_outpt),
                        min=1e-6))
            self.optimizer.zero_grad()
            mdl_outpt = (torch.vmap(ensemble_wrapper,in_dims=(0,0,None),randomness='different')
                            (self.params,self.buffers,inpt))
            loss_batch = self.loss(mdl_outpt,outpt)
            # For improving variance
            ####################################################################
            uniform_inputs_lst = []
            for j in range(10):
                uniform_ood = torch.clamp(torch.rand(torch.numel(input_data)-1),
                                          min=0.2,max=0.8)
                uniform_ood *= (sorted_input_data[1:] - sorted_input_data[:-1])
                uniform_ood += sorted_input_data[:-1]
                uniform_inputs_lst.append(uniform_ood)

            uniform_inputs = torch.cat(uniform_inputs_lst,dim=0)
            loss_batch -= self.gamma*self.loss(self.get_train_std(uniform_inputs)
                                    ,torch.zeros_like(uniform_inputs))
            ####################################################################
            if i % int(n_freq) == 0:
                print(f"loss_batch:{loss_batch.item()}")
            loss_batch.backward()
            self.optimizer.step()

    # mode = True means models are in training state
    def ensemble_inference(self, input_data, mode = False):
        # Set in evaluation mode
        [self.models_ddp[i].train(mode=mode) for i in range(self.n_ensembles)]
        # Get transformed inputs
        inpt = self.trnsfrm_inputs(input_data)

        if mode:
            base_model = self.models_ddp[0]
            def ensemble_wrapper(params,buffers,data):
                return torch.func.functional_call(base_model,
                        (params,buffers),data)

            mdl_outpt = (torch.vmap(ensemble_wrapper,in_dims=(0,0,None),randomness='different')
                            (self.params,self.buffers,inpt))
        else:
            with torch.no_grad():

                base_model = self.models_ddp[0]
                def ensemble_wrapper(params,buffers,data):
                    return torch.func.functional_call(base_model,
                            (params,buffers),data)

                mdl_outpt = (torch.vmap(ensemble_wrapper,in_dims=(0,0,None),randomness='different')
                                (self.params,self.buffers,inpt))
        return mdl_outpt

    def get_local_new_points(self,local_data, train_data,
            conc_local, min_conc):
        # local_data corresponds to test points,
        # train_data corresponds to train points
        # Figure out the location of each unique test point
        # among train points

        # Ignore strainrates where the second fluid is minimal
        conc_curated_local = local_data[torch.where(
                                conc_local >= min_conc)]
        # Ensure there are non-zero curated local points
        if conc_curated_local.size(dim=0) == 0:
            return torch.empty(0,)

        # Remove existing training points
        train_mask = ~torch.isin(conc_curated_local,train_data)
        conc_curated_local = conc_curated_local[train_mask]

        # Again ensure there are non-zero curated local points
        if conc_curated_local.size(dim=0) == 0:
            return torch.empty(0,)

        srtd_train = torch.unique(train_data,sorted=True,dim=0)
        srtd_local = torch.unique(conc_curated_local,sorted=True,dim=0)
        # Upper index location
        up_idx = torch.searchsorted(srtd_train,srtd_local,side='left')
        low_idx = torch.clamp(up_idx-1,min=0)
        up_idx = torch.clamp(up_idx,max=srtd_train.size(0)-1)
        # Get predictions
        mdl_outpt_up = self.ensemble_inference(srtd_train[up_idx])
        mdl_outpt_low = self.ensemble_inference(srtd_train[low_idx])
        mdl_outpt_local = self.ensemble_inference(srtd_local)
        # Remove normalization
        mdl_outpt_up = self.inv_trnsfrm_outputs(mdl_outpt_up)
        mdl_outpt_low = self.inv_trnsfrm_outputs(mdl_outpt_low)
        mdl_outpt_local = self.inv_trnsfrm_outputs(mdl_outpt_local)
        # Calculate std and mean of predictions
        std_up, mn_up = torch.std_mean(mdl_outpt_up,dim=0)
        std_low, mn_low = torch.std_mean(mdl_outpt_low,dim=0)
        std_local, mn_local = torch.std_mean(mdl_outpt_local,dim=0)
        # Get (std/mean) ratio for low and up
        std_mn_rt_up = std_up/mn_up
        std_mn_rt_low = std_low/mn_low
        # get min value between up and low tensors
        #std_mn_rt_min = torch.where(std_mn_rt_up < std_mn_rt_low,
        #                            std_mn_rt_up,std_mn_rt_low)
        # Get mean value between up up and low tensors
        #std_mn_rt_mean = 0.5*std_mn_rt_up + 0.5*std_mn_rt_low
        # Get (std/mean) ratio for local
        std_mn_rt_local = std_local/mn_local
        # How many points do not meet the criterion
        std_mn_rt_alpha = (self.trnsfrm_inputs(srtd_local).squeeze()
			   - self.trnsfrm_inputs(srtd_train[low_idx]).squeeze())

        std_mn_rt_alpha /= (self.trnsfrm_inputs(srtd_train[up_idx]).squeeze()
                            - self.trnsfrm_inputs(srtd_train[low_idx]).squeeze())
        std_mn_rt_wtmean = (1-std_mn_rt_alpha)*std_mn_rt_low + std_mn_rt_alpha*std_mn_rt_up
        crtrn_indices = torch.where(
                        std_mn_rt_local > std_mn_rt_wtmean*self.new_pts_fctr)
        # Get the uq based points
        uq_new_points = srtd_local[crtrn_indices]
        # Get points that are significantly outside on both sides
        low_new_points = srtd_local[torch.where(srtd_local < srtd_train[0])]
        up_new_points = srtd_local[torch.where(srtd_local > srtd_train[-1])]
        up_new_points,_ = torch.sort(up_new_points,dim=0,descending=True)
        # total new points
        total_new_points = torch.cat([low_new_points,uq_new_points,up_new_points],dim=0)
        return torch.unique(total_new_points,sorted=False,dim=0)

    # Get the stability condition
    def get_stability_condition(self, input_data):
        # Set in evaluation mode
        [self.models_ddp[i].train(mode=False) for i in range(self.n_ensembles)]
        # Get transformed inputs
        inpt = self.trnsfrm_inputs(input_data).requires_grad_(True)

        base_model = self.models_ddp[0]
        def ensemble_wrapper(params,buffers,data):
            return torch.func.functional_call(base_model,
                    (params,buffers),data)

        mdl_outpt = (torch.vmap(ensemble_wrapper,in_dims=(0,0,None),randomness='different')
                        (self.params,self.buffers,inpt))
        # Remember that you are taking derivative wrt log(I)
        first_dervtv = torch.autograd.grad(mdl_outpt,inpt,
                                grad_outputs=torch.ones_like(mdl_outpt))[0]
        # What is being returned is the sum across all networks
        # Get the mean
        first_dervtv /= self.n_ensembles
        # Obtain mean mu at each training point
        mu_mean = torch.mean(mdl_outpt,dim=0,keepdim=False)
        req_ratio = first_dervtv/mu_mean
        # Stability condition
        stab_C = 4.0*req_ratio**2 - 4*req_ratio + (mu_mean**2) *((1 - 0.5*req_ratio)**2)
        stab_C = stab_C.squeeze(-1)
        return stab_C

    # This function is used to check if there exists a point
    # among the training dataset that can act as low neutral Inertial number
    def check_inertial_num_stability(self, input_data, I_neutral):
        stab_C = self.get_stability_condition(input_data)
        # Get points where the required conditions are met
        critrn_I = input_data[torch.where(stab_C <= 0)]
        new_pts_stab = 1
        # There exists at least one point
        if (torch.numel(critrn_I) > 0):
            new_pts_stab = 0
            I_neutral,_ = torch.min(critrn_I,0,keepdim=True)

        return new_pts_stab, I_neutral

    ### This function leverages the regularized form for low Inertial Numbers
    def low_inertial_num_branch(self,input_data,I_neutral):
        mu_neutral = self.final_inference(I_neutral)
        alpha = 1.9 # Needs to be less than 2
        A_minus = I_neutral*torch.exp(alpha/(mu_neutral**2))
        mu_low = alpha/torch.log(A_minus/(input_data+1e-18))
        mu_low = torch.sqrt(mu_low)
        return mu_low

    def final_inference(self, input_data):
        mdl_outpt = self.ensemble_inference(input_data)
        # Need to remove normalization to
        # ensure monotonic nature before averaging
        mdl_outpt = self.inv_trnsfrm_outputs(mdl_outpt)
        return torch.mean(mdl_outpt,dim=0)

    def get_output_std(self, input_data):
        mdl_outpt = self.ensemble_inference(input_data, mode=False)
        # Need to remove normalization to
        # ensure monotonic nature before averaging
        mdl_outpt = self.inv_trnsfrm_outputs(mdl_outpt)
        return torch.std(mdl_outpt,dim=0)

    def get_train_std(self, input_data):
        mdl_outpt = self.ensemble_inference(input_data,mode=True)
        mdl_outpt = self.inv_trnsfrm_outputs(mdl_outpt)
        return torch.std(mdl_outpt,dim=0)

    def check_training_quality(self, train_inpt,
             train_outpt_mn, train_outpt_std, factor=5.0):
        mdl_outpt = self.ensemble_inference(train_inpt)
        # Remove normalization
        mdl_outpt = self.inv_trnsfrm_outputs(mdl_outpt)
        cond_tensr = torch.where(
		      torch.abs(mdl_outpt -
                          torch.unsqueeze(train_outpt_mn,0))
                      > factor*train_outpt_std)
        if torch.numel(mdl_outpt[cond_tensr]) == 0:
            return True
        else:
            return False

    # This refers to loading a model trained without DDP
    def load_pretrained_state(self,path="./"):
        params = torch.load(path+"models_params.tar")
        buffers = torch.load(path+"models_buffers.tar")

        for key_ddp, key_no_ddp in zip(self.params,params):
            self.params[key_ddp] = params[key_no_ddp]

        for key_ddp, key_no_ddp in zip(self.buffers,buffers):
            self.buffers[key_ddp] = buffers[key_no_ddp]

        chkpt_dct = torch.load(path+"models_optimizer.tar")
        self.optimizer.load_state_dict(chkpt_dct["optimizer"])
        dist.barrier()

    def load_pretrained_ddp_state(self,model_num,path="./",map_location='cpu'):
        self.params = torch.load(path+"models_params_"+str(model_num)+".tar",
                        map_location=map_location,weights_only=True)
        self.buffers = torch.load(path+"./models_buffers_"+str(model_num)+".tar",
                        map_location=map_location,weights_only=True)
        self.optimizer = torch.optim.Adam(list(self.params.values()),
                                            lr=self.lr)
        chkpt_dct = torch.load(path+"models_optimizer_"+str(model_num)+".tar",
                        map_location=map_location,weights_only=True)
        self.optimizer.load_state_dict(chkpt_dct["optimizer"])
        dist.barrier()

    def save_current_ddp_state(self,model_num):
        torch.save(self.params,"models_params_"+str(model_num)+".tar")
        torch.save(self.buffers,"models_buffers_"+str(model_num)+".tar")
        chkpt_dct = {}
        chkpt_dct["optimizer"] = self.optimizer.state_dict()
        torch.save(chkpt_dct,"models_optimizer_"+str(model_num)+".tar")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    sr_batch = torch.tensor([1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3])
    mu_batch = torch.tensor([1.233e-1,1.3097e-1,1.4756e-1,1.6598e-1,
                        2.0405e-1,2.6220e-1,3.7022e-1])

    mdl_frmwrk = Model_Framework()
    mdl_frmwrk.load_pretrained_state()
    mdl_frmwrk.train(sr_batch,mu_batch,max_iter=1)
    print(mdl_frmwrk.final_inference(sr_batch),mu_batch)
