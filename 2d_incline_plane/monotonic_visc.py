import torch
import numpy as np
import torch.nn.utils.parametrize as parametrize

# The current application only has a single monotonic input
class Monotonic_Convex_Viscosity_Network(torch.nn.Module):
    def __init__(self,input_size, output_size,
                    n_layers, layer_width,dropout_val=0.5):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.layer_width = layer_width

        self.module_list = torch.nn.ModuleList([
                            torch.nn.Linear(self.input_size,self.layer_width,bias=True)])
        self.module_list.append(torch.nn.ReLU())
        self.module_list.append(torch.nn.Dropout(p=dropout_val))
        for i in range(1, self.n_layers):
            self.module_list.append(torch.nn.Linear(
                self.layer_width,self.layer_width,bias=True))
            self.module_list.append(torch.nn.ReLU())
            self.module_list.append(torch.nn.Dropout(p=dropout_val))

        self.module_list.append(torch.nn.Linear(self.layer_width,self.output_size,bias=True))

        class AbsParameterization(torch.nn.Module):
            def forward(self, X):
                return torch.abs(X)
        for lyr in self.module_list:
            if isinstance(lyr,torch.nn.Linear):
                parametrize.register_parametrization(lyr, "weight", AbsParameterization())

    def forward(self,x):
        for lyr in self.module_list:
            x = lyr(x)
        return x

if __name__ == "__main__":
    visc_net = Monotonic_Convex_Viscosity_Network(input_size=1, output_size=1,
                                            n_layers=2,layer_width=50)

    aa = torch.linspace(-6,-3,10).unsqueeze(-1)

    print(visc_net(aa))
    print(visc_net(-1*aa))
