import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):

    def __init__(self, n_inputs, n_hidden, n_out):     
        super().__init__()      
        self.nn = nn.Sequential(nn.Linear(n_inputs, n_hidden),
                                nn.ReLU(),
                                nn.Linear(n_hidden, n_out))

    def forward(self, x):
        return self.nn(x)
    
class MLP_body(torch.nn.Module):

    def __init__(self, n_inputs, n_hidden):     
        super().__init__()      
        self.nn = nn.Sequential(nn.Linear(n_inputs, 2 * n_hidden),
                                nn.ReLU(),
                                nn.Linear(2 * n_hidden, n_hidden))
        
    def forward(self, x):
        return self.nn(x)
      

"""
From https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
"""


def get_coupling_output_mask(in_features, out_features, num_hidden):
    
    num_dim = in_features
    num_bins = int(out_features / in_features)
    
    mask = torch.zeros(num_bins*num_dim, num_hidden)

    block_ones = torch.ones(num_bins,num_hidden)

    for i in range(1, num_dim, 2):

        mask[num_bins * i : num_bins * (i+1)] = block_ones

    return mask.to(device) , mask[:,0].to(device)     


# TODO: fix for if in_features = 1
def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """

    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None):
        
        super(MaskedLinear, self).__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)
                
        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        
        # for autoregressive flows we don't have a bias mask
        # that's why it could be None
        output = F.linear(inputs, self.linear.weight * self.mask, self.linear.bias)
   
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
            
        return output


# MLP for AR
class MLP_masked(torch.nn.Module):

    def __init__(self, 
                 num_inputs, 
                 num_hidden, 
                 num_outputs_widhts_heights,
                 num_outputs_derivatives,
                 mask_type='autoregressive',
                 num_dim_conditioner=None):
                 
        super().__init__()
        
        nn.MaskedLinear = MaskedLinear
        
        num_bins = int(num_outputs_widhts_heights / num_inputs)
        num_bins_deriv = int(num_outputs_derivatives / num_inputs)
        
        assert mask_type in ('autoregressive', 'coupling'), 'Invalid mask_type. Choose between "autoregressive" and "coupling"'
        
        if num_inputs == 1:
            # NOTE: this ensures that nothing from the input arrives at the hidden layer. The hidden layer contains a bias term which itself is fully connected to the output. 
            # Further NOTE: If we additionally have an conditional_linear layer (w/o bias term), it is fully connected to the hidden layer resulting in a normal fully connected MLP.

            input_mask = torch.zeros(num_hidden,num_inputs).to(device)
            output_mask_widths_heights = torch.ones(num_outputs_widhts_heights,num_hidden).to(device)
            output_mask_derivatives = torch.ones(num_outputs_derivatives,num_hidden).to(device)
            
            
        # autoregressive only works if num_hidden >= num_inputs - 1 with this type of masking     
        elif mask_type == 'autoregressive':
            
            input_mask = get_mask(in_features=num_inputs, 
                                  out_features=num_hidden, 
                                  in_flow_features=num_inputs, 
                                  mask_type='input')

            output_mask_widths_heights = get_mask(in_features=num_hidden, 
                                                  out_features=num_outputs_widhts_heights, 
                                                  in_flow_features=num_inputs, 
                                                  mask_type='output')

            output_mask_derivatives = get_mask(in_features=num_hidden, 
                                               out_features=num_outputs_derivatives, 
                                               in_flow_features=num_inputs, 
                                               mask_type='output')
                  
            bias_mask_widths_heights = torch.ones(num_outputs_widhts_heights)
            bias_mask_derivatives = torch.ones(num_outputs_derivatives)            
            
            
        # for checking final mask of AR
        # debugging purpose
        self.prod = torch.matmul(output_mask_widths_heights, input_mask)
#         print('self.prod',self.prod)
        
        self.input = nn.MaskedLinear(in_features=num_inputs, 
                                     out_features=num_hidden, 
                                     mask=input_mask,
                                     cond_in_features=num_dim_conditioner)
        
        
        self.relu = nn.ReLU()
        
        self.out_widths = nn.MaskedLinear(in_features=num_hidden, 
                                          out_features=num_outputs_widhts_heights, 
                                          mask=output_mask_widths_heights)
        
        self.out_heights = nn.MaskedLinear(in_features=num_hidden, 
                                           out_features=num_outputs_widhts_heights, 
                                           mask=output_mask_widths_heights)
        
        self.out_derivatives = nn.MaskedLinear(in_features=num_hidden, 
                                               out_features=num_outputs_derivatives, 
                                               mask=output_mask_derivatives)

        
    def forward(self, x, x_conditioner=None):

        if x_conditioner is None:
            h = self.input(x)
        else:
            h = self.input(inputs=x, cond_inputs=x_conditioner)
            
        h = self.relu(h)
        
        widths = self.out_widths(h)
        heights = self.out_heights(h)
        derivatives = self.out_derivatives(h)

        return widths, heights, derivatives#, self.prod
    
    
### MLP for coupling

class Linear_conditional(nn.Module):
    
    def __init__(self,
                 in_features,
                 out_features,
                 cond_in_features=None):
        
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(cond_in_features, out_features, bias=False)
                

    def forward(self, inputs, cond_inputs=None):
        
        output = F.linear(inputs, self.linear.weight, self.linear.bias )
                        
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
            
        return output

class MLP_simple_coupling(torch.nn.Module):

    def __init__(self, 
                 num_inputs, 
                 num_hidden, 
                 num_outputs_widhts_heights,
                 num_outputs_derivatives,
                 mask_alternate_flag,
                 num_dim_conditioner=None):
                
                 
        super().__init__()
        
        nn.MaskedLinear = MaskedLinear
        
        num_bins = int(num_outputs_widhts_heights / num_inputs)
        num_bins_deriv = int(num_outputs_derivatives / num_inputs)
        
        # every second input as input 
        input_mask = torch.zeros(num_inputs).to(device)
        input_mask[::2] = 1
        input_mask = input_mask.view(1,-1)
        
        self.num_bins = int(num_outputs_widhts_heights/num_inputs)
        self.num_bins_deriv = int(num_outputs_derivatives/num_inputs)
        self.num_dim = num_inputs 
    
        if mask_alternate_flag:
            input_mask = 1 - input_mask

        self.register_buffer('input_mask', input_mask) 
            
        self.input = Linear_conditional(in_features=num_inputs, 
                                         out_features=num_hidden, 
                                         cond_in_features=num_dim_conditioner)
        
        self.relu = nn.ReLU()
        
        self.out_widths = Linear_conditional(in_features=num_hidden, 
                                          out_features=num_outputs_widhts_heights) 
        
        self.out_heights = Linear_conditional(in_features=num_hidden, 
                                           out_features=num_outputs_widhts_heights)

        self.out_derivatives = Linear_conditional(in_features=num_hidden, 
                                               out_features=num_outputs_derivatives)

    def forward(self, x, x_conditioner=None):

        x = x * self.input_mask
        
        if x_conditioner is None:
            h = self.input(x)
        else:
            h = self.input(inputs=x, cond_inputs=x_conditioner)
            
        h = self.relu(h)
        
        widths = self.out_widths(h).view(-1, self.num_dim, self.num_bins) 
        heights = self.out_heights(h).view(-1, self.num_dim, self.num_bins)     
        derivatives = self.out_derivatives(h).view(-1, self.num_dim, self.num_bins_deriv)
        
        output_mask = 1 - self.input_mask 
        
        # for broadcasting
        output_mask = output_mask.view(1, -1, 1)
        
        widths = output_mask * widths
        heights = output_mask * heights
        derivatives = output_mask * derivatives
        
        return widths, heights, derivatives 

