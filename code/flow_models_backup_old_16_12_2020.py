import torch
from .mlp_models import *
from .transformations import *
import math
import time
import numpy as np
from .splines import *
from .constants import *
from .moebius import *


def convert_none_to_zero(x):
    if x is None:
        return 0
    else:
        return int(x)
    
def convert_none_to_one(x):
    if x is None:
        return 1
    else:
        return int(x)    
    

### flow functions ###    

class Moebius_Flow(torch.nn.Module):
    
    def __init__(self, 
                 num_centers, 
                 rezero_flag,
                 num_dim_conditioner=None,
                 num_dim_out=None,
                 learnable_convex_weights=False,
                 num_hidden=HIDDEN_DIM_MOEBIUS_MLP,
                 add_bias_term=False):
        """
        Args: 
            num_centers: 
            num_dim_conditioner: 
            num_dim_out: N_0 = d/2
            learnable_convex_weights: 
        """
        
        super().__init__()
        
        
        self.num_centers = num_centers
        num_dim_out = convert_none_to_one(num_dim_out)
        self.d_halve = num_dim_out
        self.add_bias_term = add_bias_term
        
        # 'dim_coupling_out' dimension of variables that are left out in coupling layer
        
        if num_dim_conditioner is not None:            
            self.param_predictor = MLP(n_inputs=num_dim_conditioner, 
                                        n_hidden=num_hidden, 
                                        n_out= 2 * num_centers * num_dim_out + 1)
        else:
            self.params = nn.Parameter(torch.randn(2 * num_centers + 1), requires_grad=True)

        # we model the weights as weight = log rho
        self.unconstr_rho = nn.Parameter(torch.zeros(num_centers).cuda(), requires_grad=learnable_convex_weights)

        # NOTE: we model rho[i] = exp(unconstr_rho[i]) / sum exp [ unconstr_rho ] . This enforces rho being convex weights
        # thus log_rho[i] = unconstr_rho[i] - log sum exp [ unconstr_rho ] = unconstr_rho[i] - Z_rho        
        
        # reset to zero parameter for proper init of parameters
        if rezero_flag:
            self.eta = nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)  
        else:
            self.eta = nn.Parameter(torch.ones(1).cuda(), requires_grad=False)        
        
    def forward(self, 
                theta,
                r,
                x_conditioner = None,
                inverse=False):
        
        """
        d: dimension of the 'coupling in' set. Must be an even number. 
           Along this dimension the parameters of the transformation are repeated.
           
        N dimension of the conditioner. N = d in case of non S-VAE.
        
        Args:
            theta:         (B1, B2,.., BN, d/2)
            r:             (B1, B2,.., BN, d/2)         
            x_conditioner: (B1, B2,.., BN, N)

            
        Return:
            theta_out:   (B1, B2,.., BN, d/2)
            ldj:         (B1, B2,.., BN, d/2)
        """
        
        # if we do not ensure same shapes, broadcasting can lead to huge dimensions 
        assert theta.shape == r.shape, "Theta and r must have same shape"
        assert theta.shape[:-1] == x_conditioner.shape[:-1], "Must have same batch dimensions"
        
        batch_dims = theta.shape[:-1] # (B1, B2, .., BN)
        d_halve = theta.shape[-1]        
              
        if self.num_centers > 1 and inverse:
            raise NotImplementedError('Not implemented for more than one center')
    
        if x_conditioner is not None:
            
            # w (B1, B2, .., BN, 2N_C * d/2 + 1)
            params = self.param_predictor(x_conditioner)
            
            # init params with zero 
            params = params * self.eta
            
            # w (B1, B2, .., BN, 2N_C * d/2)
            w = params[..., :-1]
            
            # w (B1, B2, .., BN)
            bias = params[..., -1]            

            # (B1, B2, .., d/2, N_C, 2)
            w = w.view(*w.shape[:-1], self.d_halve, self.num_centers, 2)
            
            # old ; when the output of the neural net was only 2*N_C and not 2*N_C*d/2 as now
            # w = w.view(*w.shape[:-1], self.num_centers, 2)
            
        else:
            # init params with zero
            self.params = self.params * self.eta
            
            # (2 N_C)
            w = self.params[..., :-1]
            
            # shape ([])
            bias = self.params[..., -1]
            
            # (N_C, 2)
            w = self.params.view(self.num_centers,2)

            # (B1, B2, .., N_C, 2), (None,)*length acts as n times unsqueeze
            w = w[(None,)*len(batch_dims)].repeat(*batch_dims, 1, 1)
            
            # (B1, B2, .., BN)
            bias = bias[(None,)*len(batch_dims)].repeat(*batch_dims)
         
        # bring tensors to same shape for vectorized forward pass
        seq_ones = [1]*len(batch_dims)
        
        r = r.unsqueeze(-1).repeat(*seq_ones, 1, self.num_centers)         # (B1, B2, .., d/2, N_C)
        theta = theta.unsqueeze(-1).repeat(*seq_ones, 1, self.num_centers) # (B1, B2, .., d/2, N_C)
        
        # old ; not necessary anymore because incorporated in reshaping
#         w = w.unsqueeze(-3).repeat(*seq_ones, d_halve, 1, 1)               # (B1, B2, .., d/2, N_C, 2)
    
        bias = bias.unsqueeze(-1).unsqueeze(-1).repeat(*seq_ones, d_halve, self.num_centers)               # (B1, B2, .., d/2, N_C)
        
        # normalize (B1, B2, .., d/2, N_C, 2)         
        w = r.unsqueeze(-1) * 0.99 * w / (1 + torch.norm(w, dim = -1, keepdim=True))
        
        if inverse and self.add_bias_term:
            theta = (theta - bias) % (2 * np.pi)

        theta_out = f_moebius(theta, w, r, inverse=inverse) # (B1, B2,.., d/2, N_C)
        det_jac = jacobian_moebius(theta, w, r, inverse=inverse) # (B1, B2,.., d/2, N_C)
        
        if not inverse and self.add_bias_term:
            theta_out = (theta_out + bias) % (2 * np.pi)

        # convex combin. over number of centers (B1, .., d/2, NC) -> (B1, .., d/2)
        theta_out = torch.mean(theta_out, dim = -1) 
        det_jac   = torch.mean(det_jac, dim = -1) 
        
        return theta_out, torch.log(det_jac + EPS)
    
    def get_MLP_predictor(self):
        return self.center_predictor

    
class Circular_Spline_Flow(torch.nn.Module):
    
    """
    Circular spline flow transforms d/2 angles theta. Note that each pair is transformed with the another parameter set consisting of width, height and derivatives.
    """
    
    def __init__(self, 
                 num_bins, 
                 num_dim_conditioner, 
                 rezero_flag,
                 num_dim_out=None,
                 num_hidden=HIDDEN_DIM_SPLINE_MLP):

        super().__init__()

        # if circular we have K parameters for derivative because s_0 = s_K. So 3K in total.
        # + 1 for bias term
           
        self.num_bins = num_bins

        num_dim_out = convert_none_to_one(num_dim_out)
        self.d_halve = num_dim_out
        
        self.params_predictor = MLP(n_inputs=num_dim_conditioner, 
                                    n_hidden=num_hidden, 
                                    n_out= 3 * num_bins * num_dim_out)
        
        # reset to zero parameter for proper init of parameters
        if rezero_flag:
            self.eta = nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)    
        else:
            self.eta = nn.Parameter(torch.ones(1).cuda(), requires_grad=False)        
                                          
    def forward(self, 
                theta, 
                r=None, 
                x_conditioner=None,
                inverse=False):
        
        # NOTE: added r as argument so that Neural Spline Flow and Moebius have the same arguments.

        """
        d/2 is the dimension of the set of thetas that are transformed with the same parameter set. 
        
        Important for Coupling layer where we split x_1:D in two halves x_1:d and x_d+1:D. 
        
        Then we build pairs of two in the first set and obtain d/2 angles which are transformed at the same time. 
        
        For Cylindrical flow d/2 = 1. 
        
        Args:
            theta:          (B1, B2,.., BN, d/2)
            r:              (B1, B2,.., BN, d/2)            
            x_conditioner:  (B1, B2,.., BN, N)
            
        Return:
            theta_out:   (B1, B2,.., BN, d/2)
            ldj:         (B1, B2,.., BN, d/2)
        """
        
        assert theta.shape[:-1] == x_conditioner.shape[:-1], "Must have same batch dimensions"
        
        batch_dims = theta.shape[:-1] # (B1, B2, .., BN)
        d_halve = theta.shape[-1]  
        
        # (B1, .., BN, 3K * d/2) ; old without params for each d/2 (B1, .., BN, 3K)  
        params = self.params_predictor(x_conditioner)
        

        
        # reset params to zero for proper init 
        params = params * self.eta

        # (B1, .., BN, d/2, 3K) 
        params = params.view(*batch_dims, self.d_halve, 3 * self.num_bins)   
        
        # split in three equal parts. Each has
        # (B1, .., BN, d/2, K)
        width, height, deriv = params.split(self.num_bins, dim=-1)
        
        # init (unnormalized) derivative with ln(e-1) such that normalized deriv via softplus becomes 1
        deriv = deriv + torch.ones(deriv.shape).cuda() * torch.log(torch.exp(torch.tensor(1.)) - 1) 
#         deriv = deriv + torch.ones(deriv.shape) * torch.log(torch.exp(torch.tensor(1.)) - 1)         
        
        theta_out, logabsdet = circular_RQ_NSF(
                inputs = theta,
                unnormalized_widths = width,
                unnormalized_heights = height,
                unnormalized_derivatives = deriv,
                left = 0,
                right = 2 * np.pi,
                inverse = inverse)
        
        
        return theta_out, logabsdet

    
    # TODO: rename to get_conditioner
    
    def get_unnormalized_params(self, x, x_conditioner=None):
        """
        Args: 
            x (batch, D)
        Returns:
            params (batch, D, 3 * num_bins + 1) or (batch, D, 3 * num_bins) for interval or circle transformation
        """

        return self.params_predictor(x, x_conditioner)
   
    
    

class Interval_Spline_Flow(torch.nn.Module):
    
    """
    Neural Spline Flow implementation to fit a given distribution on the unit interval [0,1].
    
    Conditionally transforms a set (z1, z2, .. zD) where each z_i lives in [-1, 1] in an autoregressive manner.

    For each transformer f_{psi_i}(z_i) the function predicts a parameter set consisting of widths, heights and derivatives denoted by
    psi_i = g(z_<i). This means psi_i depends on all z_j up to (excluding) z_i. 

    Furthermore, it is possible to condition this transformation on an additional set 
    x_conditioner, that is psi_i = g(z_<i | x_conditioner) and out_i = f_i(z_i | x_conditioner). 

    This transformation is realized for a batch of data points.  
    """
    
    def __init__(self, 
                 num_bins, 
                 num_dim,
                 mask_alternate_flag,
                 rezero_flag,
                 num_dim_conditioner=None, 
                 num_hidden=HIDDEN_DIM_SPLINE_MLP,
                 mask_type = 'autoregressive'):
                


        super().__init__()
        
        assert mask_type in ('autoregressive', 'coupling'), 'Invalid mask_type. Choose between "autoregressive" and "coupling"'
        
        self.num_bins = num_bins
        self.num_dim = num_dim
    
        # we have 3K + 1 parameters because have K + 1 derivatives and K for widhts and heights
        
        self.num_bins_deriv = num_bins + 1
        self.left, self.right = -1, 1
        
        if mask_type == 'coupling':
            self.params_predictor = MLP_simple_coupling(num_inputs=self.num_dim, 
                                             num_hidden=num_hidden, 
                                             num_outputs_widhts_heights=self.num_bins * num_dim,
                                             num_outputs_derivatives=self.num_bins_deriv * num_dim,
                                             mask_alternate_flag=mask_alternate_flag,
                                             num_dim_conditioner=None)
            
        
        elif mask_type == 'autoregressive':
            self.params_predictor = MLP_masked(num_inputs=self.num_dim, 
                                               num_hidden=num_hidden, 
                                               num_outputs_widhts_heights=self.num_bins * num_dim,
                                               num_outputs_derivatives=self.num_bins_deriv * num_dim,
                                               mask_type = mask_type,
                                               num_dim_conditioner=num_dim_conditioner)
        
#         print('mask_type in interval spline', mask_type)
        
        self.mask_type = mask_type
        
        # reset to zero parameter for proper init of parameters
        if rezero_flag:
            self.eta = nn.Parameter(torch.zeros(1).cuda(), requires_grad=True)
        else:
            self.eta = nn.Parameter(torch.ones(1).cuda(), requires_grad=False)
        
                                          
    def forward(self, 
                x, 
                x_conditioner=None,
                inverse=False):
        
        # NOTE: added r as argument so that Neural Spline Flow and Moebius have the same arguments.

        """
        Autoregressive transformation of [-1,1]^D. Note that we do this in reversed order. 
        
        D is the dimension of the interval.
        N is dimension of the conditioner. For S-VAE setting 
        
        Args:
            x:             (B, D)
            x_conditioner: (B, N)
            
        Return:
            output:      (B, D)
            ldj:         (B, D)
        """
        
        batch_size = x.shape[0]
        
        if x.max() > 1 or x.min() < -1:
            
            print(' in interval spline flow')
            print('x.max()',x.max())
            print('x.min()',x.min())       

        if not inverse:  
            
            # params (B, D, 3K+1) 
            width, height, deriv = self.params_predictor(x=x, x_conditioner=x_conditioner)
            
            # TODO: do reshaping within maskedMLP    
            if self.mask_type == 'autoregressive':
                
                # NOTE: this way of reshaping is veeery! important for retaining the autoregressive structure of the masked MLP
                width = width.view(-1, self.num_bins, self.num_dim).permute(0,2,1) # (B, D, K) 
                height = height.view(-1, self.num_bins, self.num_dim).permute(0,2,1) # (B, D, K)
                deriv = deriv.view(-1, self.num_bins_deriv, self.num_dim).permute(0,2,1) # (B, D, K+1)
                    
            # init (unnormalized) width and length with 0 and and derivative with ln(e-1)
            # such that width and length are equidistant and derivatives init with 1
            
            width = width * self.eta
            height = height * self.eta
            deriv = deriv * self.eta + torch.ones(deriv.shape).cuda()*torch.log(torch.exp(torch.tensor(1.))-1)       
#             deriv_ = deriv * self.eta + torch.ones(deriv.shape) * torch.log(torch.exp(torch.tensor(1.)) - 1)              
            
            outputs, logabsdet = rational_quadratic_spline(
                inputs = x,
                left= self.left,
                right = self.right,
                bottom = self.left,
                top = self.right,
                unnormalized_widths = width,
                unnormalized_heights = height,
                unnormalized_derivatives = deriv)
            
        elif inverse and self.mask_type == 'autoregressive':
            """
            Follow inversion of masked autoregressive flows of Papamakarios 2019 https://arxiv.org/pdf/1912.02762.pdf section 3.1.2

            """
            logabsdet = torch.tensor([]).to(device)

            z_random = torch.rand(batch_size, self.num_dim).to(device) * (self.right - self.left) + self.left

            for idx in range(self.num_dim):
                
                width, height, deriv = self.params_predictor(z_random, x_conditioner = x_conditioner)

                # NOTE: this way of reshaping is veeery! important for retaining the autoregressive structure of the masked MLP
                width = width.view(-1, self.num_bins, self.num_dim).permute(0,2,1) # (B, D, K) 
                height = height.view(-1, self.num_bins, self.num_dim).permute(0,2,1) # (B, D, K)
                deriv = deriv.view(-1, self.num_bins_deriv, self.num_dim).permute(0,2,1) # (B, D, K+1)
                
                # init (unnormalized) width and length with 0 and and derivative with ln(e-1)
                # such that width and length are equidistant and derivatives init with 1
                width = width * self.eta
                height = height * self.eta
                deriv = deriv * self.eta + torch.ones(deriv.shape).cuda()*torch.log(torch.exp(torch.tensor(1.))-1)                   
                
                outputs_0, logabsdet_0 = rational_quadratic_spline(
                    inputs = x,
                    left= self.left,
                    right = self.right,
                    bottom = self.left,
                    top = self.right,
                    unnormalized_widths = width,
                    unnormalized_heights = height,
                    unnormalized_derivatives = deriv,
                    inverse = True)

                z_random[:, idx] = outputs_0[:, idx]

                logabsdet = torch.cat([logabsdet, logabsdet_0[:, idx].view(-1,1)], dim=1)
                
            outputs = z_random 
            
        elif inverse and self.mask_type == 'coupling':
            
            # params (B, D, 3K+1) 
            width, height, deriv = self.params_predictor(x=x, x_conditioner=x_conditioner)            
            
            # init (unnormalized) width and length with 0 and and derivative with ln(e-1)
            # such that width and length are equidistant and derivatives init with 1
            width = width * self.eta
            height = height * self.eta
            deriv = deriv * self.eta + torch.ones(deriv.shape).cuda()*torch.log(torch.exp(torch.tensor(1.))-1)
            
            outputs, logabsdet = rational_quadratic_spline(
                inputs = x,
                left= self.left,
                right = self.right,
                bottom = self.left,
                top = self.right,
                unnormalized_widths = width,
                unnormalized_heights = height,
                unnormalized_derivatives = deriv,
                inverse = True)            
            
        return outputs, logabsdet
    
    def get_unnormalized_params(self, x, x_conditioner=None):
        """
        Args: 
            x (batch, D)
        Returns:
            params (batch, D, 3 * num_bins + 1) or (batch, D, 3 * num_bins) for interval or circle transformation
        """

        return self.params_predictor(x, x_conditioner)


class Coupling(torch.nn.Module):

    def __init__(self, 
                num_dim_data, 
                flow_type,
                rezero_flag,
                num_centers=None, 
                num_bins= None,
                num_dim_conditioner=None,
                learnable_convex_weights=False,
                cap_householder_refl=False):
        """
        TODO: add description

        """

        super().__init__()

        num_dim_conditioner = convert_none_to_zero(num_dim_conditioner)
        
        self.num_dim_data = num_dim_data

        # dim coupling in is dimension of the vector that is taken 'in' in the coupling layer
        # dim coupling out is dimension of the vector that is taken 'out' in the coupling layer
        
        def round_up_to_even(number):
            return math.ceil(number / 2.) * 2
            
        # we take the closes rounded up even number to the half of the size of the data
        self.dim_coupling_in = round_up_to_even(num_dim_data / 2)
        self.dim_coupling_out = num_dim_data - self.dim_coupling_in
        
        
        self.cond_rotation = Rotation(num_dim_data=self.dim_coupling_in, 
                            num_dim_conditioner=self.dim_coupling_out + num_dim_conditioner,
                            cap_householder_refl=cap_householder_refl)
        
        if flow_type == 'moebius':
             
            self.circle_transf = Moebius_Flow(num_centers=num_centers, 
                                rezero_flag=rezero_flag,
                                num_dim_conditioner=self.dim_coupling_out + num_dim_conditioner,
                                num_dim_out = self.dim_coupling_in / 2,
                                num_hidden=HIDDEN_DIM_MOEBIUS_MLP) 
                                        
        elif flow_type == 'spline':  
            
            self.circle_transf = Circular_Spline_Flow(num_bins=num_bins, 
                                num_dim_conditioner=self.dim_coupling_out + num_dim_conditioner,
                                rezero_flag=rezero_flag,
                                num_dim_out = self.dim_coupling_in / 2,
                                num_hidden=HIDDEN_DIM_SPLINE_MLP)

            
        # maybe we do not need that because the global rotation acts as a scalar bias term on the angle
        # However, we first do global rotation and then Moebius transformation
        self.bias = nn.Parameter(torch.rand(1).cuda(), requires_grad=True)
        
    def forward(self, x, sldj, x_conditioner=None, inverse=False):
        """
        
        D dimension of the data, 
        N dimension of the conditioner 
        d dimenions of transformed set == self.dim_coupling_in. D/2
        
        Args:
            x:            (B1, B2, .., D) 
            sldj          (B1, B2, .., B_N)
            x_conditioner (B1, B2, .., N) 
        Returns:
            z             (B1, B2, .., D)
            sldj          (B1, B2, .., B_N)
            x_conditioner (B1, B2, .., N) identity mapping
        """
        
        # define globabl variables for timing
        global slice_time
        
        x1 = x[..., :self.dim_coupling_in] # x1 (B1, B2, .., d)
        x2 = x[..., self.dim_coupling_in:] # x2 (B1, B2, .., d)
        
        if x_conditioner is not None:
            x2_and_conditioner = torch.cat([x2, x_conditioner], dim = -1)  # (B1, B2, .., N + d)            
        else:
            x2_and_conditioner = x2


        
        if not inverse:
            x1, sldj, x2_and_conditioner = self.cond_rotation(x=x1, 
                                                              sldj=sldj, 
                                                              x_conditioner=x2_and_conditioner, 
                                                              inverse = inverse) 
        
        # create dimension for pairs of two for distinct transformation
        # (B1, B2, .., d) -> (B1, B2, .., d/2, 2)
        x1 = x1.view(*x1.shape[:-1], int(self.dim_coupling_in / 2), 2) 
        
        # (B1, B2, .., d / 2)
        r = torch.norm(x1, dim = -1)
        # (B1, B2, .., d/2)
        theta = T_2(x1)
                
        # Do Moebius transformation pair of dimension 2 vectorized
        start_time = time.time()
        theta_out, ldj  = self.circle_transf(theta=theta,
                                      r=r,
                                      x_conditioner=x2_and_conditioner, 
                                      inverse=inverse)
        
        slice_time = time.time() - start_time
        
        z1 = T_1(theta_out, r) 
        
#         print('z1 norm', torch.norm(z1, dim = -1))
        
#         print('norm equals r ', torch.norm(z1, dim = -1) == r)
        
        # theta_out: (B1, B2,.., BN, d/2)
        # z1:        (B1, B2,.., BN, d/2, 2)
        # ldj:       (B1, B2,.., BN, d/2)        
        
        
        # undo creation of pairs
        
        # (B1, B2, .., d/2, 2) -> (B1, B2, .., d)
        z1 = z1.view(*z1.shape[:-2], self.dim_coupling_in)
        
        # (B1, B2, ..,BN, d/2) -> (B1, B2, .., BN)
        ldj = torch.sum(ldj, dim=-1)
        
        sldj = sldj + ldj
        
        # if inverse we have to switch the order of rotation and circle slices
        if inverse:
            z1, sldj, x2_and_conditioner = self.cond_rotation(x=z1, 
                                                              sldj=sldj, 
                                                              x_conditioner=x2_and_conditioner, 
                                                              inverse = inverse)
        
        z = torch.cat([z1, x2], dim=-1) # (B1, B2, .., D)
        
        return z, sldj, x_conditioner
    
    
class Rotation(torch.nn.Module):
    
    def __init__(self, num_dim_data, 
                       num_dim_conditioner=None,
                       cap_householder_refl=False):

        super().__init__()

        self.num_dim_data = num_dim_data
        self.cap_householder_refl = cap_householder_refl
        
        if cap_householder_refl:
            self.num_reflections = min(64, self.num_dim_data) 
        else:
            self.num_reflections = self.num_dim_data
        
        if num_dim_conditioner is not None:
            
            self.householder_params = MLP(n_inputs=num_dim_conditioner,          
                                            n_hidden=HIDDEN_DIM_ROTATION_MLP, 
                                            n_out=self.num_dim_data * self.num_reflections)
        else:
            self.householder_params = nn.Parameter(torch.randn(num_dim_data, self.num_reflections).cuda(), requires_grad=True)

    def forward(self, x, sldj, x_conditioner=None, inverse=False):
        """
        D == num_dim_data
        N == number dimensions conditioner
        
        Args: 
            x:             (B1, B2,.., BN, D)
            sldj:          (B1, B2,.., BN)
            x_conditioner: (B1, B2,.., BN, N)
        Returns 
            out:           (B1, B2,.., BN, D)
            sldj:          (B1, B2,.., BN)
        """
        # define global variables for timing test
        global cond_rotation
        global glob_rotation
        
        batch_dims = x.shape[:-1]  
        
        start_time = time.time()    
        
        if x_conditioner is not None:
            params = self.householder_params(x_conditioner).view(*batch_dims, self.num_dim_data, self.num_reflections)
        else:
            params = self.householder_params.unsqueeze(0).repeat(*batch_dims, 1, 1) 
        
        out = self.perform_rotation(x=x, params=params, inverse=inverse)
        
#         if x_conditioner is not None:
#             print('conditional parameters', params.shape)
#         elif x_conditioner is None:
#             print('unonditional params', params.shape)
        
        if x_conditioner is None:
            glob_rotation = time.time() - start_time
        else:
            cond_rotation = time.time() - start_time

        return out, sldj, x_conditioner
    
    def perform_rotation(self, x, params, inverse):
        """
        Args:
            x:      (B1, B2,.., BN, d)
            params: (B1, B2,.., BN, d, d)

        Returns
            out:    (B1, B2,.., BN, d)

        """

        dim = x.shape[-1]

        if self.cap_householder_refl:
            max_iter = min(dim, 64) # hardcoded maximum of 64 householder reflections
        else:
            max_iter = dim

        for i in range(max_iter) if not inverse else reversed(range(max_iter)):

            u = params[...,i]

            unit_u = u / torch.norm(u, dim = -1, keepdim=True)

            # calculate unit_u^T * x 
            scalar = unit_u.unsqueeze(-2).matmul(x.unsqueeze(-1)).squeeze(-1) # (B1, B2, .., 1)

            x = x - 2 * unit_u * scalar 

        return x
    

    
class Coupling_Flow(nn.Module):
    
    def __init__(self, 
                num_flows, 
                flow_type,
                num_dim_data,
                rezero_flag=True,
                num_centers=None,
                num_bins=None, 
                num_dim_conditioner=None,
                cap_householder_refl=False):
        """
        # TODO: Have to add num_dim_data here such that its not hardcoded when initializing the coupling layer
        """

        super().__init__()
        
        assert num_dim_data >= 3, 'data dimension must be greater equal 3'
        assert flow_type in ('spline', 'moebius'), 'flow_type not implemented'

        if flow_type == 'moebius' and num_centers is None:
                raise Exception('Add nr of centers as argument when using Moebius')
        if flow_type == 'spline' and num_bins is None:
                raise Exception('Add nr of of bins when using circular Spline!')
        

        self.num_dim_conditioner = num_dim_conditioner
        
        self._num_flows = num_flows
        self._num_dim_data = num_dim_data
        self._flow_type = flow_type

        if num_dim_conditioner is not None:
            self.mlp_body = MLP_body(n_inputs=num_dim_conditioner, n_hidden=HIDDEN_DIM_BASE_MLP)
            hidden_dim_base_mlp = HIDDEN_DIM_BASE_MLP
        else:
            hidden_dim_base_mlp = None

        self.scale = nn.ModuleList([])
        
        for _ in range(num_flows):
            
            self.scale.append(Rotation(num_dim_data = num_dim_data,
                                       num_dim_conditioner = hidden_dim_base_mlp,
                                       cap_householder_refl=cap_householder_refl))

            self.scale.append(Coupling(num_dim_data=num_dim_data, 
                                       flow_type=flow_type,
                                       rezero_flag=rezero_flag,
                                       num_centers=num_centers, 
                                       num_bins=num_bins,
                                       num_dim_conditioner = hidden_dim_base_mlp,                                 
                                       learnable_convex_weights=False,
                                       cap_householder_refl=cap_householder_refl))
            
                               
    def forward(self, x, 
                x_conditioner=None, 
                inverse=False):
                                
        sldj = 0

        if self.num_dim_conditioner is not None and x_conditioner is None:
            raise Exception('Have to pass x_condtioner to forward()')

        if x_conditioner is not None:
            x_conditioner = self.mlp_body(x_conditioner)
            
        if not inverse:
            for idx, block in enumerate(self.scale):                
                    
                x, sldj, x_conditioner = block(x=x, sldj=sldj, x_conditioner=x_conditioner, inverse=inverse)    

        if inverse:
            for block in reversed(self.scale):
                x, sldj, x_conditioner = block(x=x, sldj=sldj, x_conditioner=x_conditioner, inverse=inverse)
                


        return x, sldj, x_conditioner
    
    @property
    def model_type(self):
        """to distinguish between model types in visualize densities in S-VAE"""
        return 'flow'
    
    @property
    def flow_type(self):
        return self._flow_type
    
    @property
    def num_dim_data(self):
        return self._num_dim_data
    
    @property
    def num_flows(self):
        return self._num_flows





    
    
### flow functions ###    

class _Cylindrical_Flow(torch.nn.Module):

    def __init__(self, 
            num_bins, 
            flow_type, 
            num_dim_data, 
            mask_type,
            mask_alternate_flag,
            rezero_flag,
            num_dim_conditioner = None,
            num_centers = None):
        """
        Args:
            dim_base_conditioner is the dimensionality of the obersvable data x in p(z|x) when replacing the variational distribution with a flow
        """

        super().__init__()
        assert flow_type in ('moebius','spline'), 'Choose between moebius and spline flow'
        
        if flow_type == 'moebius' and num_centers is None:
            raise Exception('Specify number of centers')
        
        self.num_dim_data = num_dim_data
        self.num_dim_conditioner = num_dim_conditioner
        self.mask_type = mask_type
        self.mask_alternate_flag = mask_alternate_flag
        
        self.interval_spline = Interval_Spline_Flow(num_bins=num_bins,
                                                  num_dim=num_dim_data - 2,
                                                  num_dim_conditioner=num_dim_conditioner,
                                                  mask_alternate_flag= mask_alternate_flag,
                                                  rezero_flag=rezero_flag,                                                    
                                                  num_hidden=HIDDEN_DIM_SPLINE_MLP,
                                                  mask_type=self.mask_type)
        
        if flow_type == 'spline':
            self.circle_transf = Circular_Spline_Flow(num_bins=num_bins,
                    num_dim_conditioner=num_dim_data - 2 + convert_none_to_zero(num_dim_conditioner),
                    rezero_flag=rezero_flag,                                                      
                    num_hidden=HIDDEN_DIM_SPLINE_MLP)

        elif flow_type == 'moebius':
            self.circle_transf = Moebius_Flow(num_centers=num_centers,
                  rezero_flag=rezero_flag,
                  learnable_convex_weights=False, 
                  num_dim_conditioner=num_dim_data - 2 + convert_none_to_zero(num_dim_conditioner),
                  num_hidden=HIDDEN_DIM_MOEBIUS_MLP)
        
    def forward(self, 
                x, 
                sldj, 
                x_conditioner = None, 
                inverse=False):
        """
        D dimension of the data
        N dimension of the conditioner
        
        Args:
            x:             (B, D)
            x_conditioner: (B, M)
        Returns
            z:      (B, D)
            log_dj: (B)
        """
        cylinder_trafo_ldj = torch.zeros(x.shape[0]).to(x.device)
        
        if self.num_dim_conditioner is not None and x_conditioner is None:
            raise Exception('Have to pass x_condtioner to forward()')

        batch_size = x.shape[0]
        
        sphere = x[:, :2]
        heights = x[:, 2:].view(-1, x.shape[1] - 2) # (batch, D - 1)
        
        
        # for AR transformation of the intervals in correct order
        
        z_heights, log_dj_interval = self.interval_spline(x = heights, 
                                                          x_conditioner = x_conditioner,
                                                          inverse = inverse)        
        
        cylinder_trafo_ldj += log_dj_interval.sum(dim=1)
        
        # create mask for theta transformation, such that theta is conditioned only on the correct part
        if self.mask_type == 'coupling':
            
            mask = torch.zeros(z_heights.shape[1]).to(device)
            # set every second entry to 1
            mask[::2] = 1
            
            # change mask for theta transformation at every flow
            if self.mask_alternate_flag:
                mask = 1 - mask

        if self.mask_type == 'coupling' and not inverse:
            _z_heights = z_heights * mask
            
        elif self.mask_type == 'coupling' and inverse:            
            _heights = heights * mask
            
        else:
            _heights = heights
            _z_heights = z_heights
        

        # TODO: think about if this should be switched
        if x_conditioner is not None and not inverse:
            conditioner = torch.cat([_z_heights, x_conditioner], dim=1)
            
        elif x_conditioner is not None and inverse: 
            conditioner = torch.cat([_heights, x_conditioner], dim=1)
            
        elif x_conditioner is None and not inverse:
            conditioner = _z_heights
            
        elif x_conditioner is None and inverse:
            conditioner = _heights

        # (B, 1)            
        theta = T_2(sphere).unsqueeze(-1) 
        
        # (B, 1)            
        r = torch.ones(batch_size, 1).to(x.device)
        
        
        z_theta, log_dj_circle = self.circle_transf(theta = theta, 
                                                    x_conditioner = conditioner, 
                                                    r = r,
                                                    inverse = inverse)
        
        
        cylinder_trafo_ldj += log_dj_circle.view(-1)
        
        sldj += cylinder_trafo_ldj

        # transform theta back to (x1, x2) ; squeeze because they had both (B, 1) shape
        z_sphere = T_1(z_theta.squeeze(-1), r=r.squeeze(-1))
              
        z_cylinder = torch.cat([z_sphere, z_heights], dim=1)
        
#         print('z_cylinder \n', z_cylinder)
#         print()        

        return z_cylinder, sldj, x_conditioner
    
    
    def get_NSF_transformation(self, x):
        return self.interval_spline(x)
    
    def get_unnormalized_params(self, x):
        return self.interval_spline.get_unnormalized_params(x)

    

### flow functions ###    

class Cylindrical_Flow(torch.nn.Module):
    """
    Neural Spline Flow implementation to fit a given distribution 
        
    """
    def __init__(self, 
                num_flows,
                num_bins, 
                flow_type, 
                num_dim_data,
                mask_type = 'autoregressive',
                rezero_flag=True, 
                num_dim_conditioner = None,
                num_centers = None):

        super().__init__()

        assert flow_type in ('spline', 'moebius'), 'flow_type not implemented'
        assert num_dim_data >= 3, 'data dimension must be greater equal 3'
        assert mask_type in ('autoregressive', 'coupling'), 'Invalid mask_type. Choose between "autoregressive" and "coupling"'
        
        if mask_type=='coupling' and num_dim_data == 3: 
                raise Exception('Coupling only works for S^3 and higher dims')

        self.num_dim_conditioner = num_dim_conditioner
        
        self._flow_type = flow_type
        self._num_dim_data = num_dim_data
        self._num_flows = num_flows
        
        if flow_type == 'moebius' and num_centers is None:
            raise Exception('Add nr of centers as argument!')
        
        if num_dim_conditioner is not None:
            self.mlp_body = MLP_body(n_inputs=num_dim_conditioner, n_hidden=HIDDEN_DIM_BASE_MLP)
            hidden_dim_base_mlp = HIDDEN_DIM_BASE_MLP
        else:
            hidden_dim_base_mlp = None
        
        self.scale = nn.ModuleList([])
        
        for flow_id in range(num_flows):
            
            self.scale.append(_Cylindrical_Flow(num_bins=num_bins, 
                                                flow_type=flow_type, 
                                                num_dim_data=num_dim_data, 
                                                mask_type=mask_type,
                                                mask_alternate_flag=bool(flow_id % 2),
                                                rezero_flag=rezero_flag,
                                                num_dim_conditioner=hidden_dim_base_mlp,
                                                num_centers=num_centers))
   
            
    def forward(self, x, x_conditioner=None, inverse=False):
        
        # define globabl variables for timing
        global cyl_trafo_time
        global fold_time
                               
        sldj = 0
        
        if x_conditioner is not None:
            x_conditioner = self.mlp_body(x_conditioner)
        
        start_time = time.time() 
        
        x_cylinder, ldj_s_to_c = T_s_to_c(x)
        
        unfolding_time = time.time() - start_time
        
#         sphere = x_cylinder[:, :2]
#         heights = x_cylinder[:, 2:]
            
        start_time = time.time()   
        
#         print('x_cylinder \n',x_cylinder)

        for block in self.scale if not inverse else reversed(self.scale):

            x_cylinder, sldj, x_conditioner = block(x=x_cylinder, 
                                                    sldj=sldj, 
                                                    x_conditioner=x_conditioner, 
                                                    inverse=inverse)

#         print('out x_cylinder \n' ,x_cylinder)
        
        cyl_trafo_time = time.time() - start_time
          
        start_time = time.time() 
        z_sphere, ldj_c_to_s = T_c_to_s(x_cylinder)
        fold_time = time.time() - start_time + unfolding_time
        
#         print()
#         print(f'ldj_s_to_c + ldj_c_to_s exp {(ldj_s_to_c + ldj_c_to_s).exp().mean()}')
#         print(f'ldj_s_to_c + ldj_c_to_s {(ldj_s_to_c + ldj_c_to_s).mean()}')    
#         print(f'ldj_s_to_c {(ldj_s_to_c).mean()}')   
#         print(f'ldj_c_to_s {(ldj_c_to_s).mean()}')          
#         print()
        
        sldj += ldj_s_to_c + ldj_c_to_s
        
#         print()
#         print(f'sldj {sldj.exp().mean()}')
        
        return z_sphere, sldj, x_conditioner

    
    def get_num_centers(self):
        return self.num_centers

    @property
    def model_type(self):
        """to distinguish between model types in visualize densities in S-VAE"""
        return 'flow'
    
    @property
    def flow_type(self):
        return self._flow_type
    
    @property
    def num_dim_data(self):
        return self._num_dim_data    
    
    @property
    def num_flows(self):
        return self._num_flows

    
    def return_NSF_transformation(self, x):
        """
        Args:
            x (batch, D)
        Returns
            out (batch, D)
            ldj (batch, D)
        """
        return self.scale[0].get_NSF_transformation(x=x)
    
    def get_unnormalized_params(self, x):
        """
        Args:
            x (batch, D)
        Returnsb
            params (batch, D, 3K+1)
        """
        return self.scale[0].get_unnormalized_params(x)

    


