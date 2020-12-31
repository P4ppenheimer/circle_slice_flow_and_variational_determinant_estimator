import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instead of writing my own clamp function with a straight trough gradient I think I could have also simply used 
# with torch.no_grad()
#    torch.clamp()
    
class Clamp(torch.autograd.Function):
    
    """Straight through Gradient version of clamp. Set NaN values to a and clamp to [a,b] afterwards."""
  
    @staticmethod
    def forward(ctx, input_,lower, upper):
        with torch.autograd.set_detect_anomaly(True):    
            # replace nan values with lower
            input_cloned = input_.clone()
            input_cloned[torch.isnan(input_)] = lower

            # clamp
            return input_cloned.clamp(min=lower, max=upper)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def T_s_to_c(x_sphere):
    """
    
    """
    
    # cumsum starting from x_n till x_3. So except x_1 and x_2
    sum_squares = torch.sqrt(1-torch.cumsum(x_sphere.flip(dims=[1]) ** 2, dim=1)[...,:-2])
    
    # for numerical reasons we set the minimum of sum_squares to 5*10^-4
    # otherwise we divide by 0 
    if torch.isnan(sum_squares).any():
        print('nan in sum squares')
    
    custom_clamp = Clamp.apply
    
    # clamp to [5e-4, 1]
    sum_squares = custom_clamp(sum_squares, 5e-4, 1)

    # first element: 1- x3^2 - .. - xn^2. Last element 1-xn
    sum_squares = sum_squares.flip(dims=[1])

    # duplicate first entry
    sum_squares = torch.cat([sum_squares[...,0].view(-1,1), sum_squares],dim=1)

    # add ones in the very end
    sum_squares = torch.cat([sum_squares, torch.ones(sum_squares.shape[0], 1).cuda()], dim=1)
#     sum_squares = torch.cat([sum_squares, torch.ones(sum_squares.shape[0], 1)], dim=1)    

    # do underscore for avoiding inplace operation
    x_sphere = x_sphere / sum_squares
    
    # clamp heights to [-1 + 2e-3, 1 - 2e-3] interval 
    # Otherwise, values can not be processes by interval spline
    x_sphere[..., 2:] = custom_clamp(x_sphere[..., 2:], -1 + 2e-3, 1 - 2e-3)     

    # ldj calculation
    n_dim_spheres = torch.arange(x_sphere.shape[1]).to(device).float()
#     n_dim_spheres = torch.arange(x_sphere.shape[1]).float()    

    ldjs = - (n_dim_spheres[2:] / 2 - 1) * torch.log(1-x_sphere[...,2:] ** 2)  

    ldj = torch.sum(ldjs, dim=1)
        
    return x_sphere, ldj 



def T_c_to_s(x_cylinder):
    
    """
    S^{1} x [-1,1]^{D-1} -> S^{D} 
    
    Args:
        x_cylinder torch.tensor S^{1} x [-1,1]^{D-1} (batch, n_dim_data)
    Returns:
        x_sphere S^{D} torch.tensor (batch, n_dim_data)
        sldj torch.tensor (batch)
    """    
    
    def T_c_to_s_one_step(x_sphere, x_height):
        
        """
        S^{D-1} x [-1,1] -> S^{D}
        """
        n_dim_sphere = x_sphere.shape[1] # equal to dim of the target sphere, that is D
        
        x_one_to_last = x_sphere * torch.sqrt(1 - x_height ** 2)
        x_last = x_height 
        
        ldj = (n_dim_sphere / 2 - 1) * torch.log(1 - x_height ** 2)

        x_sphere_new = torch.cat([x_one_to_last, x_last], dim=1)

        return x_sphere_new, ldj

    sldj = 0

    n_dim_data = x_cylinder.shape[1]

    x_sphere = x_cylinder[:,:2]
    x_heights = x_cylinder[:,2:]

    # till n_dim_data - 2 because we  S^1 x [-1,1]^D-1 = S^1 x [-1,1]^n_dim_data-2 -> S^D
    for idx in range(n_dim_data - 2):

        x_height = x_heights[:,idx].view(-1,1)

        x_sphere, ldj = T_c_to_s_one_step(x_sphere, x_height)

        sldj += ldj

    return x_sphere, sldj.view(-1)

    
    
    

########### moebius transformation functions ##### 

class set_to_zero_within_interval(torch.autograd.Function):
    """
    Differentiable way of rounding with given decimals. 
    
    Uses straigth-trough estimator to calculate a gradient. Gradient is set to 1.
    """
    
    @staticmethod
    def forward(ctx, input_, boundary):

        # set values within (- boundary, boundary) to zero
        input_[torch.abs(input_) <= boundary] = 0
        
        return input_

    @staticmethod
    def backward(ctx, grad_output):    
        '''Straight through gradient''' 
        return grad_output, None

    
    
def T_1(theta,r):
    """
    Args:
        theta: (B1, B2, .., B_N)
        r:     (B1, B2, .., B_N)
    Return
        x:     (B1, B2, .., B_N, 2)
    """
    
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
    
    
def T_2(x):
    """
    Transforms Euclidean to spherical coordinates.
    
    Args:
        x:     (B1, B2, .., B_N, 2)
    Returns:
        theta: (B1, B2, .., B_N)
    """
    assert x.shape[-1] == 2, "Input must be 2D"
    
    # 2.x e-7 für zeros input and 1.8 e-6 for 2pi input
    boundary = 2e-6        
    set_to_zero = set_to_zero_within_interval.apply
    
    theta = torch.atan2(x[...,1], x[...,0])   

    return set_to_zero(theta, boundary) % (2*math.pi) 