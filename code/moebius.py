import torch
from .transformations import *
import math
import time
import numpy as np


### moebius transformation functions    
    
def get_rotation_back_angle(w, r):
    """
    Calculates the angle between the point (1,0) and h(1,0)
    
    Args:
        w:   (B1, B2, .., B_N, 2)
        r:   (B1, B2, .., B_N)
        
    Returns:
        phi: (B1, B2, .., B_N)
    """
    # now let us compute the image of the vector (r,0), that is h(r,0)

    zeros = torch.zeros_like(r)

    origin_circle = torch.stack([r, zeros], dim=-1) # create (B1, B2, .., B_N, r, 0)

    h_orig_circle = h(origin_circle, w, r)

    return T_2(h_orig_circle)    
    
    
    
def h(x, w, r):
    """
    Fully vectorized basic Moebius transformation in terms of Cartesian coordinates.
    Args:
        x: (B1, B2, .., B_N, 2)
        w: (B1, B2, .., B_N, 2)f
        r: (B1, B2, .., B_N)
    Return:
        h: (B1, B2, .., B_N, 2)
    """
    r = r.unsqueeze(-1)
    
    return (r ** 2 - torch.norm(w, dim = -1, keepdim=True) ** 2) / (torch.norm(x - w, dim = -1, keepdim = True) ** 2) * (x - w) - w    



def f_moebius(theta, w, r, inverse=False):     
    """
    Function that takes theta as input, does rotation back and transforms Moebius transformation.
    
    Ensures fixpoint property f(0) = 0 and f(2pi) = 0. 
    
    Note that the inverse of the basic Moebius transformation h_w is h_w^{-1} = h_{-w}
    Args:
        
        theta:  (B1, B2, .., B_N)
        w:      (B1, B2, .., B_N, 2)
        r:      (B1, B2, .., B_N)
        inverse: boolean
        
    Returns:
        theta:  (B1, B2, .., B_N)
    """
    
    phi = get_rotation_back_angle(w, r)
    
    if not inverse:
        x = T_1(theta, r)
        z = h(x, w, r)
        theta = T_2(z)
        theta = (theta - phi) % (2 * np.pi)
        
    elif inverse:
        
        theta = (theta + phi) % (2 * np.pi) # undo the rotation back
        x = T_1(theta,r) # transform to Cartesian
        z = h(x, -w, r) # inverse Moebius transform with -w 
        theta = T_2(z) # transform back to theta
        
    return theta


#### jacobian transformation functions

def pder_T_1(theta,r):
    """
    Args:
        theta: (B1, B2, .., B_N)
        r:     (B1, B2, .., B_N)
    Return:
        dT1_dtheta: (B1, B2, .., B_N, 2, 1)
        
    """
    return torch.stack([- r * torch.sin(theta), r * torch.cos(theta)], dim = -1).unsqueeze(-1)


def pder_T_2(z):
    """
    Computes dT2_dh(x,y) = 1 / ( x^2 + y^2 ) * (- y, x)
    
    Args:
        z:      (B1, B2, .., B_N, 2)
    Return:  
        dT2_dh: (B1, B2, .., B_N, 1, 2)
            
    """
    assert z.shape[-1] == 2, "Input must be 2D"
    
    out = 1 / (torch.norm(z, dim=-1,keepdim=True)**2) * torch.stack([-z[..., 1], z[..., 0]], dim=-1)
    
    return out.unsqueeze(-2)


def pder_h(z,w,r):
    
    """
    Computes dh_dz = - 2 (r^2 - norm(w)^2)/norm(z-w)^4 (z-w)(z-w)^T + (r^2 - norm(w)^2)/norm(z-w)^2 * Identity
    
    Args:
        z:     (B1, B2, .., B_N, 2)
        w:     (B1, B2, .., B_N, 2)
        r:     (B1, B2, .., B_N)
    Return 
        dh_dz: (B1, B2, .., B_N, 2, 2)
    """

    # (B1, B2, .., B_N, 2, 2)
    z_w_matrix = torch.matmul((z-w).unsqueeze(-1), (z-w).unsqueeze(-2))

    # (B1, B2, .., B_N)
    factor_two = (r ** 2 - torch.norm(w, dim=-1) ** 2) / torch.norm(z-w, dim=-1) ** 2
    factor_one = - 2 * factor_two / torch.norm(z-w, dim=-1) ** 2
    
    # (B1, B2, .., B_N, 1, 1)
    factor_one = factor_one.unsqueeze(-1).unsqueeze(-1)
    factor_two = factor_two.unsqueeze(-1).unsqueeze(-1)
    
    # list [B1, B2, .., B_N]
    batch_dims = z.shape[:-1]

    # (2, 2)
    I = torch.eye(2).to(z.device) # send I to same device as z
    # (B1, B2, .., B_N, 2, 2)
    I = I[(None,)*len(batch_dims)].repeat(*batch_dims, 1, 1)
    
    return factor_one * z_w_matrix + factor_two * I

    

def jacobian_moebius(theta, w, r, inverse=False):
    """
    Compute Jacobian of Moebius transformation dT2/dh * dh/dT1 * dT1/dtheta 
        
    d/2 dimension of the variables that are 'coupling in'    
    N_C is number of centers
    
    Args:
        theta:      (B1, B2, .., d/2, N_C)
        w:          (B1, B2, .., d/2, N_C, 2)
        r:          (B1, B2, .., d/2, N_C)
    Returns:
        df_dtheta:  (B1, B2, .., d/2, N_C)
    """
    
    if inverse:
        # undo the phase translation, which ensures fixpoint property
        phi = get_rotation_back_angle(w, r)
        theta = (theta + phi) % (2 * np.pi)
    
    z = T_1(theta, r)

    # (B1, B2, .., 1, 2)
    dT2_dh = pder_T_2(h(z, w = w if not inverse else - w, r=r))

    # (B1, B2, .., 2, 2)
    dh_dT1 = pder_h(z, w = w if not inverse else - w, r=r) 

    # (B1, B2, .., 2, 1)
    dT1_dtheta = pder_T_1(theta,r)

    # (B1, B2, .., BN)
    return dT2_dh.matmul(dh_dT1).matmul(dT1_dtheta).squeeze(-1).squeeze(-1)
    
