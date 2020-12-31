import math
import torch
import torch.nn as nn
from .constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_func_KL(ldj):
    """
    Args:
        ldj log determinant jacobian (batch, 1)
    Returns:
        loss torch.tensor of size []
        
    q_0 is the flow "prior"
    
    F_1 = log q_0(z_0) - log |det J_g (z_0) | 
    
    """
    # Note that if flow_prior = latent space prior, then they cancel out each other
    
#     return - torch.log(torch.tensor([(4*np.pi)])) - torch.mean(ldj) + torch.log(torch.tensor([(4*np.pi)])) 

    return - torch.mean(ldj)

def log_surface_hypersphere(n_dim_sphere, r=1):
    """
    n_dim_sphere of the sphere embedded in the n+1 dimensional Euclidean space
    
    Calculate surface in log space for numerical stability. Work with log gamma function for numerical stability.
    """
    
    s1 = (n_dim_sphere + 1) / 2 * torch.log(torch.tensor(math.pi)) + torch.log(torch.tensor(2.))
    
    s2 = - torch.lgamma( torch.tensor(n_dim_sphere + 1, dtype=torch.float32) / 2)

    s3 = n_dim_sphere * torch.log(torch.tensor(r).float())

    return s1 + s2 + s3
    
    

class Loss_on_sphere():
    """
    Made loss a class because then log surface is calculated only once which might be expensive in high dimensions.
    """
    def __init__(self, n_dim_sphere, r=1):
        
        self.log_surface = log_surface_hypersphere(n_dim_sphere, r=r)
        
    def calc_loss(self, ldj):
        
        return self.log_surface - torch.mean(ldj)
    
    
def loss_func_recon(psi, x_mb):
    return nn.BCEWithLogitsLoss(reduction='none')(psi, x_mb.reshape(-1, 784)).sum(-1).mean()
        
        
