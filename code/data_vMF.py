### Functions for sampling from von Mises Fischer distribution

"""
In what follows, we reproduce the von Mises Fisher sampling method proposed by Ulrich (1984) and Davidson (2008)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from matplotlib import rcParams
import numpy as np
import math
from tqdm import tqdm
from collections import Counter

import torch.nn as nn
from random import randint
import time
from power_spherical import HypersphericalUniform, MarginalTDistribution, PowerSpherical
from .visualize_densities import T_spherical_to_cartesian, T_cartesian_to_spherical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### get samples from vMF density

def g_cdf_inverse(y,k): 
    """
    Args:
        u np.array (batch_size) uniforms samples 
        k np.array (batch_size) list of concetration parameters
    """
    Z = (np.exp(k) - np.exp(-k)) / k 
    return 1 / k * np.log(y * k * Z + np.exp(-k))

def get_vMF_samples(mu, k):
    """
    Generates (nr_samples) samples of von Mises Fisher samples. 
    Args:s
        mu np.array (nr_samples, 3)
        k np.array (nr_samples)
    Returns:
        u np.array (nr_samples, 3) samples where each u_i has distr. paramters (mu_i, k_i)
    """
    nr_samples = len(mu)
    
    uniform_samples = np.random.uniform(size = nr_samples)

    omega_samples = g_cdf_inverse(uniform_samples, k)

    theta = np.arcsin(omega_samples)
    phi = np.random.uniform( - np.pi, np.pi, size=nr_samples)

    # theta elevation from reference plane
    x = np.cos(phi) * np.sqrt(1 - omega_samples ** 2)
    y = np.sin(phi) * np.sqrt(1 - omega_samples ** 2)
    z = omega_samples 

    # # basis vectors (0,0,1) with shape (batch_size, 3)
    e3 = np.eye(N=1, M=3, k=2).repeat(nr_samples,axis=0)

    # np.array([x,y,z]) (3, batch)
    u = (e3 - mu) / np.linalg.norm(e3 - mu, axis = 1, keepdims=True)

    # U is of shape (batch, 3, 3)
    # U = Id - u * u^T
    U = np.expand_dims(np.eye(3),0).repeat(nr_samples, axis=0) - 2 * np.einsum('ij,ik->ijk', u, u)

    # (batch, 3) ; all samples are concentrated around (0, 0, 1)
    unlocated_samples = np.array([x,y,z]).T
    
    # calculates U * samples batch wise
    return np.einsum('ijk,ik->ij', U, unlocated_samples)


def mixture_entropy_von_mises_fisher(mu_list, k_list, nr_samples = int(1e6)):
    """
    Sample entropy of the form p(x) = sum rho_i p_i(x)
    with equal mixture components, i.e. rho_i = 1 / N
    Args: 
        mu (list)
        k (list) 
        nr_mixtures (int)
    """
    nr_mixtures = len(mu_list)
    
    assert len(mu_list) == len(k_list), 'len of lists must be equal'
    
    mixt_coeff = np.random.randint(low=0, high=nr_mixtures,size = nr_samples)

    # select randomly nr_samples many mu and k parameters
    selected_mu = np.take(mu_list, mixt_coeff, axis=0)
    selected_k = np.take(k_list, mixt_coeff, axis=0)
    
    # get samples for these parameters
    samples = get_vMF_samples(mu = selected_mu, k =selected_k)

    return np.mean(-np.log(mixture_vMF_density(samples, mu_list,k_list)))
  

class vonMisesFisherData(torch.utils.data.Dataset):
    """
    Dataset class of 3 dimensional (euclidean coordinates) von MisesFisher distribution on the 2 dimensional circle S^2 embedded in S^3.

    We use the following azimut, polar angle convention:

    - phi angle in x-y plane with range [-pi,pi]
    - theta elevation from reference x-y plane to z axis with range [-pi/2,pi/2]

    see https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:Kugelkoord-lokale-Basis-s.svg, except that theta angle is measured differently


        This results in 
        x = r * cos(phi) * cos(theta)
        y = r * sin(phi) * cos(theta)
        z = r * sin(theta) 

        NOTE: z = sin(theta) = omega 
        => cos(theta) = sqrt(1-sin^2(theta)) = sqrt(1-omega^2) 

        Thus with r = 1 
        x = cos(phi) * sqrt(1-omega^2) 
        y = sin(phi) * sqrt(1-omega^2) 
        z = omega 

    """

    def __init__(self,mu_list, k_list, nr_samples):
        """ 
        Args:
            mu (list of np.arrays) where each el is (3,) float:  location parameter 
            k (list of np.array) where each el is (1) float:  concentration parameter
            nr_samples (int): nr of samples in dataset
        Returns:
            sample (np.array) shape (3,)
        """
        
        assert len(mu_list) == len(k_list),"Nr of el in parameter lists must be equal"

        self._n_dim_data = mu_list[0].shape
        
        nr_mixtures = len(mu_list)
        
        self.mu_list = mu_list
        self.k_list = k_list      
        self.nr_samples = nr_samples
        
        self._entropy = mixture_entropy_von_mises_fisher(mu_list = mu_list, k_list = k_list)
        
        mixt_coeff = np.random.randint(low=0, high=nr_mixtures,size = nr_samples)
        
        # select nr_samples many location and concentration parameters
        selected_k = np.take(k_list, mixt_coeff, axis=0)
        selected_mu = np.take(mu_list, mixt_coeff, axis=0)
        
        self.data = get_vMF_samples(selected_mu, selected_k)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx] 

    def get_data(self):
        return self.data
    
    @property
    def cartesian_parameters(self):
        """Get params of vMF df as cartesian mu vector and scalar concetration param"""
        return self.mu_list, self.k_list
    
    @property
    def spherical_parameters(self):
        """Get spherical part of mu vector"""
        phi_mu_list = []
        theta_mu_list = []
        
        for mu in self.mu_list:
            r, phi, theta = T_cartesian_to_spherical(x=mu[0], y=mu[1], z=mu[2])
            phi_mu_list.append(phi)
            theta_mu_list.append(theta)
            
        return phi_mu_list, theta_mu_list
    
    @property 
    def train_samples(self):
        return self.nr_samples
    
    @property 
    def entropy(self):
        return self._entropy

    @property 
    def n_dim_data(self):
        return self._n_dim_data
    
    
### plot vMF density

def mixture_vMF_density(x, mu_list, k_list):
    """
    von Mises Fisher distribution in 3 dimensions on S^2.
    Args: 
        x (batch,3)
        mu_list (list) and each el is (3,1)
        k_list (list) and each el is (1)
    Returns 
        p (batch,1)
    """
    return_value = 0
    
    nr_mixtures = len(mu_list)
    
    for mu, k in zip(mu_list,k_list):
        
        Z = 2 * np.pi * ( np.exp(k) - np.exp(- k) ) / k
        
        return_value += 1 / Z * np.exp( k * np.dot(x, mu) )
        
    return return_value / nr_mixtures



def plot_vMF_density(mu_list, k_list,phi_mu_list,theta_mu_list):
    
    nr_grid_points = 100
    eps = 0.5e-1
    phi_linspace = np.linspace(-np.pi, np.pi, nr_grid_points)
    theta_linspace = np.linspace(-np.pi / 2 + eps, np.pi / 2 - eps, nr_grid_points)
    
    nr_grid_points = len(phi_linspace)

    dphi = phi_linspace[1] - phi_linspace[0]
    dtheta = theta_linspace[1] - theta_linspace[0]

    # NOTE: contourf takes array [X,Y] as argument where len(X) is number of columns and len(Y) number of rows. That is why probs[j][i]

    probs = np.empty([nr_grid_points, nr_grid_points])
    probs_with_cos = np.empty([nr_grid_points, nr_grid_points])

    for i, phi0 in tqdm(enumerate(phi_linspace)):

        for j, theta0 in enumerate(theta_linspace):

            x = T_spherical_to_cartesian(phi0, theta0)
            probs[j, i] = mixture_vMF_density(x, mu_list, k_list)

            # dx dy dz = r cos theta dr dtheta dphi in 'our' convention of theta
            probs_with_cos[j, i] = mixture_vMF_density(x, mu_list, k_list) * np.cos(theta0)

            
    # NOTE: this creates weird artefacts, at the bottom 
    fig = plt.figure(figsize=(20,8))

    ax = fig.add_subplot(121, projection="mollweide")
    ax.grid(linestyle='--')

    ax.contourf(phi_linspace, theta_linspace, probs, levels=50, cmap=plt.cm.jet)
    ax.scatter(phi_mu_list, theta_mu_list, s=20, color='black')

    ax = fig.add_subplot(122)
    ax.grid(linestyle='--')

    ax.contourf(phi_linspace, theta_linspace, probs, levels = 50, cmap = plt.cm.jet, extend='both')
    ax.scatter(phi_mu_list, theta_mu_list, s=20, color='black')
    
    plt.show()

    return probs, probs_with_cos, phi_linspace, theta_linspace
