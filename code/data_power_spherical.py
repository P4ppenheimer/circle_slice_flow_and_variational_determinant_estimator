import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from collections import Counter

from random import randint
import time
from power_spherical import PowerSpherical
from .visualize_densities_train import T_spherical_to_cartesian, T_cartesian_to_spherical

# need to add PROJ LIB for basemap
import os
os.environ["PROJ_LIB"] = os.path.join(os.environ["CONDA_PREFIX"], "share", "proj")
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as tck

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### create samples and density on sphere
def create_random_parameters(nr_mixtures,num_dim_data=3):

    loc = torch.randn(nr_mixtures, num_dim_data)
    # loc is uniformly distributed on the n sphere
    loc = loc / torch.norm(loc, dim=1, keepdim=True)

    scale = np.random.uniform(size=nr_mixtures) * 5 + 10
    
    if num_dim_data == 3:

        x, y, z = np.array(loc[:,0]), np.array(loc[:,1]), np.array(loc[:,2]) 
    
        spherical_coordin = T_cartesian_to_spherical(x,y,z)

        phi = spherical_coordin[:,1]
        theta = spherical_coordin[:,2]
        
        # TODO: decide here if we would like to have everything in pytorch or if numpy is also okay
        return np.array(loc), scale, phi, theta

    else: 
        return np.array(loc), scale, None, None

#### plot power spherical distribution ####  


def log_power_spherical_density(x, mu, k):
    """    
    x (batch, 3)
    mu (3,1)
    k (1)
    """
    dim = x.shape[1]
    
    alpha = (dim - 1) / 2 + k 
    beta = (dim - 1) / 2
    
    log_normalizer = -(
            (alpha + beta) * math.log(2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + beta)
            + beta * math.log(math.pi)
    )

    return log_normalizer + k * torch.log1p((mu * x).sum(-1))


def mixture_power_spherical_density(x, mu_list, k_list):
    """
    mixture of power spherical distributions in 3 dimensions on S^2.
    Args: 
        x (batch,3)
        mu_list(list) and each el is (3,1)
        k_list (list) and each el is (1)
    Returns 
        p (batch, 1) log probability
    """
    log_return_value = 0 
    log_ps = torch.tensor([])
    
    nr_mixtures = len(mu_list)
    
    for mu, k in zip(torch.tensor(mu_list), torch.tensor(k_list)):
        
#         log_pi = PowerSpherical(mu,k).log_prob(x)
        log_pi = log_power_spherical_density(x, mu, k)
        log_ps = torch.cat([log_ps,log_pi.float().view(-1,1)],dim=1)

    log_ps = log_ps - torch.log(torch.tensor(nr_mixtures).float())
    log_p = torch.logsumexp(log_ps,dim=1)  

        
    return torch.exp(log_p)

def plot_power_spherical_density(mu_list, k_list,phi_mu_list,theta_mu_list):
    
    nr_grid_points = 100

    phi_linspace = np.linspace(-np.pi, np.pi, nr_grid_points)
    theta_linspace = np.linspace(-np.pi / 2, np.pi / 2, nr_grid_points)
    
    nr_grid_points = len(phi_linspace)

    dphi = phi_linspace[1] - phi_linspace[0]
    dtheta = theta_linspace[1] - theta_linspace[0]

    # NOTE: contourf takes array [X,Y] as argument where len(X) is number of columns and len(Y) number of rows. That is why probs[j][i]

    probs = np.empty([nr_grid_points, nr_grid_points])
    probs_with_cos = np.empty([nr_grid_points, nr_grid_points])
    
    x = torch.zeros(nr_grid_points,nr_grid_points,3)
    theta_values = torch.zeros(nr_grid_points,nr_grid_points)

    for i, phi0 in tqdm(enumerate(phi_linspace)):

        for j, theta0 in enumerate(theta_linspace):

            x[j,i] = torch.tensor(T_spherical_to_cartesian(phi0, theta0)).float()
            theta_values[j,i] = torch.cos(torch.tensor(theta0)).float()            
    
    probs = mixture_power_spherical_density(x.view(-1,3), mu_list, k_list).reshape(nr_grid_points,nr_grid_points)
    probs_with_cos = probs * theta_values.reshape(nr_grid_points,nr_grid_points)
            
    X,Y = np.meshgrid(phi_linspace,theta_linspace)
    fig = plt.figure(figsize=(15,7),dpi=300)

    ax = fig.add_subplot(121)
    ax.grid(linestyle='--')
    RAD = 180/np.pi
    m = Basemap(ax=ax, projection='moll',lon_0=0,lat_0=0,resolution='c')
    cnt = m.contourf(X*RAD, Y*RAD, probs, levels=50, cmap=plt.cm.jet,latlon=True)
    
    # This is the fix for the white lines between contour levels when using high resol images like svg or pdf
    for c in cnt.collections:
        c.set_edgecolor("face")        
    
    m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0],labelstyle='+/-',color = 'white') # draw parallels
    m.drawmeridians(np.arange(0.,420.,30.),labels=[0,0,0,0],color = 'white')
    
    x,y=m(phi_mu_list*RAD,theta_mu_list*RAD)
    m.scatter(x, y,s=20,color='black')

    ax = fig.add_subplot(122)
    ax.grid(linestyle='--')
    cnt2 = ax.contourf(phi_linspace/np.pi, theta_linspace/np.pi, probs, levels = 50, cmap = plt.cm.jet, extend='both')
    
    # This is the fix for the white lines between contour levels when using high resol images like svg or pdf
    for c in cnt2.collections:
        c.set_edgecolor("face")        
    
    
    ax.scatter(phi_mu_list/np.pi, theta_mu_list/np.pi, s=20, color='black')
    
    plt.gca().xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().xaxis.set_major_locator(tck.MultipleLocator(base=0.25))
    plt.gca().yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().yaxis.set_major_locator(tck.MultipleLocator(base=0.2))
    
#     wandb.init(project=PROJECT_NAME,
#                name=f"target_density")
    
#     wandb.log({"Target_density" : wandb.Image(plt) })

    plt.savefig("density_plot/3D.svg", format='svg')

    plt.show()    
    
    return probs, probs_with_cos, phi_linspace, theta_linspace

    
    
#### Power Spherical data set class ####

class PowerSphericalData(torch.utils.data.Dataset):
    """
    TODO: does it make sense to define the functions inside the class?
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
        
        self.mu_list = torch.tensor(mu_list)
        self.k_list = torch.tensor(k_list)
        self.nr_samples = nr_samples
        self._n_dim_data = mu_list[0].shape
        
        start_time = time.time()     
        
        # if the dimension is higher than 100, we take 5 Mio samples to estimate the entropy ; else 1 Mio
        nr_samples = int(1e6) if mu_list.shape[1] < 100 else int(5e6)
        
        self._entropy = self.mixture_entropy_power_spherical(mu_list, k_list, nr_samples = nr_samples)
        print('Secs for entropy calc', time.time() - start_time)
        print(f'With entropy {self._entropy}')        
        
        
    def log_mixture_power_spherical_density(self, x, mu_list, k_list):
        """
        TODO: add docstring d
        """

        nr_mixtures = len(mu_list)
        return_value = torch.tensor([])
        neg_log_nr_mixtures = - torch.log(torch.tensor(nr_mixtures).float())

        for mu, k in zip(mu_list,k_list):

            dist = PowerSpherical(loc=mu, scale=k)
            return_value = torch.cat([return_value,  dist.log_prob(x).view(1,-1) ], dim = 0)

        # return_value shape (nr_mixtures, nr_samples)
        return torch.logsumexp(return_value + neg_log_nr_mixtures, dim = 0) # shape (nr_samples)  
            
    def get_power_spherical_samples(self, mu_list, k_list, nr_samples):
        """
        Args:
            mu_list np.array
            k_list np.array
            nr_samples scalar
        Returns:
            out torch.tensor 
        """
        nr_mixtures = len(mu_list)
        
        mixt_components = np.random.randint(low=0, high=nr_mixtures,size = nr_samples)

        # then count how often every mixture components occurs in sample
        # mix_comp_counter is dict with mix comps as keys and nr of samplings as values
        mix_comp_counter = Counter(mixt_components)

        data = torch.tensor([])

        # the sample for each mixture component, as many samples as they occured in the sampling of the components

        for mix_comp in mix_comp_counter:

            dist = PowerSpherical(loc=mu_list[mix_comp].clone().detach().float(), 
                                  scale=k_list[mix_comp].clone().detach().float())

            sample_per_comp = dist.sample((mix_comp_counter[mix_comp],))

            data = torch.cat([data, sample_per_comp], dim=0)

        # shuffle tensor
        return data[torch.randperm(nr_samples),:]
    
    def mixture_entropy_power_spherical(self, mu_list, k_list, nr_samples = int(1e5)):

        """
        Args: 
            mu_list torch.tensor (nr_mixtures, n_dim_data)
            k_list torch.tensor (nr_mixtures, n_dim_data)
        Returns:
            out torch.tensor scalar
        """

        mu_list = torch.tensor(mu_list)
        k_list = torch.tensor(k_list)
        nr_mixtures = len(mu_list)

        assert len(mu_list) == len(k_list), 'len of lists must be equal'
        
        # fixed random seeds for consistent entropy calculation
        np.random.seed(42) # for sampling of mixt components
        torch.manual_seed(42) # for sampling from PowerSpherical Distr.

        samples = self.get_power_spherical_samples(mu_list, k_list, nr_samples)

        return torch.mean(- self.log_mixture_power_spherical_density(samples, mu_list, k_list) )

            
    def __len__(self):
        return self.nr_samples

    def __getitem__(self, idx):
        #return value is of shape (D = dimension)
        return self.get_power_spherical_samples(self.mu_list, self.k_list, 1).squeeze()

    def get_data(self):
        return self.get_power_spherical_samples(self.mu_list, self.k_list, self.nr_samples)
    
    def get_test_set(self, nr_samples):
        return self.get_power_spherical_samples(self.mu_list, self.k_list, nr_samples)
    
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
            
        return np.array(phi_mu_list), np.array(theta_mu_list)
    
    @property 
    def train_samples(self):
        return self.nr_samples

    @property 
    def n_dim_data(self):
        return self._n_dim_data
    
    @property 
    def entropy(self):
        return self._entropy




