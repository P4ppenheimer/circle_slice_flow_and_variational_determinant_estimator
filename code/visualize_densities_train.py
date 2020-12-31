from mpl_toolkits.basemap import Basemap
import torch
import numpy as np
import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import wandb
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""These functions are used to visualize the density in the 3d and S_VAE setting during traing"""

# coordinate transformation on $\mathbb{S}^2$
def T_spherical_to_cartesian(phi,theta):
    """
    3d transformation between spherical and euclidean coordinates. 
    NOTE: Convention see vMF dataset class

    Args: 
        phi (batch,1) - angle in x-y plane
        theta (batch,1) z axis evelation
    Returns: 
        x vector (shape,3)
    """
    
    # "usual" convention
    # x = np.sin(phi) * np.cos(theta)
    # y = np.sin(phi) * np.sin(theta)
    # z = np.cos(phi)

    # "my" convention where theta measures elevation from x-y plane
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    return np.array([x, y, z]).T

def T_cartesian_to_spherical(x, y, z):
    """
    3d transformation between spherical and euclidean coordinates theta is the angle. 
    NOTE: Convention see vMF dataset class
    """
    
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = np.arctan2(y, x) # range [-pi, pi]
    theta = np.arcsin(z / r) # range [-pi/2, pi/2]
    
    return np.array([r, phi, theta]).T




def evaluate_model_s_vae(VAE_model, nr_gridpoints, x_conditioner = None):
    
    """
    Args:
        VAE_model either modelS or modelN
    """

    _x_conditioner = x_conditioner.view(-1, 784).repeat(nr_gridpoints ** 2, 1)

    variational_posterior = VAE_model.get_variational_posterior(_x_conditioner)

    eps = 1e-2 # for numerical stability at unfolding
    phi_linspace = np.linspace(-np.pi, np.pi, nr_gridpoints)
    theta_linspace = np.linspace(-np.pi/2 + eps, np.pi/2 - eps, nr_gridpoints)

    x_vec = torch.zeros([nr_gridpoints, nr_gridpoints, 3]).to(device)
    cos_vec = torch.zeros([nr_gridpoints, nr_gridpoints]).to(device)

    for i, phi0 in enumerate(phi_linspace):

        for j, theta0 in enumerate(theta_linspace):

            # transform to cartesian coordiantes 
            x = torch.tensor(T_spherical_to_cartesian(phi0,theta0)).float().to(device)

            # x_vec shape (nr_gridpoints, nr_gridpoints, 3)
            x_vec[j,i] = x
            cos_vec[j,i] = np.cos(theta0)


    log_prob = variational_posterior.log_prob(x_vec.view(-1,3))

    dphi = phi_linspace[1] - phi_linspace[0]
    dtheta = theta_linspace[1] - theta_linspace[0]

    log_prob = log_prob.detach().cpu().numpy().reshape(nr_gridpoints,nr_gridpoints)
    cos_vec = cos_vec.detach().cpu().numpy()

    model_density = cos_vec * np.exp(log_prob)

    print(f'Sum to one check {np.sum(model_density) * dphi * dtheta}')
    

    nr_datapoints_MC = int(1e5)
    _x_conditioner = x_conditioner.view(-1, 784).repeat(nr_datapoints_MC, 1)
    variational_posterior = VAE_model.get_variational_posterior(_x_conditioner)
    
    print('model.num_dim_data',VAE_model.num_dim_data)
    x_MC_eval = torch.randn(nr_datapoints_MC, 3).to(device)
    x_MC_eval = x_MC_eval / torch.norm(x_MC_eval, dim=1, keepdim=True)

    log_prob = variational_posterior.log_prob(x_MC_eval)

    print(f'Sum to one check MC {(4*np.pi)*torch.mean(torch.exp(log_prob))} with {nr_datapoints_MC} datapoints')
    
    return model_density


def evaluate_model(model, nr_gridpoints, x_conditioner = None):
    
    # evaluate model via numerical approximation. Only works for num_dim_data == 3

    eps = 1e-2 # for numerical stability at unfolding
    phi_linspace = np.linspace(-np.pi, np.pi, nr_gridpoints)
    theta_linspace = np.linspace(-np.pi/2 + eps, np.pi/2 - eps, nr_gridpoints)

    x_vec = torch.zeros([nr_gridpoints, nr_gridpoints, 3]).to(device)
    cos_vec = torch.zeros([nr_gridpoints, nr_gridpoints]).to(device)
    
    if x_conditioner is not None:
        _x_conditioner = x_conditioner.view(-1, 784).repeat(nr_gridpoints ** 2, 1)
    else:
        _x_conditioner = None

    for i, phi0 in enumerate(phi_linspace):

        for j, theta0 in enumerate(theta_linspace):

            # transform to cartesian coordiantes 
            x = torch.tensor(T_spherical_to_cartesian(phi0,theta0)).float().to(device)

            # x_vec shape (nr_gridpoints, nr_gridpoints, 3)
            x_vec[j,i] = x
            cos_vec[j,i] = np.cos(theta0)

    # obtain log dj for mesh grid
    model.eval()
    
    _, ldj_vec, _ = model(x=x_vec.view(-1,3), x_conditioner = _x_conditioner)

    dphi = phi_linspace[1] - phi_linspace[0]
    dtheta = theta_linspace[1] - theta_linspace[0]
    
    ldj_vec = ldj_vec.detach().cpu().numpy().reshape(nr_gridpoints,nr_gridpoints)
    cos_vec = cos_vec.detach().cpu().numpy()
    
    # this model density is used for plotting
    model_density = 1 / (4 * np.pi) * np.exp(ldj_vec)
    
    # this model density is used for sum to one check
    model_density_with_cos = model_density * cos_vec

    print()
    print('Evaluate and plot model')
    print(f'Sum to one check {np.sum(model_density_with_cos) * dphi * dtheta}')
    
    # evaluate model via MC
    x_MC_eval = torch.randn(int(5e4), model.num_dim_data).to(device)
    x_MC_eval = x_MC_eval / torch.norm(x_MC_eval, dim=1, keepdim=True)
    
    if x_conditioner is not None:
        _x_conditioner = x_conditioner.view(-1, 784).repeat(int(5e4), 1)
    else:
        _x_conditioner = None

    _, ldj_MC_eval, _ = model(x=x_MC_eval, x_conditioner = _x_conditioner)

    print(f'Sum to one check MC {torch.mean(torch.exp(ldj_MC_eval))}')
    print(f'Avg ldj {torch.mean(ldj_MC_eval)}')
    print()
    model.train()

    return model_density


def plot_model_density(model_density, nr_gridpoints, epoch, batch_idx, phi_mu_list=None, theta_mu_list=None):
    
    """
    Args:
        phi_mu_list: list of phi coordinates of centers 
        theta_mu_list: list of theta coordinates of centers 
    """
    
    phi_linspace = np.linspace(-np.pi, np.pi, nr_gridpoints)
    theta_linspace = np.linspace(-np.pi/2, np.pi/2, nr_gridpoints)

    RAD = 180/np.pi
    
    # Model density plot
    
    X,Y = np.meshgrid(phi_linspace, theta_linspace)
    fig = plt.figure(figsize=(14,7))

    ax = fig.add_subplot(121)
    ax.grid(linestyle='--')
    
    m = Basemap(ax=ax, projection='moll',lon_0=0,lat_0=0,resolution='c')
#     m.scatter(phi_mu_list, theta_mu_list, s=20, color='black')
    m.contourf(X*RAD, Y*RAD, model_density, levels=50, cmap=plt.cm.jet,latlon=True)
    m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0],labelstyle='+/-',color = 'white') # draw parallels
    m.drawmeridians(np.arange(0.,420.,30.),labels=[0,0,0,0],color = 'white')
    
    if phi_mu_list is not None:
        x, y = m(phi_mu_list * RAD, theta_mu_list * RAD)
        m.scatter(x, y, s=20,color='black')

    ax = fig.add_subplot(122)
    ax.grid(linestyle='--')
    ax.contourf(phi_linspace/np.pi, theta_linspace/np.pi, model_density, levels = 50, cmap = plt.cm.jet, extend='both')
    
    if phi_mu_list is not None:
        ax.scatter(phi_mu_list/np.pi, theta_mu_list/np.pi, s=20, color='black') 
    
    plt.gca().xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().xaxis.set_major_locator(tck.MultipleLocator(base=0.25))
    plt.gca().yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().yaxis.set_major_locator(tck.MultipleLocator(base=0.2))
    plt.show()
    
    # plt for weights and biases in high resolution
    fig = plt.figure(figsize=(15,7),dpi=300)

    m = Basemap(projection='moll',lon_0=0,lat_0=0,resolution='c')
    m.contourf(X*RAD, Y*RAD, model_density, levels=50, cmap=plt.cm.jet,latlon=True)
    m.drawparallels(np.arange(-90.,120.,15.),labels=[1,0,0,0],labelstyle='+/-',color = 'white') # draw parallels
    m.drawmeridians(np.arange(0.,420.,30.),labels=[0,0,0,0],color = 'white')
    
    if phi_mu_list is not None:
        x, y = m(phi_mu_list * RAD, theta_mu_list * RAD)
        m.scatter(x, y, s=20,color='black')
        
    plt.close()
    
    # log the density fit figure to weights and biases. Epoch is counted as it starts from 1. 
    # We specify the step manually because the function is not called at every iteration step
    
    print(f'Epoch:{epoch} batch id: {batch_idx}')
    
    wandb.log({"Model density mollweide" : wandb.Image(fig, caption=f"Epoch:{epoch} batch id: {batch_idx}") }, 
              commit = False)
    
    fig = plt.figure(figsize=(15,7),dpi=300)
    plt.grid(linestyle='--')
    plt.contourf(phi_linspace/np.pi, theta_linspace/np.pi, model_density, levels = 50, cmap = plt.cm.jet, extend='both')
    
    if phi_mu_list is not None:
        plt.scatter(phi_mu_list/np.pi, theta_mu_list/np.pi, s=20, color='black')
    
    plt.gca().xaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().xaxis.set_major_locator(tck.MultipleLocator(base=0.25))
    plt.gca().yaxis.set_major_formatter(tck.FormatStrFormatter('%g $\pi$'))
    plt.gca().yaxis.set_major_locator(tck.MultipleLocator(base=0.2))
    plt.close()    
    
    wandb.log({"Model density unfolded" : wandb.Image(fig, caption=f"Epoch:{epoch} batch id: {batch_idx}") }, 
              commit = False)
    

def eval_and_plot_model(model, nr_gridpoints, epoch, batch_idx, phi_mu_list, theta_mu_list, x_conditioner = None):
    
    if model.model_type in ('vmf'):
        
        model_density = evaluate_model_s_vae(VAE_model=model, nr_gridpoints=nr_gridpoints, x_conditioner = x_conditioner)
        
    elif model.model_type == 'flow':
        
        model_density = evaluate_model(model, nr_gridpoints,x_conditioner = x_conditioner)    
        
    if model.num_dim_data == 3 and model.model_type not in ('normal'):
        
        plot_model_density(model_density, nr_gridpoints, epoch, batch_idx, phi_mu_list, theta_mu_list)
    
