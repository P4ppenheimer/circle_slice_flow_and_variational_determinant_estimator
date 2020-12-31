from torch.nn import functional as F
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3



class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""
    pass

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left= -1.0,
    right=1.0,
    bottom= -1.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
    ):
    """
    Constrained rational quadratic splines in the sense that input must lie in the interval [left, right]. 
    Args:
        inputs: (B1, B2, .., N)
        widths: (B1, B2, .., N, K) K widths correspond to K + 1 knots
        heights: same as widhts
        derivatives: (B1, B2, .. ,N, K + 1) K + 1 derivatives. For each knot one derivative. 
    Returns
        Outputs: (B1, B2, .., N)
        
    
    """
    
    if torch.min(inputs) < left or torch.max(inputs) > right:
        print(f'min {torch.min(inputs)} and max {torch.max(inputs)}')
        raise InputOutsideDomain()

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)    
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
   
    cumwidths = torch.cumsum(widths, dim=-1) 
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)
    
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]
    
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]

    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        
        # root = (- b + torch.sqrt(discriminant)) / (2 * a)
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        """
        In the notation of neural spline flows paper, arXiv:1906.04032v2 we have

        inputs = x 
        input_cumwidths = x^(k)
        input_cumheights = y^(k)
        input_bin_widths = x^(k+1) - x^(k)
        theta = xi(x)
        theta_one_minus_theta = xi * (1 - xi)
        input_heights = y^(k+1) - y^(k)
        input_delta = s^(k) = y^(k+1) - y^(k) / x^(k+1) - x^(k)
        input_derivatives = delta^(k)
        input_derivatives_plus_one = delta^(k+1)
        """
            
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet

    
def circular_RQ_NSF(inputs,
                unnormalized_widths,
                unnormalized_heights,
                unnormalized_derivatives,
                inverse = False,
                left = 0,
                right = 2 * np.pi):
                
    """
    Circular splines have the equal first and last derivative, that is d_0 = d_K.
    Args:
        inputs: (B1, B2, ..)
        widths: (B1, B2, .. , K) K widths correspond to K + 1 knots
        heights: same as widhts
        derivatives: (B1, B2, .. , K) K derivatives because actually K + 1 derivatives for each knot but d_0 = d_K because circular RQ-NSF. Thus K + 1 - 1 = K.
    Returns:
        Output: (B1, B2, ..)
            
    """

    # add a 0 column in the last column
    
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(0, 1))
    
    # assign the first derivative value to the last column. d_K <- d_0 to obtain circular spline.

    unnormalized_derivatives[..., -1] = unnormalized_derivatives[..., 0]
     
    return rational_quadratic_spline(
            inputs = inputs,
            unnormalized_widths = unnormalized_widths,
            unnormalized_heights = unnormalized_heights,
            unnormalized_derivatives = unnormalized_derivatives,
            inverse = inverse,
            left = left,
            right = right,
            bottom = left,
            top = right)
    

def normalize_NSF_params(unnormalized_widths, 
                   unnormalized_heights, 
                   unnormalized_derivatives,
                   left=0.,
                   right=1., 
                   bottom=0., 
                   top=1.,
                   inverse = False,    
                   min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                   min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                   min_derivative=DEFAULT_MIN_DERIVATIVE,
                   ):
    num_bins = unnormalized_widths.shape[-1]
    
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]
    
    return cumwidths, cumheights, derivatives

