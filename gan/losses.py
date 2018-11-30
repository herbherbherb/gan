import torch
import math
# from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.autograd import Variable

def bce_loss(input, target):
    neg_abs = - input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    dtype = torch.cuda.FloatTensor
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N = logits_real.size()
    loss = (bce_loss(logits_real, Variable(torch.ones(N)).type(dtype)) + \
        bce_loss(logits_fake, Variable(torch.zeros(N)).type(dtype)))
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    dtype = torch.cuda.FloatTensor
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N= logits_fake.size()
    loss = bce_loss(logits_fake, Variable(torch.ones(N)).type(dtype))
    ##########       END      ##########
    
    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    dtype = torch.cuda.FloatTensor
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N = scores_real.size()
    scores_real_loss = torch.mean(torch.pow(scores_real-Variable(torch.ones(N)).type(dtype), 2))/2
    scores_fake_loss = torch.mean(torch.pow(scores_fake, 2))/2
    loss = scores_real_loss + scores_fake_loss    
    ##########       END      ##########
    
    return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    dtype = torch.cuda.FloatTensor
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    N = scores_fake.size()
    loss = torch.mean(torch.pow(scores_fake - Variable(torch.ones(N)).type(dtype), 2))/2
    ##########       END      ##########
    
    return loss
