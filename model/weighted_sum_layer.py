import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Weighted_Sum_Layer(nn.Module):
    '''Either does weight times inputs
    or weight times inputs + bias
    Input to be provided as:
      - Weights
      - ndim biases (if applicable)
      - ndim items to sum
    Currently works for 3-dim input, summing over the 2nd axis'''
    def __init__(self, ndim=2, withbias=False):
        super(Weighted_Sum_Layer,self).__init__()
        self.ndim = ndim
        self.withbias = withbias
        
    def forward(self,X):
        # input #B x E x F
        # B is batch size (num events in a batch)
        # E is num of particles in event (num particles in an event)
        # F is num features produced by the algorithm, index 0 is the weights
        weights = X[:,0:1]  # B x E x 1
        
        if not self.withbias:
            tosum = X[:,1:]  # B x E x F-1
            #weighted is weights multiplied by tosum element wise
            weighted = weights * tosum  # broadcast to B x E x F-1
        else:
            tosum = X[:, self.ndim+1:]  # B x E x F-1
            biases = X[:, 1:self.ndim+1]
            weighted = weights * (biases + tosum)  # broadcast to B x E x F-1
        out = torch.sum(weighted,axis=0)
        return out # B x E, particles in the event reduced to a single value representing MET
    

