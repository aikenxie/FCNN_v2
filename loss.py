import torch
from torch_scatter import scatter_add
import numpy as np

def getdot(vx, vy):
   return torch.einsum('bi,bi->b',vx,vy)
def getscale(vx):
    return torch.sqrt(getdot(vx,vx))
def scalermul(a,v):
    return torch.einsum('b,bi->bi',a,v)

def standard_loss(weights,prediction,truth,batch):
    
    px_pred = prediction[:,0]
    py_pred = prediction[:,1]
    px_truth = truth[:,0]
    py_truth = truth[:,1]
   
    px_pred = px_pred*weights
    py_pred = py_pred*weights
    

    MET_x = scatter_add(px_pred,batch)
    MET_y = scatter_add(py_pred,batch)

    loss = 0.5 * torch.mean((MET_x + px_truth)**2 + (MET_y + py_truth)**2)

    return loss

    


def response_correction_loss(weights,prediction,truth, batch, c = -10000):
    px_pred = prediction[:,0]
    py_pred = prediction[:,1]
    px_truth = truth[:,0]
    py_truth = truth[:,1]
   
    px_pred = px_pred*weights
    py_pred = py_pred*weights
    

    MET_x = scatter_add(px_pred,batch)
    MET_y = scatter_add(py_pred,batch)
    loss = 0.5 * torch.mean((MET_x + px_truth)**2 + (MET_y + py_truth)**2)

    v_MET = torch.stack((MET_x, MET_y),dim=1)
    v_qT = torch.stack((px_truth,py_truth),dim=1)

    response = getdot(v_MET,v_qT)/getdot(v_qT,v_qT)
    gt_1 = response > 1
    lt_1 = response < 1
    norm = torch.sum(torch.sqrt(px_truth**2+py_truth**2))
    response_term = c * (torch.sum(1-response[gt_1]) - torch.sum(response[lt_1]-1))
    response_term = response_term/norm 
    #print("loss:",loss)
    #print('response_term',response_term)
    loss = loss + response_term
    return loss
    

    
    