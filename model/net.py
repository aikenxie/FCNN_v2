"""Defines the neural network, loss function and metrics"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from model.weighted_sum_layer import Weighted_Sum_Layer

class FCNN_MET(nn.Module):
    def __init__ (self, continuous_dim=8, categorical_dim=3, output_dim=1, hidden_dim=32, withbias=False):
        super(FCNN_MET,self).__init__()

        #embeddings
        self.charge_embedding = nn.Embedding(3,hidden_dim//4) 
        
        self.pdgid_embedding = nn.Embedding(11,hidden_dim//4)
        
        self.pv_embedding = nn.Embedding(8, hidden_dim//4)

        #embeddings_initializer=initializers.RandomNormal(mean=0., stddev=0.4/emb_out_dim) 
        # the above initilisation for embedding layers from keras implementation was not implemented

        '''
        need to implement lecun_uniform initilisation for weights for linear layers

        Draws samples from a uniform distribution within [-limit, limit], where limit = sqrt(3 / fan_in) (fan_in is the number of input units in the weight tensor).
        '''
        '''
            need to implement the variance scaling initialisation for weights of output layer as below

            kernel_initializer=initializers.VarianceScaling(scale=0.02))(x)
        '''
        self.dense_stack = nn.Sequential(
            nn.Linear(32,64), #layer 1
            nn.Tanh(),
            nn.BatchNorm1d(64,momentum=0.05), #equivalent to keras momentum of 0.95
            
            nn.Linear(64,32), #layer2
            nn.Tanh(),
            nn.BatchNorm1d(32,momentum=0.05),

            nn.Linear(32,16), #layer 3
            nn.Tanh(),
            nn.BatchNorm1d(16,momentum=0.05),

            nn.Linear(16,3 if withbias else 1), #output layer, no activation function for output layer
        )
            
        self.weighted_sum_layer = Weighted_Sum_Layer(withbias=withbias)

        self.pdgmap = {-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10}
        

    def forward(self,X_cont,X_cat):
        #embedd the categorical features
        
        pdg_remapped = X_cat[:,0]
        for i, (k,v) in enumerate(self.pdgmap.items()):
            pdg_remapped = torch.where(pdg_remapped == k, torch.full_like(pdg_remapped, v), pdg_remapped)

        embedded_pdgId = self.pdgid_embedding(pdg_remapped)
        embedded_charge = self.charge_embedding(X_cat[:,1]+1)
        embedded_pv = self.pdgid_embedding(X_cat[:,2])

        #concatenate continous with categorical features
        X = torch.cat([X_cont,embedded_pdgId,embedded_charge,embedded_pv],dim=1)
        
        #Feed through the 3 dense layers
        X = self.dense_stack(X) 
        '''
        #Compile weights and momentum in xy
        pX = X_cont[:,0]
        pX = pX.reshape(len(pX),1)
        pY = X_cont[:,1]
        pY = pY.reshape(len(pY),1)
       
         = torch.cat([X,pX,pY],dim=1)
        
        #put through weighted sum layer to extract MET
        out = self.weighted_sum_layer(X)
        '''
        weights = torch.sigmoid(X)
        weights = torch.flatten(weights)
        return weights 

        

def getdot(vx, vy):
    return torch.einsum('bi,bi->b',vx,vy)
def getscale(vx):
    return torch.sqrt(getdot(vx,vx))
def scalermul(a,v):
    return torch.einsum('b,bi->bi',a,v)

def u_perp_par_loss(weights, prediction, truth, batch):
    qTx=truth[:,0]#*torch.cos(truth[:,1])
    qTy=truth[:,0]#*torch.sin(truth[:,1])
    # truth qT
    v_qT=torch.stack((qTx,qTy),dim=1)

    px=prediction[:,0]
    py=prediction[:,1]
    METx = -scatter_add(weights*px, batch)
    METy = -scatter_add(weights*py, batch)
    # predicted MET/qT
    vector = torch.stack((METx, METy),dim=1)

    response = getdot(vector,v_qT)/getdot(v_qT,v_qT)
    v_paral_predict = scalermul(response, v_qT)
    u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
    v_perp_predict = vector - v_paral_predict
    u_perp_predict = getscale(v_perp_predict)
    
    return 0.5*(u_paral_predict**2 + u_perp_predict**2).mean()
    
def resolution(weights, prediction, truth, batch):
    
    def getdot(vx, vy):
        return torch.einsum('bi,bi->b',vx,vy)
    def getscale(vx):
        return torch.sqrt(getdot(vx,vx))
    def scalermul(a,v):
        return torch.einsum('b,bi->bi',a,v)    

    qTx=truth[:,0]#*torch.cos(truth[:,1])
    qTy=truth[:,1]#*torch.sin(truth[:,1])
    # truth qT
    v_qT=torch.stack((qTx,qTy),dim=1)

    pfMETx=truth[:,2]#*torch.cos(truth[:,3])
    pfMETy=truth[:,3]#*torch.sin(truth[:,3])
    # PF MET
    v_pfMET=torch.stack((pfMETx, pfMETy),dim=1)

    puppiMETx=truth[:,4]#*torch.cos(truth[:,5])
    puppiMETy=truth[:,5]#*torch.sin(truth[:,5])
    # PF MET                                                                                                                                                            
    v_puppiMET=torch.stack((puppiMETx, puppiMETy),dim=1)

    has_deepmet = False
    if truth.size()[1] > 6:
        has_deepmet = True
        deepMETResponse_x=truth[:,6]#*torch.cos(truth[:,7])
        deepMETResponse_y=truth[:,7]#*torch.sin(truth[:,7])
        # DeepMET Response Tune
        v_deepMETResponse=torch.stack((deepMETResponse_x, deepMETResponse_y),dim=1)
    
        deepMETResolution_x=truth[:,8]#*torch.cos(truth[:,9])
        deepMETResolution_y=truth[:,9]#*torch.sin(truth[:,9])
        # DeepMET Resolution Tune
        v_deepMETResolution=torch.stack((deepMETResolution_x, deepMETResolution_y),dim=1)
    
    px_pred=prediction[:,0]
    py_pred=prediction[:,1]
    MET_x = scatter_add(weights*px_pred, batch)
    MET_y = scatter_add(weights*py_pred, batch)

    # predicted MET/qT
    v_MET=torch.stack((MET_x, MET_y),dim=1)
    
    def compute(vector):
        response = getdot(vector,v_qT)/getdot(v_qT,v_qT)
        v_paral_predict = scalermul(response, v_qT)
        u_paral_predict = getscale(v_paral_predict)-getscale(v_qT)
        v_perp_predict = vector - v_paral_predict
        u_perp_predict = getscale(v_perp_predict)
        return [u_perp_predict.cpu().detach().numpy(), u_paral_predict.cpu().detach().numpy(), response.cpu().detach().numpy()]

    resolutions= {
        'MET':      compute(-v_MET),
        'pfMET':    compute(v_pfMET),
        'puppiMET': compute(v_puppiMET)
    }
    if has_deepmet:
        resolutions.update({
            'deepMETResponse':   compute(v_deepMETResponse),
            'deepMETResolution': compute(v_deepMETResolution)
        })
    return resolutions, torch.sqrt(truth[:,0]**2+truth[:,1]**2).cpu().detach().numpy()

def response(weights,prediction,truth,batch):
    qTx = truth[:,0]
    qTy = truth[:,1]
    v_qT=torch.stack((qTx,qTy),dim=1)

    # PF MET
    pfMET_x=truth[:,2]
    pfMET_y=truth[:,3]
    v_pfMET=torch.stack((pfMET_x, pfMET_y),dim=1)



# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'resolution': resolution,
    'response': response
}