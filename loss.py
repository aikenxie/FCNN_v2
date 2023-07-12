import torch
from torch_scatter import scatter_add
import numpy as np

def custom_loss(weights,prediction,truth,batch):
    
    px_pred = prediction[:,0]
    py_pred = prediction[:,1]
    px_truth = truth[:,0]
    py_truth = truth[:,1]
   
    px_pred = px_pred*weights
    py_pred = py_pred*weights
    

    MET_x = scatter_add(px_pred,batch)
    MET_y = scatter_add(py_pred,batch)

    loss = 0.5 * torch.mean((MET_x - px_truth)**2 + (MET_y - py_truth)**2)

    return loss

    

'''
def custom_loss(y_true, y_pred):
    
    cutmoized loss function to improve the recoil response,
    by balancing the response above one and below one
    
   
    
    px_truth = torch.flatten(y_true[0])
    py_truth = torch.flatten(y_true[1])
    px_pred = torch.flatten(y_pred[0])
    py_pred = torch.flatten(y_pred[1])

    pt_truth = torch.sqrt(px_truth*px_truth + py_truth*py_truth)

    px_truth1 = px_truth / pt_truth
    py_truth1 = py_truth / pt_truth

    # using absolute response
    # upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred)/pt_truth
    upar_pred = (px_truth1 * px_pred + py_truth1 * py_pred) - pt_truth
    pt_cut = pt_truth > 0./50.

    
    from FCNN_v1 code, no direct implementationof boolean_mask in pytorch
    but i think the dimensions work out where we can use
    torch.mask_select
    upar_pred = tf.boolean_mask(upar_pred, pt_cut)
    pt_truth_filtered = tf.boolean_mask(pt_truth, pt_cut)
    

    upar_pred = torch.masked_select(upar_pred,pt_cut)
    pt_truth_filtered = torch.masked_select(pt_truth,pt_cut)

    filter_bin0 = pt_truth_filtered < 5./50.
    filter_bin1 = torch.logical_and(pt_truth_filtered > 5./50., pt_truth_filtered < 10./50.)
    filter_bin2 = pt_truth_filtered > 10./50.
    
    same as above comment
    upar_pred_pos_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred > 0.))
    upar_pred_neg_bin0 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin0, upar_pred < 0.))
    upar_pred_pos_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred > 0.))
    upar_pred_neg_bin1 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin1, upar_pred < 0.))
    upar_pred_pos_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred > 0.))
    upar_pred_neg_bin2 = tf.boolean_mask(upar_pred, tf.logical_and(filter_bin2, upar_pred < 0.))
    
    upar_pred_pos_bin0 = torch.masked_select(upar_pred, torch.logical_and(filter_bin0, upar_pred > 0.))
    upar_pred_neg_bin0 = torch.masked_select(upar_pred, torch.logical_and(filter_bin0, upar_pred < 0.))
    upar_pred_pos_bin1 = torch.masked_select(upar_pred, torch.logical_and(filter_bin1, upar_pred > 0.))
    upar_pred_neg_bin1 = torch.masked_select(upar_pred, torch.logical_and(filter_bin1, upar_pred < 0.))
    upar_pred_pos_bin2 = torch.masked_select(upar_pred, torch.logical_and(filter_bin2, upar_pred > 0.))
    upar_pred_neg_bin2 = torch.masked_select(upar_pred, torch.logical_and(filter_bin2, upar_pred < 0.))



 
    norm = torch.sum(pt_truth_filtered)
    dev = torch.abs(torch.sum(upar_pred_pos_bin0) + torch.sum(upar_pred_neg_bin0))
    dev += torch.abs(torch.sum(upar_pred_pos_bin1) + torch.sum(upar_pred_neg_bin1))
    dev += torch.abs(torch.sum(upar_pred_pos_bin2) + torch.sum(upar_pred_neg_bin2))
    dev /= norm

    loss = 0.5 * torch.mean((px_pred - px_truth)**2 + (py_pred - py_truth)**2)

    #loss += 200.*dev
    loss += 10.*dev
    return loss

'''