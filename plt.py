from utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
import os.path as osp

import torch
import torch.nn as nn

from evaluate import evaluate
from evaluate import plot_eval
from tqdm import tqdm

import model.net as net
import model.data_loader as data_loader
import utils
import loss
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--ckpts', default='/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8',
                    help="Name of the ckpts folder")
parser.add_argument('--save_file', default='/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8_pngs',
                    help="Name of the ckpts folder")
parser.add_argument('--data', default='/hildafs/projects/phy230010p/xiea/npzs/FCNN_data',
                    help="Name of the data folder")


args = parser.parse_args()
a = load(args.ckpts + '/' +args.restore_file+ '.resolutions')

model_dir = args.ckpts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
metrics = net.metrics

print("loading model")
'''
model = net.FCNN_MET(continuous_dim=8, categorical_dim=3).to(device) #include puppi
optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.001)
loss_fn = loss.standard_loss
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = 0.0001, max_lr = 0.001, cycle_momentum = False)

restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)

print("loading data")
dataloaders = data_loader.fetch_dataloader(data_dir=args.data, 
                                               batch_size=128,
                                               validation_split=.3)

    
test_dl = dataloaders['test']

print("evaluating")
test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics)
a = resolutions
'''

'''
code in here is for when plotting multiple loss functions 

loss_fns = [var for var in dir(loss) if 'loss' in  var]
models = dict()
for loss_fn_name in loss_fns:
    model = net.FCNN_MET(continuous_dim=8, categorical_dim=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = 0.0001, max_lr = 0.001, cycle_momentum = False)
    loss_fn = loss.loss_fn_name

    if loss_fn_name == 'standard_loss':
        restore_ckpt = osp.join(model_dir+'/FCNN_train3', args.restore_file + '.pth.tar')
    elif loss_fn_name == 'response_correction_loss':
        restore_ckpt = osp.join(model_dir+'/FCNN_train4', args.restore_file + '.pth.tar')
    else:
        continue
    
    ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
    models.update((loss_fn_name,(model,loss_fn)))
    

resolutions = plot_eval(models, device, loss_fn, test_dl)
 
'''



colors = {
    'pfMET': 'black',
    'puppiMET': 'red',
    'deepMETResponse': 'blue',
    'deepMETResolution': 'green',
    'MET':  'magenta',
}
label_arr = {
    'MET':     'DeepMET Pytorch (no tuning)' ,
    'pfMET':    'PF MET',
    'puppiMET': 'PUPPI MET',
    'deepMETResponse': 'DeepMETResponse',
    'deepMETResolution': 'DeepMETResolution',
}
resolutions_arr = {
    'MET':      [[],[],[]],
    'pfMET':    [[],[],[]],
    'puppiMET': [[],[],[]],
    'deepMETResponse': [[],[],[]],
    'deepMETResolution': [[],[],[]],
}
num_bins=40
for key in resolutions_arr:
         plt.figure(1)
         xx = a[key]['u_perp_resolution'][1][0:num_bins]
         yy = a[key]['u_perp_resolution'][0]
         
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(2)
         xx = a[key]['u_perp_scaled_resolution'][1][0:num_bins]
         yy = a[key]['u_perp_scaled_resolution'][0]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(3)
         xx = a[key]['u_par_resolution'][1][0:num_bins]
         yy = a[key]['u_par_resolution'][0]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(4)
         xx = a[key]['u_par_scaled_resolution'][1][0:num_bins]
         yy = a[key]['u_par_scaled_resolution'][0]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(5)
         xx = a[key]['R'][1][0:num_bins]
         yy = a[key]['R'][0]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])

if(True):
    save_dir=args.save_file+'/'+args.restore_file+'_'
    plt.figure(1)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(save_dir+'resol_perp.png')
    plt.clf()
    plt.close()

    plt.figure(2)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(save_dir+'resol_perp_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(3)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(save_dir+'resol_parallel.png')
    plt.clf()
    plt.close()

    plt.figure(4)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(save_dir+'resol_parallel_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(5)
    plt.axis([0, 400, 0, 1.2])
    plt.axhline(y=1.0, color='black', linestyle='-.')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
    plt.legend()
    plt.savefig(save_dir+'response_parallel.png')
    plt.clf()
    plt.close()



