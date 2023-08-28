import json
import os.path as osp
import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import argparse
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate
import warnings
warnings.simplefilter('ignore')
from time import strftime, gmtime
import loss

from torch.profiler import profile, record_function, ProfilerActivity
import time

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--data', default='/hildafs/projects/phy230010p/xiea/npzs/FCNN_data',
                    help="Name of the data folder")
parser.add_argument('--ckpts', default='/hildafs/projects/phy230010p/xiea/checkpoints/FCNN_train8',
                    help="Name of the ckpts folder")



def train(model, device, optimizer, scheduler, loss_fn, dataloader, epoch):
    model.train()
    loss_avg_arr = []
    loss_avg = utils.RunningAverage()
    

    with tqdm(total=len(dataloader)) as t:
        for data in dataloader:
            #event_start_t = time.perf_counter_ns()
            optimizer.zero_grad()
            data = data.to(device)
            
            x_cont = data.x[:,:8] #include puppi
            #x_cont = data.x[:,:7] #remove puppi
            x_cat = data.x[:,8:].long()
            
            #data_prep_end_t= time.perf_counter_ns()
    

            # NB: there is a problem right now for comparing hits at the +/- pi boundary
            #inference_start_t = time.perf_counter_ns()
            #weights = model(x_cont, x_cat,timings)

            weights = model(x_cont, x_cat)

            #inference_end_t = time.perf_counter_ns()
            
            #loss_start_t = time.perf_counter_ns()
            loss = loss_fn(weights,data.x,data.y,data.batch)
            #loss_end_t = time.perf_counter_ns()

            #loss_backward_start_t = time.perf_counter_ns()
            loss.backward()
            #loss_backward_end_t = time.perf_counter_ns()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            #optimizer_step_start_t = time.perf_counter_ns()
            optimizer.step()
            #optimizer_step_end_t = time.perf_counter_ns()

            # update the average loss
            loss_avg_arr.append(loss.item())
            loss_avg.update(loss.item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            #event_end_t = time.perf_counter_ns()
            '''
            timings['DATA'].append(data_prep_end_t-event_start_t)
            timings["MI"].append(inference_end_t - inference_start_t)
            timings["LOSSFN"].append(loss_end_t - loss_start_t)
            timings["LOSSBW"].append( loss_backward_end_t-loss_backward_start_t)
            timings["OPTIM"].append(optimizer_step_end_t-optimizer_step_start_t)
            timings["ERT"].append(event_end_t - event_start_t)
            print("dense stack avg:",np.mean(timings['DENSE']))
            print("concat avg:",np.mean(timings['CAT']))
            print("lossfn avg:",np.mean(timings['LOSSFN']))
            print("lossbackward avg:",np.mean(timings['LOSSBW']))
            print("optimizer avg:",np.mean(timings['OPTIM']))
            print("data prep avg:",np.mean(timings["DATA"]))
            print("model inference avg:",np.mean(timings['MI']))
            print("event avg:",np.mean(timings['ERT']))
            '''

    scheduler.step(np.mean(loss_avg_arr))
    print('Training epoch: {:02d}, MSE: {:.4f}'.format(epoch, np.mean(loss_avg_arr)))

    

    return np.mean(loss_avg_arr)

if __name__ == '__main__':
    args = parser.parse_args()
    print("checkpoint dir:",args.ckpts)
    print("data dir:",args.data)
    print("restore file:",args.restore_file)
    loss_fn = loss.response_correction_loss
    print("loss function:",loss_fn)
    print('loading data')
    dataloaders = data_loader.fetch_dataloader(data_dir=args.data, 
                                               batch_size=64,
                                               validation_split=.3)
    
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pdg_map = {-211.0: 0, -13.0: 1, -11.0: 2, 0.0: 3, 1.0: 4, 2.0: 5, 11.0: 6, 13.0: 7, 22.0: 8, 130.0: 9, 211.0: 10}
            
    print(f"data loaded, creating model and sending to device {device}")

    model = net.FCNN_MET(continuous_dim=8, categorical_dim=3).to(device) #include puppi
    
    print("model created")
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay=0.001)

    #need to figure out the call back thing in FCNN_v1
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, threshold=0.05)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr = 0.0001, max_lr = 0.001, cycle_momentum = False)

    first_epoch = 0
    best_validation_loss = 10e7
    #loss_fn = loss.standard_loss
    
    metrics = net.metrics

    model_dir = args.ckpts
    loss_log = open(model_dir+'/loss.log', 'w')
    loss_log.write('# loss log for training starting in '+strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
    loss_log.write('epoch, loss, val_loss\n')
    loss_log.flush()

    # reload weights from restore_file if specified
    if args.restore_file is not None:
        restore_ckpt = osp.join(model_dir, args.restore_file + '.pth.tar')
        ckpt = utils.load_checkpoint(restore_ckpt, model, optimizer, scheduler)
        first_epoch = ckpt['epoch']
        print('Restarting training from epoch',first_epoch)
        with open(osp.join(model_dir, 'metrics_val_best.json')) as restore_metrics:
            best_validation_loss = json.load(restore_metrics)['loss']

    for epoch in range(first_epoch+1,101):

        print('Current best loss:', best_validation_loss)
        if '_last_lr' in scheduler.state_dict():
            print('Learning rate:', scheduler.state_dict()['_last_lr'][0])

        # compute number of batches in one epoch (one full pass over the training set)

        #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], record_shapes=True) as prof:
        '''
        timings = {
            "ERT":[],
            "LOSSFN":[],
            "LOSSBW":[],
            "OPTIM":[],
            "MI":[],
            "EMB":[],
            "CAT":[],
            "DENSE":[],
            "SIG":[],
            "FLAT":[],
            "DATA":[]
        }
        '''
        train_loss = train(model, device, optimizer, scheduler, loss_fn, train_dl, epoch)
        #print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        

        # Save weights
        utils.save_checkpoint({'epoch': epoch,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(),
                               'sched_dict': scheduler.state_dict()},
                              is_best=False,
                              checkpoint=model_dir)

        # Evaluate for one epoch on validation set
        test_metrics, resolutions = evaluate(model, device, loss_fn, test_dl, metrics)

        validation_loss = test_metrics['loss']
        loss_log.write('%d,%.2f,%.2f\n'%(epoch,train_loss, validation_loss))
        loss_log.flush()
        is_best = (validation_loss<=best_validation_loss)

        # If best_eval, best_save_path
        if is_best: 
            print('Found new best loss!') 
            best_validation_loss=validation_loss

            # Save weights
            utils.save_checkpoint({'epoch': epoch,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict(),
                                   'sched_dict': scheduler.state_dict()},
                                  is_best=True,
                                  checkpoint=model_dir)
            
            # Save best val metrics in a json file in the model directory
            utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_best.json'))
            utils.save(resolutions, osp.join(model_dir, 'best.resolutions'))

        utils.save_dict_to_json(test_metrics, osp.join(model_dir, 'metrics_val_last.json'))
        utils.save(resolutions, osp.join(model_dir, 'last.resolutions'))

    loss_log.close()
