import argparse
import torch
from torch import nn

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from pytorch_lightning.callbacks import StochasticWeightAveraging
from n_body_system.dataset_nbody import NBodyDataset
from n_body_system.model import GNN, EGNN, Baseline, Linear, EGNN_vel, Linear_dynamics, RF_vel, EGNN_vel_feat
import os
from torch import nn, optim
import json
import time

import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from models.gcl import GCL, E_GCL, E_GCL_vel, GCL_rf_vel, E_GCL_vel_feat
from models.wl import TwoFDisInit, TwoFDisLayer

from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler


import os
import torch
import tempfile
import pytorch_lightning as pl
import torch.nn.functional as F
from filelock import FileLock
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from models.wl import TwoFDisInit, TwoFDisLayer



torch.set_float32_matmul_precision('medium')

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--samples', type=int, default=10, metavar='N',
                    help='number of samples for hyparameter tuning (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=5e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=64, metavar='N',
                    help='learning rate')
parser.add_argument('--model', type=str, default='egnn_vel', metavar='N',
                    help='available models: gnn, baseline, linear, linear_vel, se3_transformer, egnn_vel, rf_vel, tfn')
parser.add_argument('--attention', type=int, default=0, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--degree', type=int, default=2, metavar='N',
                    help='degree of the TFN and SE3')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--sweep_training', type=int, default=0, metavar='N',
                    help='0 nor sweep, 1 sweep, 2 sweep small')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')
parser.add_argument('--clip', type=float, default=1.5, metavar='N',
                    help='grad clip')
parser.add_argument('--div', type=float, default=1, metavar='N',
                    help='timing experiment')
parser.add_argument('--norm_diff', type=eval, default=False, metavar='N',
                    help='normalize_diff')
parser.add_argument('--tanh', type=eval, default=False, metavar='N',
                    help='use tanh')
#from ClofNet

parser.add_argument('--data_mode', type=str, default='small', metavar='N',
                    help='folder to dataset')
parser.add_argument('--data_root', type=str, default='n_body_system/dataset/data', metavar='N',
                    help='folder to dataset root')

#new
parser.add_argument('--patience', type=int, default=10, metavar='N',
                    help='patience for early stopping')
parser.add_argument('--light', action='store_true', default=False,
                    help='orthogonal transformations in WL')
parser.add_argument('--num_gpus', type=int, default=1, metavar='N',
                    help='# gpus')
parser.add_argument('--precision', type=int, default=32, metavar='N',
                    help='precision in calculations')
parser.add_argument('--max_epochs', type=int, default=500, metavar='N',
                    help='max epochs')
parser.add_argument("--devices", nargs="+", type=int, default=None)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
class Net(nn.Module):
    def __init__(self, hidden_nf=64, n_layers=4, color_steps=3):
        super(Net, self).__init__()
        self.model = EGNN_vel(in_node_nf=1, in_edge_nf=2, hidden_nf=hidden_nf, device=device, n_layers=n_layers, recurrent=True, norm_diff=args.norm_diff, tanh=args.tanh)
         
        self.to(device)
    #forward pass    
    def forward(self, nodes, loc, edges, vel, edge_attr):
         return self.model(nodes, loc, edges, vel, edge_attr)

    
def train_wl(config):
    net = Net(config["hidden_nf"], config["n_layers"], config["color_steps"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    loss_mse = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config["lr"], weight_decay=args.weight_decay, amsgrad=True)

    # To restore a checkpoint, use `train.get_checkpoint()`.
    #loaded_checkpoint = train.get_checkpoint()
    #if loaded_checkpoint:
    #    with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
    #       model_state, optimizer_state = torch.load(os.path.join(loaded_checkpoint_dir, "checkpoint.pt"))
    #    net.load_state_dict(model_state)
    #    optimizer.load_state_dict(optimizer_state)

    dataset_train = NBodyDataset(partition='train', dataset_name="nbody_small",
                                 max_samples=args.max_training_samples)
    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
    
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)

    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    for epoch in range(0, args.epochs):
        running_loss = 0.0
        epoch_steps = 0
        #train
        for batch_idx, data in enumerate(loader_train):
            batch_size, n_nodes, _ = data[0].size()
            data = [d.to(device) for d in data]
            data = [d.view(-1, d.size(2)) for d in data]
            loc, vel, edge_attr, charges, loc_end = data
    
            edges = loader_train.dataset.get_edges(batch_size, n_nodes)
            edges = [edges[0].to(device), edges[1].to(device)]
              
            #zero parameter gradientw
            optimizer.zero_grad()
    
            # forward + backward + clip + optimize
            nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
            rows, cols = edges
            loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
            edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
            loc_pred = net(nodes, loc.detach(), edges, vel, edge_attr)
     
            loss = loss_mse(loc_pred, loc_end)
    
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), args.clip, norm_type=2.0, error_if_nonfinite=True)
            optimizer.step()
            #res['loss'] += loss.item()*batch_size
            #res['counter'] += batch_size
    
            # print statistics
            running_loss += loss.item()*batch_size
            epoch_steps += batch_size
            if batch_idx % 10 == 0:  # print every 10 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_idx + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
                
        # Validate
        val_loss = 0.0
        val_steps = 0
        for batch_idx, data in enumerate(loader_val):
            with torch.no_grad():
                batch_size, n_nodes, _ = data[0].size()
                data = [d.to(device) for d in data]
                data = [d.view(-1, d.size(2)) for d in data]
                loc, vel, edge_attr, charges, loc_end = data
        
                edges = loader_val.dataset.get_edges(batch_size, n_nodes)
                edges = [edges[0].to(device), edges[1].to(device)]
        
                # forward
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
                loc_pred = net(nodes, loc.detach(), edges, vel, edge_attr)
         
                loss = loss_mse(loc_pred, loc_end)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and can be accessed through `train.get_checkpoint()`
        # API in future iterations.
        os.makedirs("my_model", exist_ok=True)
        torch.save(
            (net.state_dict(), optimizer.state_dict()), "my_model/checkpoint.pt")
        checkpoint = Checkpoint.from_directory("my_model")
        train.report({"loss": (val_loss / val_steps)}, checkpoint=checkpoint) #TOFO: change to val_loss here and other places
    print("Finished Training")

def test_best_model(best_result):
    best_trained_model = Net(best_result.config["hidden_nf"], best_result.config["n_layers"], best_result.config["color_steps"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader_test):
                batch_size, n_nodes, _ = data[0].size()
                data = [d.to(device) for d in data]
                data = [d.view(-1, d.size(2)) for d in data]
                loc, vel, edge_attr, charges, loc_end = data
        
                edges = loader_test.dataset.get_edges(batch_size, n_nodes)
                edges = [edges[0].to(device), edges[1].to(device)]
        
                # forward
                nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
                rows, cols = edges
                loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
                edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
                loc_pred = best_trained_model(nodes, loc.detach(), edges, vel, edge_attr)
         
                loss = loss_mse(loc_pred, loc_end)
                test_loss += loss.cpu().numpy()*batch_size
                test_steps += batch_size

    print("Best trial test loss: {}".format(test_loss / test_steps))
    


def main(num_samples=args.samples, max_num_epochs=args.epochs, gpus_per_trial=args.num_gpus):
    config = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "hidden_nf": tune.choice([60, 64, 68]),
    "color_steps": tune.choice([2, 3, 4]),
    "n_layers": tune.choice([3, 4])
    }
    scheduler = ASHAScheduler( #TODO: how does this affect training?
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_wl),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))

    test_best_model(best_result)

        
if __name__ == "__main__":
    main()

    




