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

time_exp_dic = {'time': 0, 'counter': 0}


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


device = torch.device("cuda" if args.cuda else "cpu")
loss_mse = nn.MSELoss()

print(args)
try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass

# prepare data root and save path for checkpoint
data_root = os.path.join(args.data_root, args.data_mode)
   

class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.in_node_nf=1
        self.in_edge_nf=2
        edges_in_d=self.in_edge_nf
        
        self.n_layers=args.n_layers
        self.recurrent=True
        self.norm_diff=args.norm_diff
        self.tanh=args.tanh
        self.lr=config["lr"]
        self.hidden_nf=config["hidden_nf"]
        self.attention = False
        hidden_nf=self.hidden_nf
        input_nf=self.hidden_nf
        output_nf=self.hidden_nf
        nodes_att_dim=0
        self.color_steps = config["color_steps"]
        act_fn=nn.SiLU()
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())
        
        
        self.embedding = nn.Linear(self.in_node_nf, self.hidden_nf)
        self.coords_weight = 1
        #instantiate classes
        
        
        #self.egcl = nn.ModuleList()
        #for i in range(0, self.n_layers):
        #    self.egcl.append(E_GCL_vel_feat(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=self.in_edge_nf, act_fn=nn.SiLU(), coords_weight=1.0, recurrent=self.recurrent, norm_diff=self.norm_diff, tanh=self.tanh, model_config=dict(), color_steps=self.color_steps))
        
        #convolution operators
        #self.convolution_ops = nn.ModuleList()
        #for i in range(0, self.n_layers):
        #    self.convolution_ops.append(self.egcl[i])
        dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset,
                                     max_samples=args.max_training_samples)
        self.loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
        dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
        self.loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)
    
        dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
        self.loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

        self.loss_mse = nn.MSELoss()
        
        # Important: This property activates manual optimization if False.
        self.automatic_optimization = True
        
        self.norm_diff = False
        self.coord_mlp_vel = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.coord_mlp_vel_other = nn.Sequential(
            nn.Linear(input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1))
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))
        self.edge_mlp_feat = nn.Sequential(
                    nn.Linear(input_nf * 2 + 10 + hidden_nf + edges_in_d, hidden_nf),
                    act_fn,
                    nn.Linear(hidden_nf, hidden_nf),
                    act_fn
            )#maybe add a layer
        self.init_color = TwoFDisInit(ef_dim=4, k_tuple_dim=hidden_nf)
        
        
                # interaction layers
        self.interaction_layers = nn.ModuleList()
        for _ in range(self.color_steps):
            self.interaction_layers.append(
                    TwoFDisLayer(
                        hidden_dim=hidden_nf,
                        activation_fn=act_fn,
                        )
                    )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = False
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)
        
    def mixed_wl(self, edge_index, coord, vel):
        row, col = edge_index
        # apply WL to batch
        coord, vel = coord.clone().reshape(coord.size(0)//5, 5, 3), vel.clone().reshape(coord.size(0)//5, 5, 3) # TODO: 5-body task only now
        
        coord_dist = coord.unsqueeze(2) - coord.unsqueeze(1) # (B, N, N, 3)
        coord_dist = torch.norm(coord_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
        
        vel_dist = vel.unsqueeze(2) - vel.unsqueeze(1) # (B, N, N, 3)
        vel_dist = torch.norm(vel_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
        
        mixed_dist = coord.unsqueeze(2) - vel.unsqueeze(1) # (B, N, N, 3)
        mixed_dist_coord = torch.norm(mixed_dist.clone(), dim=-1, keepdim=True) # (B, N, N, 1)
        
        mixed_dist_vel = torch.transpose(mixed_dist_coord, dim0=1, dim1=2)
        
        #add norms 
        #vel_norms = torch.diag_embed(torch.linalg.vector_norm(vel, ord=2, dim=2)).unsqueeze(3) # (B, N, N, 1)
        
        #concatenate along -1 dimension
        mixed_dist = torch.cat([coord_dist, vel_dist, mixed_dist_coord, mixed_dist_vel], dim=-1) # (B, N, N, 4)
        
        #run wl
        kemb = self.init_color(mixed_dist)
        for i in range(self.color_steps):
            kemb += self.interaction_layers[i](
                        kemb=kemb.clone(),
                        )   # (B, N ,N, hidden_nf)
        #return to reg shape and create three lists of index list to query result from wl
        batch   = torch.floor_divide(row, 5)
        rowidx  = torch.remainder(row, 5)
        colidx  = torch.remainder(col, 5)
        
        #assert same sizes
        
        return kemb[batch, rowidx, colidx]
        
        
    def coordvel2feat(self, edge_index, coord, vel):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)
        coord_i_vel_i = coord[row] - vel[row]
        coord_i_vel_i = torch.sum((coord_i_vel_i.clone())**2, 1).unsqueeze(1)
        
        coord_i_vel_j = coord[row] - vel[col]
        coord_i_vel_j = torch.sum((coord_i_vel_j.clone())**2, 1).unsqueeze(1)
        
        coord_j_vel_i = coord[col] - vel[row]
        coord_j_vel_i = torch.sum((coord_j_vel_i.clone())**2, 1).unsqueeze(1)
        
        coord_j_vel_j = coord[col] - vel[col]
        coord_j_vel_j = torch.sum((coord_j_vel_j.clone())**2, 1).unsqueeze(1)
        
        vel_i_vel_j   = vel[row] - vel[col]
        vel_i_vel_j = torch.sum((vel_i_vel_j.clone())**2, 1).unsqueeze(1)
        
        norms = torch.cat(( torch.sum((coord[row])**2, 1).unsqueeze(1),  torch.sum((coord[col])**2, 1).unsqueeze(1), torch.sum((vel[col])**2, 1).unsqueeze(1), torch.sum((vel[col])**2, 1).unsqueeze(1) ), dim=1)
        feat = torch.cat((radial,coord_i_vel_i,coord_i_vel_j,coord_j_vel_i, coord_j_vel_j, vel_i_vel_j, norms ), dim=1)
        
        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return feat, coord_diff
        
    def edge_model_feat(self, source, target, radial, wl_feat, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, wl_feat, edge_attr], dim=1)
        out = self.edge_mlp_feat(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        #trans = torch.clamp(trans, min=-10, max=10) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        #coord = torch.clamp(coord, min=-10, max=10) 
        return coord
        
    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def forward(self, h, x, edge_index, vel, edge_attr, batch_size, n_nodes, node_attr=None):
        row, col = edge_index
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            feat, coord_diff = self.coordvel2feat(edge_index, x, vel)
            wl_feat = self.mixed_wl(edge_index, x, vel)
            
            #coord update
            edge_feat = self.edge_model_feat(h[row], h[col], feat, wl_feat, edge_attr)
            x = self.coord_model(x, edge_index, coord_diff, edge_feat)
    
            x += self.coord_mlp_vel(h) * vel
            h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
            #h, x, _ = self.convolution_ops[i](h, edges, x.clone(), vel.clone(), edge_attr=edge_attr) OLD
        return x
        
    def configure_optimizers(self):
          optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=args.weight_decay, amsgrad=True)
          return optimizer
      
    #def training_step(self, batch, batch_idx):
    #      batch_size, n_nodes, _ = batch[0].size()
    #      batch = [d for d in batch]
    #      batch = [d.view(-1, d.size(2)) for d in batch]
    #      loc, vel, edge_attr, charges, loc_end = batch
    #
    #      edges = self.loader_train.dataset.get_edges(batch_size, n_nodes)
    #      edges = [edges[0], edges[1]]
#  
#  
#          nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
#          rows, cols = edges
#          loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
#          edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
#          loc_pred = self(nodes, loc.detach(), edges, vel, edge_attr, batch_size, n_nodes)
#  
#          loss = self.loss_mse(loc_pred, loc_end)
#          self.log("loss", loss, prog_bar=True)
#          return loss

    def training_step(self, batch, batch_idx):
        #opt = self.optimizers()
        
        batch_size, n_nodes, _ = batch[0].size()
        batch = [d for d in batch]
        batch = [d.view(-1, d.size(2)) for d in batch]
        loc, vel, edge_attr, charges, loc_end = batch

        edges = self.loader_train.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0], edges[1]]


        nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
        loc_pred = self(nodes, loc.detach(), edges, vel, edge_attr, batch_size, n_nodes)

        # compute loss
        loss = self.loss_mse(loc_pred, loc_end)
        self.log("loss", loss, prog_bar=True)
        
        #opt.zero_grad()
        #self.manual_backward(loss)

        # clip gradients
        #self.clip_gradients(opt, gradient_clip_val=args.clip, gradient_clip_algorithm="norm") #TODO norm type

        #opt.step()
        return loss

          #optional
    def validation_step(self, batch, batch_idx):
          batch_size, n_nodes, _ = batch[0].size()
          batch = [d for d in batch]
          batch = [d.view(-1, d.size(2)) for d in batch]
          loc, vel, edge_attr, charges, loc_end = batch
  
          edges = self.loader_val.dataset.get_edges(batch_size, n_nodes)
          edges = [edges[0], edges[1]]
  
  
          nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
          rows, cols = edges
          loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
          edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
          loc_pred = self(nodes, loc.detach(), edges, vel, edge_attr, batch_size, n_nodes)
  
          loss = self.loss_mse(loc_pred, loc_end)
          self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
          return loss
    #test   
    def test_step(self, batch, batch_idx):
         batch_size, n_nodes, _ = batch[0].size()
         batch = [d for d in batch]
         batch = [d.view(-1, d.size(2)) for d in batch]
         loc, vel, edge_attr, charges, loc_end = batch
 
         edges = self.loader_test.dataset.get_edges(batch_size, n_nodes)
         edges = [edges[0], edges[1]]
 
 
         nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
         rows, cols = edges
         loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
         edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties
         loc_pred = self(nodes, loc.detach(), edges, vel, edge_attr, batch_size, n_nodes)
 
         loss = self.loss_mse(loc_pred, loc_end)

         self.log('test_loss', loss, prog_bar=True)
         return loss
         
class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        # dataloaders
        self.dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset,
                                 max_samples=args.max_training_samples)
        self.dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
        
        self.dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_train, batch_size=args.batch_size, num_workers=4, shuffle=True, drop_last=True)
    

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_val, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)
    

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset_test, batch_size=args.batch_size, num_workers=4, shuffle=False, drop_last=False)

search_space = {
    "lr": tune.loguniform(5e-3, 9e-2),
    "hidden_nf": tune.choice([64]),
    "color_steps": tune.choice([2,3]),
}
num_samples = 20

def train_func(config):
    dm = DataModule()
    model = LitModel(config)

    trainer = pl.Trainer(
        #devices=args.devices, 
        #accelerator='gpu', 
        precision=args.precision, 
        max_epochs=args.max_epochs,
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        default_root_dir="/home/snirhordan/egnn-wl-works",
        log_every_n_steps=5,
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
        detect_anomaly=True
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

scaling_config = ScalingConfig(
    num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
)


# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_mnist_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=args.max_epochs, grace_period=5, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()



def main():
    # dataloaders
    dataset_train = NBodyDataset(partition='train', max_samples=args.max_training_samples, data_root=data_root, data_mode=args.data_mode)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    dataset_val = NBodyDataset(partition='valid', data_root=data_root, data_mode=args.data_mode)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    
    dataset_test = NBodyDataset(partition='test', data_root=data_root, data_mode=args.data_mode)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    # init model
    model = LitModel()
    
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    trainer = pl.Trainer(devices=args.devices, 
                         accelerator='gpu', 
                         precision=args.precision, 
                         max_epochs=args.max_epochs, 
                         log_every_n_steps=5, 
                         default_root_dir="/home/snirhordan/egnn-wl", 
                         callbacks=[StochasticWeightAveraging(swa_lrs=1e-2), EarlyStopping(monitor="val_loss", mode="min", patience=50, check_finite=True)], 
                         strategy = "ddp_find_unused_parameters_true")# gradient_clip_val=args.clip,  gradient_clip_algorithm="norm",
    trainer.fit(model, loader_train, loader_val)
    trainer.test(dataloaders=loader_test, ckpt_path='best') #test with best model (auto)

        
def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids.to(data.device), data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids.to(data.device), data)
    count.scatter_add_(0, segment_ids.to(data.device), torch.ones_like(data))
    return result / count.clamp(min=1)

if __name__ == "__main__":
    #main()
    results = tune_mnist_asha(num_samples=num_samples)    
    results.get_best_result(metric="val_loss", mode="min")
    




