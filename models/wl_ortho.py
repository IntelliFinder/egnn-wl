
import torch.nn as nn
import torch
from .basic_layers import Residual, Dense


class TwoFDisInit(nn.Module):
    def __init__(self,
                 ef_dim: int,   
                 k_tuple_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 **kwargs
                 ):
        super().__init__()

        self.ef_lin = Dense(
            in_features=ef_dim,
            out_features=k_tuple_dim,
            bias=False,
            activation_fn=None
        )

        self.pattern_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=k_tuple_dim,
            padding_idx=0
        )
        
        self.mix_lin = Residual(
            hidden_dim=k_tuple_dim,
            activation_fn=activation_fn,
            mlp_num=2
        )
        


    def forward(self,
                ef: torch.Tensor
                ):
        
        ef0 = self.ef_lin(ef.clone())
        
        ef_mixed = ef0 # (B, N, N, k_tuple_dim)
        

        B = ef_mixed.shape[0]
        N = ef_mixed.shape[1]
        
        idx = torch.arange(N)
        tuple_pattern = torch.ones(size=(B, N, N), dtype=torch.int64, device=ef_mixed.device)
        tuple_pattern[:, idx, idx] = 2
        tuple_pattern = self.pattern_embedding(tuple_pattern) # (B, N, N, k_tuple_dim)
        
        emb2 = ef_mixed * tuple_pattern 
        
        emb2 = self.mix_lin(emb2)
        
        return emb2
        

class TwoFDisLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 activation_fn: nn.Module = nn.SiLU(),
                 residual : bool = False,
                 **kwargs
                 ):
        super().__init__()
        
        self.emb_lins = nn.ModuleList(
            [
                nn.Sequential(
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    ),
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    )
                ) for _ in range(3)
            ] 
        )
        


        if residual:
            self.output_lin = Residual(
                    mlp_num=2,
                    hidden_dim=hidden_dim,
                    activation_fn=activation_fn,
                    )
        else:
            print("USE DENSE OUTPUT")
            self.output_lin = nn.Sequential(
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    ),
                    Dense(
                        in_features=hidden_dim,
                        out_features=hidden_dim,
                        activation_fn=activation_fn
                    )
                )
        

    def forward(self, 
                kemb: torch.Tensor,
                **kwargs
                ):
        '''
            kemb: (B, N, N, hidden_dim)
        '''
        
        
        self_message, kemb_0, kemb_1 = [self.emb_lins[i](kemb.clone()) for i in range(3)]
        
        kemb_0, kemb_1 = (kemb_0.permute(0, 3, 1, 2), kemb_1.permute(0, 3, 1, 2))
        
        kemb_multed = torch.matmul(kemb_0, kemb_1).permute(0, 2, 3, 1)

        kemb_out = self.output_lin(self_message * kemb_multed)
        
        return kemb_out


    