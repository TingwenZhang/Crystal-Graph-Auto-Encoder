import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GINEConv, BatchNorm, Set2Set
from typing import Tuple

class GAE(nn.Module):
    """An auto-encoder to reconstruct `ase.Atoms`"""

    def __init__(self,
        in_channels: int = 119,
        chem_dim: int = 11,
        hidden_dim: int = 64,
        edge_dim: int = 4,
        latent_dim: int = 64,
        num_conv_layers: int = 3,
        dropout_rate: float = 0.1,
        activation_ae: str = "swish"):
        """Creates an auto-encoder.
        
        Parameters
        ----------
        in_channels: dimension of one-hot encode atomic numbers
        chem_dim: number of input features per node in the graph
        hidden_dim: hidden dimension
        edge_dim: edge dimension
        latent_dim: latent dimension
        num_conv_layers: number of convolution layers
        drop_out_rate: drop out rate
        activation_ae: activation function
        
        Returns
        -------
        None
        """
        super().__init__()
        acts = {
            "relu":  nn.ReLU(),
            "gelu":  nn.GELU(),
            "swish": nn.SiLU(),
            "mish":  nn.Mish(),
            "elu":   nn.ELU(),
            "tanh":  nn.Tanh()
        }
        self.act_ae = acts[activation_ae]

        # node & global embeddings unchanged
        # (119+11=130, 64)
        self.node_emb = nn.Linear(in_channels + chem_dim, hidden_dim)

        # edge embedding & convs unchanged
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), self.act_ae,
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.convs = nn.ModuleList([
            GINEConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), self.act_ae,
                    nn.Linear(hidden_dim, hidden_dim)
                ), edge_dim=hidden_dim
            ) for _ in range(num_conv_layers)
        ])

        self.bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_conv_layers)])
        self.dropout = nn.Dropout(dropout_rate)
        self.pool = Set2Set(hidden_dim, processing_steps=3)

        # autoencoder heads
        self.global_enc_ae = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), self.act_ae,
            nn.Linear(hidden_dim, latent_dim)
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2), self.act_ae,
            nn.Linear(hidden_dim//2, in_channels)
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(2*latent_dim, hidden_dim), self.act_ae,
            nn.Linear(hidden_dim, hidden_dim), self.act_ae,  # Increased capacity
            nn.Linear(hidden_dim, edge_dim)  # Output full edge_dim features
        )
        self.edge_pred = nn.Sequential(
            nn.Linear(2*latent_dim, hidden_dim), self.act_ae,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data: torch_geometric.data.Data) -> Tuple[torch.Tensor]:
        """Decompose a `torch_geometric.data.Data` into features.
        
        Parameters
        ----------
        data: a `torch_geometric.data.Data` from an `ase.Atoms`
        
        Returns
        -------
        latent_ae: latent space representation of `data`
        node_recon: reconstructed node features of `data`
        edge_logits: binary classification of edge existence in `data`
        edge_recon: reconstructed edge features of `data`
        """
        # embed nodes
        raw = torch.cat([data.x.float(), data.x_node_feats], dim=1)
        x = self.act_ae(self.node_emb(raw))
        x = F.dropout(x, p=self.dropout.p, training=True)

        batch = data.batch

        # message passing
        e_attr = self.edge_embedding(data.edge_attr.float())
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, e_attr)
            x = bn(x)
            x = self.act_ae(x)
            x = self.dropout(x)

        # pooling & latent
        pooled = self.pool(x, batch)
        latent_ae = self.global_enc_ae(pooled)

        # decoders
        global_node = latent_ae[batch]
        node_recon = self.node_decoder(global_node)

        row, col = data.edge_index
        edge_in = torch.cat([global_node[row], global_node[col]], dim=1)

        edge_recon = self.edge_decoder(edge_in)

        # Ensure distance component stays positive
        edge_recon_dist = edge_recon[:, 0].unsqueeze(1) # (1, 4)
        edge_recon_dist = F.softplus(edge_recon_dist)  # Enforce positive distance
        edge_recon = torch.cat([edge_recon_dist, edge_recon[:, 1:]], dim=1)

        edge_logits = self.edge_pred(edge_in)

        # return everything except property_pred
        return latent_ae, node_recon, edge_logits, edge_recon
    
    @torch.no_grad()
    def compress(self, loader: torch_geometric.data.DataLoader, filename: str) -> None:
        f"""Produce the latent space representation of all data. Save the result to `{filename}.pt`.

        Parameters
        ----------
        loader: load `torch_geometric.data.Data`
        filename: save the latent space representation

        Returns
        -------
        None
        """
        latents = torch.cat([self.forward(data)[0].cpu() for data in loader])
        torch.save(latents, f"{filename}.pt")


def train_epoch(model, loader, optimizer):
    model.train()
    stats = {
        'total'     : 0.0,
        'node_loss' : 0.0,
        'edge_feat' : 0.0,
        'edge_bce'  : 0.0,
    }

    for data in loader:
        optimizer.zero_grad()

        # model now returns only 4 outputs
        latent_ae, node_recon, edge_logits, edge_recon = model(data)

        # 1) Node AE + accuracy
        node_loss = F.cross_entropy(node_recon, data.x.argmax(dim=1))
        # node_acc  = compute_node_accuracy(data, node_recon)

        # 2) Edge AE: distance + existence
        edge_feat_loss = F.mse_loss(edge_recon, data.edge_attr.float())

        edge_bce_loss = F.binary_cross_entropy_with_logits(edge_logits, torch.ones_like(edge_logits))

        # combine
        loss = node_loss + edge_feat_loss + edge_bce_loss

        loss.backward()

        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # accumulate
        stats['total'] += loss.item()
        stats['node_loss'] += node_loss.item()
        stats['edge_feat'] += edge_feat_loss.item()
        stats['edge_bce'] += edge_bce_loss.item()

    n = len(loader)
    return {k: stats[k] / n for k in stats}

@torch.no_grad()
def validate_epoch(model, loader):
    model.eval()
    stats = {
        'total'     : 0.0,
        'node_loss' : 0.0,
        'edge_feat' : 0.0,
        'edge_bce'  : 0.0,
    }

    for data in loader:
        latent_ae, node_recon, edge_logits, edge_recon, = model(data)

        node_loss = F.cross_entropy(node_recon, data.x.argmax(dim=1))

        edge_feat_loss = F.mse_loss(edge_recon, data.edge_attr.float())
        edge_bce_loss = F.binary_cross_entropy_with_logits(edge_logits, torch.ones_like(edge_logits))

        ae_loss = node_loss + edge_feat_loss + edge_bce_loss
        loss = ae_loss

        stats['total'] += loss.item()
        stats['node_loss'] += node_loss.item()
        stats['edge_feat'] += edge_feat_loss.item()
        stats['edge_bce'] += edge_bce_loss.item()

    n = len(loader)
    return {k: stats[k] / n for k in stats}
