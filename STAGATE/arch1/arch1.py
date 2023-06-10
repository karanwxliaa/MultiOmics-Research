import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveAttentionLayer(nn.Module):
    def __init__(self, input_dim, spatial_graph):
        super(AdaptiveAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.spatial_graph = spatial_graph
        
        self.attention_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.xavier_uniform_(self.attention_weights)
        
    def forward(self, input_tensor):
        attention_scores = torch.matmul(input_tensor, self.attention_weights)
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        attended_tensor = torch.matmul(self.spatial_graph, attention_scores.transpose(0, 1))
        attended_tensor = attended_tensor.transpose(0, 1)
        
        return attended_tensor

class Arch1(nn.Module):
    def __init__(self, num_genes, num_proteins, latent_dim, spatial_graph):
        super(Arch1, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder for gene data
        self.encoder_genes = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Encoder for protein data
        self.encoder_proteins = nn.Sequential(
            nn.Linear(num_proteins, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        
        # Spatial Attention Layer
        self.attention_layer = AdaptiveAttentionLayer(latent_dim, spatial_graph)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_genes + num_proteins),
            nn.Sigmoid()
        )
    
    def forward(self, genes, proteins):
        # Encode gene and protein data
        latent_genes = self.encoder_genes(genes)
        latent_proteins = self.encoder_proteins(proteins)
        
        # Combine latent representations
        latent_combined = torch.cat((latent_genes, latent_proteins), dim=1)
        
        # Apply spatial attention
        attended_latent = self.attention_layer(latent_combined)
        
        # Decode attended latent representation
        reconstruction = self.decoder(attended_latent)
        
        return reconstruction

