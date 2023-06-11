import torch
import torch.nn as nn


class GeneEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GeneEncoder, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        latent = self.latent_layer(hidden)
        return latent


class ProteinEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ProteinEncoder, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        latent = self.latent_layer(hidden)
        return latent


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(SpatialEncoder, self).__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.latent_layer = nn.Linear(hidden_dim, latent_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        latent = self.latent_layer(hidden)
        return latent


class SpatialAttention(nn.Module):
    def __init__(self, input_dim, spatial_dim):
        super(SpatialAttention, self).__init__()
        self.attention_layer = nn.Linear(input_dim, spatial_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        attention_weights = self.activation(self.attention_layer(x))
        attended_latent = x * attention_weights
        return attended_latent


class GeneDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(GeneDecoder, self).__init__()
        self.hidden_layer = nn.Linear(latent_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        output = self.output_layer(hidden)
        return output


class ProteinDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(ProteinDecoder, self).__init__()
        self.hidden_layer = nn.Linear(latent_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        hidden = self.activation(self.hidden_layer(x))
        output = self.output_layer(hidden)
        return output


class ArchModel(nn.Module):
    def __init__(self, gene_input_dim, protein_input_dim, gene_hidden_dim, protein_hidden_dim, latent_dim, spatial_dim):
        super(ArchModel, self).__init__()
        self.gene_encoder = GeneEncoder(gene_input_dim, gene_hidden_dim, latent_dim)
        self.protein_encoder = ProteinEncoder(protein_input_dim, protein_hidden_dim, latent_dim)
        self.spatial_encoder = SpatialEncoder(spatial_dim, spatial_dim, latent_dim)
        self.spatial_attention = SpatialAttention(latent_dim, spatial_dim)
        self.gene_decoder = GeneDecoder(latent_dim, gene_hidden_dim, gene_input_dim)
        self.protein_decoder = ProteinDecoder(latent_dim, protein_hidden_dim, protein_input_dim)

    def forward(self, gene_input, protein_input, spatial_input):
        gene_latent = self.gene_encoder(gene_input)
        protein_latent = self.protein_encoder(protein_input)
        spatial_latent = self.spatial_encoder(spatial_input)
        combined_latent = gene_latent + protein_latent + spatial_latent
        attended_latent = self.spatial_attention(combined_latent)
        gene_output = self.gene_decoder(attended_latent)
        protein_output = self.protein_decoder(attended_latent)
        return gene_output, protein_output
