import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, latent_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, gene_input_dim, protein_input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.gene_encoder = Encoder(gene_input_dim, latent_dim)
        self.gene_decoder = Decoder(latent_dim, gene_input_dim)
        self.protein_encoder = Encoder(protein_input_dim, latent_dim)
        self.protein_decoder = Decoder(latent_dim, protein_input_dim)
    
    def forward(self, gene_input, protein_input):
        gene_latent = self.gene_encoder(gene_input)
        gene_output = self.gene_decoder(gene_latent)
        protein_latent = self.protein_encoder(protein_input)
        protein_output = self.protein_decoder(protein_latent)
        return gene_output, protein_output
