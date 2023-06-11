import torch
import torch.optim as optim
import numpy as np
import scanpy as sc
import pandas as pd
from arch1 import ArchModel
import torch.nn as nn


# Load your gene data
file=''
adata = sc.read_visium(file, count_file=r'C:\Users\KARAN\Desktop\MultiOmics-Research\STAGATE\Landau\SPOTS Landau paper dataset\protein\GSE198353_mmtv_pymt_GEX_filtered_feature_bc_matrix.h5',load_images=True)
adata.var_names_make_unique()

# Load your protein data
pdata = pd.read_csv(r'C:\Users\KARAN\Desktop\MultiOmics-Research\STAGATE\Landau\SPOTS Landau paper dataset\protein\GSE198353_mmtv_pymt_ADT_t.csv', index_col=0)

# Add protein data to AnnData object
adata.obsm['protein_data'] = pdata.values


# Define hyperparameters
gene_input_dim = adata.n_vars
protein_input_dim = pdata.shape[1]
gene_hidden_dim = 128
protein_hidden_dim = 128
latent_dim = 64
spatial_dim = 32
num_epochs = 100
learning_rate = 0.001

# Create model
model = ArchModel(gene_input_dim, protein_input_dim, gene_hidden_dim, protein_hidden_dim, latent_dim, spatial_dim)


# Define loss function
criterion = nn.MSELoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import ArchModel
from utils import MultimodalDataset

import torch
from torch.utils.data import DataLoader
from model import ArchModel
from utils import MultimodalDataset


def train_arch1(adata, gene_input_dim, protein_input_dim, gene_hidden_dim, protein_hidden_dim, latent_dim, spatial_dim, num_epochs, batch_size, learning_rate):
    # Convert AnnData object to dense matrices
    gene_data = adata.X.A
    protein_data = adata.obsm["protein_data"]
    spatial_data = adata.obsm["spatial"]

    # Create a PyTorch dataset
    dataset = MultimodalDataset(gene_data, protein_data, spatial_data)

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = ArchModel(gene_input_dim, protein_input_dim, gene_hidden_dim, protein_hidden_dim, latent_dim, spatial_dim)

    # Define the loss function
    criterion = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    total_loss = 0
    for epoch in range(num_epochs):
        for gene_batch, protein_batch, spatial_batch in dataloader:
            # Reshape the spatial input
            spatial_batch = spatial_batch.view(-1, spatial_dim)

            # Forward pass
            gene_output, protein_output = model(gene_batch, protein_batch, spatial_batch)

            # Compute loss
            gene_loss = criterion(gene_output, gene_batch)
            protein_loss = criterion(protein_output, protein_batch)
            loss = gene_loss + protein_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss}")
            total_loss = 0

    return model


'''
# Convert AnnData object to tensors
gene_data = torch.FloatTensor(adata.X)
protein_data = torch.FloatTensor(adata.obsm['protein_data'])
spatial_data = torch.FloatTensor(adata.obsm['spatial'])

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    gene_output, protein_output, spatial_latent = model(gene_data, protein_data)

    # Calculate loss
    loss = criterion(gene_output, gene_data) + criterion(protein_output, protein_data) + criterion(spatial_latent, spatial_data)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'arch_model.pth')

'''