import torch
from torch.utils.data import DataLoader
from model import ArchModel
from utils import MultimodalDataset


def train_arch1(adata, gene_input_dim, protein_input_dim, gene_hidden_dim, protein_hidden_dim, latent_dim, spatial_dim, num_epochs, batch_size, learning_rate):
    # Convert AnnData object to dense matrices
    gene_data = adata.X.A
    protein_data = adata.obsm["protein_expression"]
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
