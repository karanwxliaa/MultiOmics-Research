import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GeneProteinDataset

from model import Autoencoder


# Constants
GENE_INPUT_DIM = 5000
PROTEIN_INPUT_DIM = 1000
LATENT_DIM = 256
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.001


def train():
    # Load the dataset
    dataset = GeneProteinDataset(...)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Create the model
    model = Autoencoder(GENE_INPUT_DIM, PROTEIN_INPUT_DIM, LATENT_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    total_samples = len(dataset)
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for batch in dataloader:
            gene_input = batch['gene_data']
            protein_input = batch['protein_data']
            
            # Forward pass
            gene_output, protein_output = model(gene_input, protein_input)
            loss = criterion(gene_output, gene_input) + criterion(protein_output, protein_input)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * gene_input.size(0)
        
        epoch_loss = running_loss / total_samples
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    train()
