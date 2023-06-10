import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from arch1 import Arch1
from dataset import MyDataset
from utils import train, compute_spatial_graph


# Set random seed for reproducibility
torch.manual_seed(2023)
np.random.seed(2023)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Training Arch1 model')
parser.add_argument('--genes', type=str, required=True, help='Path to gene expression data')
parser.add_argument('--proteins', type=str, required=True, help='Path to protein expression data')
parser.add_argument('--spatial', type=str, required=True, help='Path to spatial graph')
parser.add_argument('--latent_dim', type=int, default=32, help='Dimension of latent space')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
args = parser.parse_args()

# Load data and create dataloaders
genes_data = ...  # Load gene expression data
proteins_data = ...  # Load protein expression data
spatial_graph = ...  # Load spatial graph
dataset = MyDataset(genes_data, proteins_data)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Initialize model and optimizer
# Initialize model and optimizer
model = Arch1(gdata.shape[1], pdata.shape[1], args.latent_dim, spatial_graph).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Training loop
for epoch in range(args.num_epochs):
    train_loss = train(model, dataloader, optimizer)
    print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}")


# Compute spatial graph
radius = 10.0  # Adjust the radius value as needed
spatial_graph = compute_spatial_graph(gdata.obsm['spatial'], radius)
