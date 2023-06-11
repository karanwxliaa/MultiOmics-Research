import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class GATE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GATE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class STAGATE:
    def __init__(self, adata, hidden_dims, alpha, n_epochs):
        self.adata = adata
        self.hidden_dims = hidden_dims
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.scaler = StandardScaler()
        self.gate_net = None
    
    def prepare_graph_data(self):
        # Prepare the graph data (genes and proteins)
        gene_data = self.adata.X.toarray()
        protein_data = self.adata.obsm['protein_data']  # Assuming you have protein data stored in adata.obsm['protein_data']
        
        # Normalize the gene data and protein data
        gene_data = self.scaler.fit_transform(gene_data)
        protein_data = self.scaler.fit_transform(protein_data)
        
        return gene_data, protein_data
    
    def train(self):
        # Prepare the graph data
        gene_data, protein_data = self.prepare_graph_data()
        
        # Create the GATE network
        self.gate_net = GATE(gene_data.shape[1], self.hidden_dims[0])
        
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.gate_net.parameters(), lr=self.alpha)
        
        # Convert the numpy arrays to tensors
        gene_data = torch.tensor(gene_data, dtype=torch.float32)
        protein_data = torch.tensor(protein_data, dtype=torch.float32)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Forward pass
            outputs = self.gate_net(gene_data)
            
            # Compute the loss
            loss = criterion(outputs, protein_data)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print the loss for every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss.item():.4f}")
