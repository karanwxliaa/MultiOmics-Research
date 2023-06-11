import torch
from torch.utils.data import Dataset


class GeneProteinDataset(Dataset):
    def __init__(self, gene_data, protein_data):
        self.gene_data = gene_data
        self.protein_data = protein_data
    
    def __len__(self):
        return len(self.gene_data)
    
    def __getitem__(self, index):
        gene_item = self.gene_data[index]
        protein_item = self.protein_data[index]
        
        gene_tensor = torch.Tensor(gene_item)
        protein_tensor = torch.Tensor(protein_item)
        
        return {'gene_data': gene_tensor, 'protein_data': protein_tensor}
