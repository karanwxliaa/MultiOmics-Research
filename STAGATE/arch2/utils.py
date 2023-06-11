import numpy as np


def create_batches(gene_data, protein_data, batch_size):
    num_samples = gene_data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        gene_batch = gene_data[indices[start:end]]
        protein_batch = protein_data[indices[start:end]]
        yield gene_batch, protein_batch
