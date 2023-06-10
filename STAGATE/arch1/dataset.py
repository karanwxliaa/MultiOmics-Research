import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

class MyDataset(tf.keras.utils.Sequence):
    def __init__(self, genes_anndata, proteins_anndata, batch_size=32):
        self.genes_anndata = genes_anndata
        self.proteins_anndata = proteins_anndata
        self.batch_size = batch_size
        self.num_samples = genes_anndata.shape[0]

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, self.num_samples)

        genes_batch = self.genes_anndata.X[start_index:end_index]
        proteins_batch = self.proteins_anndata.X[start_index:end_index]

        return genes_batch.astype(np.float32), proteins_batch.astype(np.float32)
    
    def custom_collate_fn(batch):
        genes_batch, proteins_batch = zip(*batch)
        return np.stack(genes_batch), np.stack(proteins_batch)

