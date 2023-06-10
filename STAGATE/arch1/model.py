import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import networkx as nx
from scipy.spatial.distance import pdist, squareform
tf.compat.v1.disable_eager_execution()

class Arch1(tf.keras.Model):
    def __init__(self, num_genes, num_proteins, latent_dim):
        super(Arch1, self).__init__()
        self.num_genes = num_genes
        self.num_proteins = num_proteins
        self.latent_dim = latent_dim

        self.genes_encoder = self.build_encoder(num_genes)
        self.proteins_encoder = self.build_encoder(num_proteins)
        self.decoder = self.build_decoder()

    def build_encoder(self, input_dim):
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.latent_dim, activation='relu')(x)
        return Model(inputs, outputs)

    def build_decoder(self):
        genes_input = Input(shape=(self.latent_dim,))
        proteins_input = Input(shape=(self.latent_dim,))

        genes_decoded = Dense(32, activation='relu')(genes_input)
        proteins_decoded = Dense(32, activation='relu')(proteins_input)

        merged = Concatenate()([genes_decoded, proteins_decoded])
        outputs = Dense(self.num_genes + self.num_proteins, activation='relu')(merged)

        return Model([genes_input, proteins_input], outputs)

    def call(self, genes_inputs, proteins_inputs):
        genes_latent = self.genes_encoder(genes_inputs)
        proteins_latent = self.proteins_encoder(proteins_inputs)
        decoded_output = self.decoder([genes_latent, proteins_latent])
        return decoded_output
