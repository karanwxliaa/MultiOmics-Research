import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import networkx as nx



@tf.function
def train_step(model, optimizer, genes, proteins):
    with tf.GradientTape() as tape:
        reconstructed = model([genes, proteins])
        loss = tf.reduce_mean(tf.square(reconstructed - tf.concat([genes, proteins], axis=1)))

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def train(model, optimizer, dataset, num_epochs):
    total_loss = tf.Variable(0.0, dtype=tf.float32)  # Initialize as float32
    num_batches = tf.Variable(0, dtype=tf.int32)  # Initialize as int32

    for epoch in range(num_epochs):
        for genes, proteins in dataset:
            loss = train_step(model, optimizer, genes, proteins)
            total_loss.assign_add(loss)
            num_batches.assign_add(1)

        average_loss = total_loss / tf.cast(num_batches, tf.float32)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {average_loss:.4f}")

        # Reset total_loss and num_batches for the next epoch
        total_loss.assign(0.0)
        num_batches.assign(0)

    return average_loss


def compute_spatial_graph(coordinates, radius):
    pairwise_distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    adjacency_matrix = np.where(pairwise_distances <= radius, 1, 0)
    graph = nx.Graph(adjacency_matrix)
    return graph

