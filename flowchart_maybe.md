## Architecture 1
*High-Level Diagram:*
```mermaid
graph LR
A(Multi-Omic Data) --> C(Multi-Omic Autoencoder)
B(Spatial Data) --> D(Spatial Attention)
C --> E(Clustering)
D --> C
```
*Low-Level Diagram:*
```mermaid
graph TD
subgraph Architecture 1
    A(Multi-Omic Data)
    B(Spatial Data)
    C(Multi-Omic Autoencoder)
    D(Spatial Attention)
    E(Clustering)
    
    A --> C[Multi-Omic Encoder]
    B --> D
    C --> E
    D --> C
    end
```
**Architecture 1: Multi-Omic Autoencoder with Spatial Attention** <br>
Multi-omic data is fed into separate autoencoders for each omic type to learn their respective latent spaces. 
Spatial information (e.g., spatial coordinates) is incorporated using a spatial attention mechanism that highlights relevant spatial features. 
The latent representations from the autoencoders and spatial attention module

## Architecture 2
*High-Level Diagram:*

```mermaid
graph LR

F(Multi-Omic Data) --> H(Graph Convolutional Autoencoder)
G(Spatial Data) --> H
H --> I(Clustering)
```
*Low-Level Diagram:*
```mermaid
graph TD

subgraph Architecture 2
    F(Multi-Omic Data)
    G(Spatial Data)
    H(Graph Convolutional Encoder)
    I(Graph Convolutional Decoder)
    J(Clustering)
    
    F --> H
    G --> H
    H --> I
    I --> J
    end
```
**Architecture 2: Graph Convolutional Autoencoder with Multi-Omic Inputs** <br>
Multi-omic data is represented as a graph, where nodes represent samples and edges represent spatial relationships. 
Graph convolutional autoencoders are used to learn latent representations of the multi-omic data while capturing spatial dependencies. 
The latent representations are then used for downstream tasks such as clustering, where spatial relationships contribute to the clustering process. 

## Architecture 3
*High-Level Diagram:*
```mermaid
graph LR

J(Multi-Omic Data) --> L(Variational Autoencoder)
K(Spatial Data) --> L
L --> N(Clustering)
L --> M(Spatial Graph Embeddings)
```
*Low-Level Diagram:*
```mermaid
graph TD
subgraph Architecture 3
    J(Multi-Omic Data)
    K(Spatial Data)
    L(Variational Encoder)
    M(Variational Decoder)
    N(Spatial Graph Encoder)
    O(Spatial Graph Decoder)
    P(Spatial Graph Embeddings)
    Q(Clustering)
    
    J --> L
    K --> L
    L --> M
    M --> N
    N --> O
    K --> N
    O --> Q
    N --> P
    J --> P
    P --> Q
    end
```
**Architecture 3: Variational Autoencoder with Spatial Graph Embeddings** <br>
Variational autoencoders (VAEs) are employed to model the latent space of each omic type. 
Spatial information is incorporated by constructing a spatial graph, where nodes represent samples and edges encode spatial relationships. 
Graph embedding techniques are applied to learn spatial representations, which are fused with the VAE latent spaces for downstream clustering tasks. 

## Architecture 4
*High-Level Diagram:*

```mermaid
graph LR

O(Multi-Omic Data) --> Q(Multi-Omic Autoencoder)
P(Spatial Data) --> Q
Q --> R(Spatial Transformer Network)
R --> S(Clustering)
```
*Low-Level Diagram:*
```mermaid
graph TD

subgraph Architecture 4
    O(Multi-Omic Data)
    P(Spatial Data)
    Q(Multi-Omic Encoder)
    R(Spatial Transformer Network)
    S(Clustering)
    
    O --> Q
    P --> Q
    Q --> R
    R --> S
    end
```
**Architecture 4: Spatial Transformer Network with Multi-Omic Autoencoders** <br>
Multi-omic data is processed by individual autoencoders to obtain latent representations. 
Spatial transformer networks (STNs) are used to learn spatial transformations, enabling the model to align and integrate the spatial information. 
The transformed spatial features and the latent representations from the autoencoders are combined and used for spatial clustering tasks. 

## Architecture 5
*High-Level Diagram:*
```mermaid
graph LR

T(Multi-Omic Data) --> V(Dual-Path Autoencoder)
U(Spatial Data) --> V
V --> X(Clustering)
V --> W(Spatial Graph Convolution)
```
*Low-Level Diagram:*
```mermaid
graph TD

subgraph Architecture 5
    T(Multi-Omic Data)
    U(Spatial Data)
    V(Dual-Path Autoencoder)
    W(Spatial Graph Convolution)
    X(Clustering)
    
    T --> V
    U --> V
    V --> W
    W --> X
    end
```
**Architecture 5: Dual-Path Autoencoder with Spatial Graph Convolution** <br>
Dual-path autoencoders are utilized to learn separate latent representations for each omic type while preserving their unique characteristics. 
Spatial information is incorporated through graph convolutional layers, which capture spatial dependencies and encode them in the latent space. 
The fused latent representations and spatial graph convolutions are employed for downstream tasks such as spatial clustering. 
