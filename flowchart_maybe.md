### High Level Diagram :

#### Architecture 1
```mermaid
graph LR

A(Multi-Omic Data) --> C(Multi-Omic Autoencoder)
B(Spatial Data) --> C
C --> E(Clustering)
C --> D(Spatial Attention)
```
#### Architecture 2
```mermaid
graph LR

F(Multi-Omic Data) --> H(Graph Convolutional Autoencoder)
G(Spatial Data) --> H
H --> I(Clustering)
```
#### Architecture 3
```mermaid
graph LR

J(Multi-Omic Data) --> L(Variational Autoencoder)
K(Spatial Data) --> L
L --> N(Clustering)
L --> M(Spatial Graph Embeddings)
```
#### Architecture 4
```mermaid
graph LR

O(Multi-Omic Data) --> Q(Multi-Omic Autoencoder)
P(Spatial Data) --> Q
Q --> R(Spatial Transformer Network)
R --> S(Clustering)
```
#### Architecture 5
```mermaid
graph LR

T(Multi-Omic Data) --> V(Dual-Path Autoencoder)
U(Spatial Data) --> V
V --> X(Clustering)
V --> W(Spatial Graph Convolution)
```

### Low Level Diagram :

#### Architecture 1
```mermaid
graph TD

subgraph Architecture 1
    A(Multi-Omic Data)
    B(Spatial Data)
    C(Multi-Omic Autoencoder)
    D(Spatial Attention)
    E(Clustering)
    
    A --> C[Multi-Omic Encoder]
    B --> C
    C --> E
    C --> D
    C --> F[Multi-Omic Decoder]
    D --> G[Spatial Encoder]
    B --> G
    G --> H[Spatial Decoder]
end

```
#### Architecture 2
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
#### Architecture 3
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
#### Architecture 4
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
#### Architecture 5
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
























