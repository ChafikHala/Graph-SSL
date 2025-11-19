# Graph Convolutional Networks (GCNs) Exploration

![GCN Banner](docs/images/gcn_banner.png)

This repository contains a structured exploration of **Graph Convolutional Networks (GCNs)**.  
It covers the **theoretical foundations**, **implementations**, **experiments on benchmark datasets**, and **visualizations of graph learning dynamics**.  
This work serves as a foundation for understanding Graph Neural Networks before moving to Graph Self-Supervised Learning (SSL).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Theoretical Background](#theoretical-background)
    - [Spectral Graph Convolutions and GCN Derivation](#spectral-graph-convolutions-and-gcn-derivation)
        - [1. Spectral Convolution on Graphs](#1-spectral-convolution-on-graphs)
            - [A. Normalized Graph Laplacian](#a-normalized-graph-laplacian)
            - [B. Spectral Convolution Definition](#b-spectral-convolution-definition)
        - [2. Approximation via Chebyshev Polynomials](#2-approximation-via-chebyshev-polynomials)
        - [3. GCN Approximation](#3-gcn-approximation)
            - [A. First-Order Model (K=1)](#a-first-order-model-k1)
            - [B. Single Parameter Constraint](#b-single-parameter-constraint)
            - [C. Renormalization Trick](#c-renormalization-trick)
        - [4. Layer-wise Propagation](#4-layer-wise-propagation-in-gcn)
        - [5. Two-Layer Semi-Supervised GCN](#5-two-layer-semi-supervised-gcn)
5. [Implementation](#implementation)
6. [Experiments](#experiments)
7. [Visualizations](#visualizations)
8. [References](#references)
9. [License](#license)

---

## Introduction

Graph Convolutional Networks (GCNs) extend convolution operations to graph-structured data.  
They leverage **both node features and graph structure** for semi-supervised learning tasks such as **node classification**, **link prediction**, and **graph classification**.

GCNs have become foundational in **graph representation learning** and are a prerequisite to advanced methods like **graph self-supervised learning (SSL)**.

---
## Theoretical Background

### 1. Spectral Convolution on Graphs

#### A. Normalized Graph Laplacian

For an undirected graph $\mathcal{G}=(\mathcal{V},\mathcal{E})$ with $N$ nodes, the **normalized graph Laplacian** is defined as:

$$
\mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
$$

where:

- $\mathbf{A} \in \mathbb{R}^{N \times N}$ is the adjacency matrix.
- $\mathbf{D} \in \mathbb{R}^{N \times N}$ is the degree matrix, with $D_{ii} = \sum_j A_{ij}$.
- $\mathbf{I}_N$ is the identity matrix of size $N$.

The Laplacian can be decomposed spectrally as:

$$
\mathbf{L} = \mathbf{U} \mathbf{\Lambda} \mathbf{U}^\top
$$

where:

- $\mathbf{U}$ is the matrix of eigenvectors.
- $\mathbf{\Lambda}$ is a diagonal matrix of eigenvalues.


#### B. Spectral Convolution Definition

A convolution of a signal $\mathbf{x} \in \mathbb{R}^N$ with a filter $g_\theta$ is defined in the Fourier domain as:

$$
g_\theta \ast \mathbf{x} = \mathbf{U} \, g_\theta(\mathbf{\Lambda}) \, \mathbf{U}^\top \mathbf{x}
$$

Here, $g_\theta(\mathbf{\Lambda})$ is a function of the Laplacian eigenvalues.  
This approach is **computationally expensive**, with complexity $\mathcal{O}(N^2)$.


### 2. Approximation via Chebyshev Polynomials

To avoid the eigen-decomposition, $g_\theta(\mathbf{\Lambda})$ can be approximated using **truncated Chebyshev polynomials** up to order $K$:

$$
g_{\theta'}(\mathbf{\Lambda}) \approx \sum_{k=0}^{K} \theta'_k \, T_k(\tilde{\mathbf{\Lambda}})
$$

where:

- $\tilde{\mathbf{\Lambda}} = \frac{2}{\lambda_\text{max}} \mathbf{\Lambda} - \mathbf{I}_N$ is the rescaled eigenvalue matrix.
- $\lambda_\text{max}$ is the largest eigenvalue of $\mathbf{L}$.
- $T_k(x)$ are Chebyshev polynomials defined recursively.

The convolution becomes:

$$
g_{\theta'} \ast \mathbf{x} \approx \sum_{k=0}^{K} \theta'_k \, T_k(\tilde{\mathbf{L}}) \, \mathbf{x}
$$

with $\tilde{\mathbf{L}} = \frac{2}{\lambda_\text{max}} \mathbf{L} - \mathbf{I}_N$.  
This is a **$K$-localized operation** with complexity $\mathcal{O}(|\mathcal{E}|)$.



### 3. GCN Approximation

#### A. First-Order Model ($K=1$)

Limiting to $K=1$ and approximating $\lambda_\text{max} \approx 2$, we have:

$$
g_{\theta'} \ast \mathbf{x} \approx \theta'_0 \mathbf{x} + \theta'_1 (\mathbf{L} - \mathbf{I}_N) \mathbf{x}
$$

Substituting $\mathbf{L} - \mathbf{I}_N = - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ gives:

$$
g_{\theta'} \ast \mathbf{x} \approx \theta'_0 \mathbf{x} - \theta'_1 \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2} \mathbf{x}
$$



#### B. Single Parameter Constraint

Setting $\theta = \theta'_0 = -\theta'_1$:

$$
g_\theta \ast \mathbf{x} \approx \theta \, (\mathbf{I}_N + \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}) \, \mathbf{x}
$$



#### C. Renormalization Trick

To improve numerical stability:

- Add **self-loops**: $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$
- Compute the degree: $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$

The **renormalized propagation rule** becomes:

$$
\mathbf{Z} = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \, \mathbf{X} \mathbf{\Theta}
$$

where $\mathbf{X} \in \mathbb{R}^{N \times C}$ is the input feature matrix and $\mathbf{\Theta} \in \mathbb{R}^{C \times F}$ is a trainable weight matrix.



### 4. Layer-wise Propagation in GCN

For layer $l+1$:

$$
\mathbf{H}^{(l+1)} = \sigma \left( \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \, \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)
$$

- $\mathbf{H}^{(0)} = \mathbf{X}$  
- $\mathbf{W}^{(l)}$ is the trainable weight matrix for layer $l$  
- $\sigma$ is a non-linear activation (e.g., ReLU)



#### 5. Two-Layer Semi-Supervised GCN

For node classification:

$$
\mathbf{Z} = f(\mathbf{X}, \mathbf{A}) = \text{softmax} \Big( \tilde{\mathbf{A}} \, \text{ReLU}(\tilde{\mathbf{A}} \mathbf{X} \mathbf{W}^{(0)}) \mathbf{W}^{(1)} \Big)
$$

- $\mathbf{W}^{(0)}$ maps input features to hidden features  
- $\mathbf{W}^{(1)}$ maps hidden features to output classes  
- $\mathbf{Z} \in \mathbb{R}^{N \times F}$ contains output logits for all $F$ classes

