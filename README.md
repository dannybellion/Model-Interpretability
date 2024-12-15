# Model Interpretability Tools

A toolkit for interpreting and visualizing the internal representations of transformer-based language models like BERT and GPT-2.

## Overview

This project provides tools and visualizations to help understand how transformer models process and represent text internally. It focuses on three key aspects of transformer models:

1. Attention Mechanisms
2. Hidden State Representations  
3. Token Predictions

## Technical Background

### Transformer Architecture
Transformer models like BERT and GPT-2 use self-attention mechanisms to process text. Each layer contains multiple attention heads that learn to focus on different aspects of the input. The model builds up increasingly abstract representations through multiple layers of:

- Multi-head self-attention
- Feed-forward neural networks
- Layer normalization

### Key Components

**Attention Heads**: Each attention head computes compatibility scores between tokens to create weighted combinations of values. Visualizing these weights shows what information each head focuses on.

**Hidden States**: The vector representations of tokens are transformed layer-by-layer. Tracking how these change reveals how the model builds up understanding.

**Token Embeddings**: The initial embedding layer converts tokens to vectors. The final layer projects to vocabulary probabilities for next token prediction.

## Features

### Attention Visualization
- Heatmap views of attention weights
- Per-layer and per-head analysis
- Interactive selection of attention patterns

### Hidden State Analysis  
- PCA visualization of token representations
- Track evolution across layers
- Compare representations between tokens

### Token Probability Analysis
- View top predicted next tokens
- Compare prediction probabilities
- Analyze model confidence

## Usage

The project includes Jupyter notebooks demonstrating the key visualizations:

```python
# Attention heatmap for a specific layer/head
plot_attention_heatmap(attention_weights[layer][head], tokens)

# PCA of hidden states across layers
plot_hidden_pca(hidden_states, token_idx=1) 

# Top-k token predictions
plot_token_probabilities(logits, tokenizer)
```

See the example notebook for complete usage examples.


## Project Structure

```
.
├── notebooks/          # Example Jupyter notebooks
├── scripts/           # Analysis scripts
├── utils/             # Helper functions
│   ├── model_utils.py   # Model loading and inference
│   ├── viz_utils.py     # Visualization functions
│   └── hooks.py         # Model hooks for extracting states
└── webapp/            # Interactive visualization app
```
