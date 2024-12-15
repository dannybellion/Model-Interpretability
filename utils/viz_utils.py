import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_attention_heatmap(attention_matrix, tokens, layer_idx, head_idx):
    """Plot attention heatmap for a specific layer and head."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='viridis'
    )
    plt.title(f'Attention Heatmap (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    return plt.gcf()

def plot_hidden_state_pca(hidden_states, tokens):
    """Plot PCA projection of hidden states."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    projected = pca.fit_transform(hidden_states)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(projected[:, 0], projected[:, 1])
    for i, token in enumerate(tokens):
        plt.annotate(token, (projected[i, 0], projected[i, 1]))
    plt.title('PCA Projection of Hidden States')
    return plt.gcf() 