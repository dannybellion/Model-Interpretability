import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.decomposition import PCA

def plot_attention_heatmap(attn_matrix, tokens, layer=0, head=0):
    # attn_matrix: shape [num_heads, seq_len, seq_len]
    attn = attn_matrix[head].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8,8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap="Blues", square=True, ax=ax)
    ax.set_title(f"Layer {layer}, Head {head} Attention")
    plt.tight_layout()
    plt.show()

def plot_hidden_pca(hidden_states, token_idx=1):
    # hidden_states: List of [batch, seq_len, hidden_dim]
    # We'll extract the representation of a single token across layers
    reps = [h[0, token_idx].detach().cpu().numpy() for h in hidden_states]
    reps = np.array(reps)  # shape: [num_layers, hidden_dim]
    pca = PCA(n_components=2)
    coords = pca.fit_transform(reps)
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1])
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, f"L{i}", fontsize=9)
    plt.title("PCA of Hidden States Across Layers")
    plt.show()

def plot_token_probabilities(logits, tokenizer, top_k=5):
    # logits: [batch, seq_len, vocab_size], take last token
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_idxs = torch.topk(probs, top_k)
    top_tokens = tokenizer.convert_ids_to_tokens(top_idxs)
    top_probs = top_probs.detach().cpu().numpy()

    plt.figure(figsize=(6,4))
    plt.barh(top_tokens, top_probs)
    plt.xlabel("Probability")
    plt.title("Top-k Predicted Tokens")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()