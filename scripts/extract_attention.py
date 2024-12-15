# Example script to extract and print attention
from utils.model_utils import run_model

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    inputs, outputs = run_model(text)
    # outputs.attentions: tuple of (layer_count) of [batch, num_heads, seq_len, seq_len]
    attentions = outputs.attentions
    print("Number of layers with attention:", len(attentions))
    print("Shape of attention[0]:", attentions[0].shape)