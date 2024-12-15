# Example script to extract and analyze hidden states
from utils.model_utils import run_model

if __name__ == "__main__":
    text = "The quick brown fox jumps over the lazy dog."
    inputs, outputs = run_model(text)
    hidden_states = outputs.hidden_states  # tuple of (layer_count+1)
    print("Number of hidden state layers:", len(hidden_states))
    print("Shape of hidden_states[-1]:", hidden_states[-1].shape)