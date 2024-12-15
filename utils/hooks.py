# Utilities for registering hooks
# Hooks allow extraction of intermediate layer outputs
import torch

def get_attention_hook(attentions_list):
    def hook(module, input, output):
        # output is (hidden_states, attn_weights)
        # For a transformer block, often attention is in output.attentions or as a tuple
        # Adjust as needed for your model's structure.
        # If using a plain BERT model, attention is obtained from model outputs directly, 
        # so this hook might not be necessary.
        attentions_list.append(output)
    return hook

def get_hidden_hook(hiddens_list):
    def hook(module, input, output):
        # output: hidden states at this layer
        hiddens_list.append(output)
    return hook