from typing import Dict, List
import torch

class ModelHooks:
    def __init__(self):
        self.attention_maps = {}
        self.hidden_states = {}
        
    def attention_hook(self, layer_idx):
        def hook(module, input, output):
            self.attention_maps[f'layer_{layer_idx}'] = output[0].detach()
        return hook
    
    def hidden_state_hook(self, layer_idx):
        def hook(module, input, output):
            self.hidden_states[f'layer_{layer_idx}'] = output[0].detach()
        return hook
    
    def register_hooks(self, model):
        """Register hooks for attention and hidden states."""
        for idx, layer in enumerate(model.transformer.h):
            layer.attn.register_forward_hook(self.attention_hook(idx))
            layer.register_forward_hook(self.hidden_state_hook(idx)) 