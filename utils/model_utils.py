import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str):
    """Load a model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def get_model_outputs(model, input_ids, attention_mask=None):
    """Run a forward pass and return the model outputs."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True
        )
    return outputs

def get_token_probabilities(logits, tokenizer):
    """Convert logits to token probabilities."""
    probs = torch.softmax(logits, dim=-1)
    return probs 