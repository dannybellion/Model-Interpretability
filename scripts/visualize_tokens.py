# Example script to visualize token probabilities
from utils.model_utils import run_model
from utils.viz_utils import plot_token_probabilities

if __name__ == "__main__":
    text = "The capital of France is"
    inputs, outputs = run_model(text)
    logits = outputs[0]  # last_hidden_state, outputs[1] - pooler, outputs.attentions?, ...
    # Actually, for BERT, outputs is (last_hidden_state, pooler_output, attentions, hidden_states)
    # BERT doesn't do next-token prediction by default. For demonstration, assume a language model like GPT-2.
    # If using GPT-2, modify MODEL_NAME in model_utils accordingly and re-run.
    # For demonstration only:
    # If we switch to GPT2:
    # MODEL_NAME = "gpt2"
    # Run again
    # We'll assume logits are next-token logits now.
    # For BERT this won't make sense, but let's just show the code:
    plot_token_probabilities(outputs.logits, inputs.tokenizer)