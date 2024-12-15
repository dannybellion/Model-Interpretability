import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True, output_hidden_states=True, attn_implementation="eager")
model.eval()

# def run_model(text):
#     inputs = tokenizer(text, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     # outputs: (last_hidden_state, pooler_output, attentions, hidden_states)
#     return inputs, outputs


MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, 
                                             output_attentions=True, 
                                             output_hidden_states=True, 
                                             attn_implementation="eager", 
                                             return_dict_in_generate=True
                                             )
model.eval()

def run_model(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return inputs, outputs