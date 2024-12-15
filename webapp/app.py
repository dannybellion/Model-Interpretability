import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_utils import load_model_and_tokenizer, get_model_outputs
from utils.viz_utils import plot_attention_heatmap, plot_hidden_state_pca

st.title("Model Interpretability Explorer")

model_name = st.text_input("Enter model name:", "gpt2")
text_input = st.text_area("Enter text to analyze:", "Hello, world!")

if st.button("Analyze"):
    model, tokenizer = load_model_and_tokenizer(model_name)
    inputs = tokenizer(text_input, return_tensors="pt")
    outputs = get_model_outputs(model, inputs["input_ids"])
    
    st.subheader("Attention Visualization")
    layer_idx = st.slider("Layer", 0, model.config.num_hidden_layers-1, 0)
    head_idx = st.slider("Head", 0, model.config.num_attention_heads-1, 0)
    
    attention = outputs.attentions[layer_idx][0, head_idx].detach().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    fig = plot_attention_heatmap(attention, tokens, layer_idx, head_idx)
    st.pyplot(fig) 