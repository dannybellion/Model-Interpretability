import streamlit as st
import matplotlib.pyplot as plt
from utils.model_utils import run_model, tokenizer
from utils.viz_utils import plot_attention_heatmap, plot_token_probabilities
import io

st.title("Model Interpretability Dashboard")

text = st.text_input("Enter text:", "The quick brown fox jumps over the lazy dog.")

if st.button("Run Model"):
    inputs, outputs = run_model(text)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    layer = st.slider("Layer", 0, len(outputs.attentions)-1, 0)
    head = st.slider("Head", 0, outputs.attentions[layer].shape[1]-1, 0)

    # Attention Visualization
    fig_attn = io.BytesIO()
    plt.figure()
    plot_attention_heatmap(outputs.attentions[layer][0], tokens, layer=layer, head=head)
    st.pyplot(plt.gcf())

    # Token probabilities (if GPT-2)
    fig_probs = io.BytesIO()
    plt.figure()
    plot_token_probabilities(outputs.logits, tokenizer)
    st.pyplot(plt.gcf())