import streamlit as st
from streamlit_chat import message as st_message
from transformers import BlenderbotTokenizer
import requests
import json

@st.experimental_singleton
def get_models():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    return tokenizer

if "history" not in st.session_state:
    st.session_state.history = []

st.title("Hello Chatbot")

def generate_answer():
    tokenizer = get_models()
    user_message = st.session_state.input_text

    # Use the Hugging Face Inference API
    url = f"https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": "Bearer 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"}
    data = json.dumps({"inputs": user_message})
    response = requests.request("POST", url, headers=headers, data=data)
    message_bot = response.json()[0]["generated_text"]

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})

st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for i, chat in enumerate(st.session_state.history):
    st_message(**chat, key=str(i))
