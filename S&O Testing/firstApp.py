import os
import json
import re
import csv
from datetime import datetime
import streamlit as st

import openai
import anthropic
import google.generativeai as genai

from envVars.env_vars import openaiKey, geminiKey, claudeKey


# --- API Key Setup (expects you to set these in your environment or secrets) ---
openai.api_key = openaiKey

genai.configure(api_key=geminiKey)
gemini_model_instance = genai.GenerativeModel('gemini-2.0-pro')  # Default

anthropic_client = anthropic.Anthropic(api_key=claudeKey)

OUTPUT_DIR = "outputs"
LOG_FILE = "llm_log.csv"

# --- Model options ---
model_options = {
    "Gemini": ["gemini-2.0-pro", "gemini-2.0-flash"],
    "Claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "OpenAI": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
}

# --- JSON formatter ---
def format_json_to_prompt(data, level=1):
    result = ""
    if isinstance(data, dict):
        for key, value in data.items():
            result += f"\n{'#' * level} {key}\n"
            result += format_json_to_prompt(value, level + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            result += f"\n{'#' * level} Item {i+1}\n"
            result += format_json_to_prompt(item, level + 1)
    else:
        result += f"{data}\n"
    return result

# --- API Calls ---
def call_gemini_api(model: str, temperature: float, prompt: str) -> str:
    try:
        global gemini_model_instance
        if gemini_model_instance.model_name != model:
            gemini_model_instance = genai.GenerativeModel(model)
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return response.text
    except Exception as e:
        return f"Gemini API call failed: {e}"

def call_claude_api(model: str, temperature: float, prompt: str) -> str:
    try:
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0]["text"]
    except Exception as e:
        return f"Claude API call failed: {e}"

def call_openai_api(model: str, temperature: float, prompt: str) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"OpenAI API call failed: {e}"

# --- Filename Sanitizer ---
def clean_filename(text):
    text = re.sub(r'[^a-zA-Z0-9_\- ]', '', text)
    return text.replace(" ", "_")[:40]

# --- Save to CSV Log ---
def log_to_csv(data: dict, path: str = "llm_log.csv"):
    file_exists = os.path.isfile(path)
    with open(path, "a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

def save_output_file(content: str, filename: str, directory: str = OUTPUT_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# --- UI State ---
st.set_page_config(page_title="LLM JSON Prompt Tool", layout="wide")
st.title("🧠 LLM API Caller with JSON Upload + Rating + Logging")

uploaded_file = st.file_uploader("📁 Upload a JSON file", type="json")

api_choice = st.selectbox("🧠 Choose LLM API", list(model_options.keys()))
selected_model = st.selectbox("🛠 Select Model", model_options[api_choice])
temperature = st.slider("🔥 Temperature", 0.0, 1.0, 0.7, step=0.05)

if "response" not in st.session_state:
    st.session_state.response = None
if "submitted" not in st.session_state:
    st.session_state.submitted = False
if "question" not in st.session_state:
    st.session_state.question = ""
if "filename" not in st.session_state:
    st.session_state.filename = ""

default_question = "Please give subjective and objective notes for the dialogue as a doctor."
question = st.text_area("🔍 Question to ask the model", default_question)
st.session_state.question = question

if uploaded_file:
    try:
        data = json.load(uploaded_file)
        formatted_json = format_json_to_prompt(data)

        st.subheader("📄 Final Prompt Sent to Model")
        full_prompt = f"{formatted_json}\n\n## Question\n{question}"
        st.text_area("Prompt Preview", full_prompt, height=300)

        if st.button("🚀 Call API"):
            with st.spinner("Calling API..."):
                if api_choice == "Gemini":
                    response = call_gemini_api(selected_model, temperature, full_prompt)
                elif api_choice == "Claude":
                    response = call_claude_api(selected_model, temperature, full_prompt)
                elif api_choice == "OpenAI":
                    response = call_openai_api(selected_model, temperature, full_prompt)
                else:
                    response = "Unknown API."

            st.session_state.response = response
            st.session_state.submitted = False

    except Exception as e:
        st.error(f"❌ Failed to process JSON: {e}")

# --- Post-response Actions ---
if st.session_state.response:
    st.subheader("📬 API Response")
    st.text_area("Output", st.session_state.response, height=300)

    if not st.session_state.submitted:
        rating = st.slider("⭐ Rate the Response (0–10)", 0, 10, 5, key="rating")
        if st.button("✅ Submit Rating"):
            now = datetime.now()
            filename = f"{api_choice}_{selected_model}_temp{temperature}_r{rating}_{clean_filename(st.session_state.question)}.txt"
            save_output_file(st.session_state.response, filename)
            st.session_state.filename = filename

            log_data = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "prompt": st.session_state.question,
                "api": api_choice,
                "model": selected_model,
                "temperature": temperature,
                "rating": rating
            }
            log_to_csv(log_data)
            st.success("✅ Rating submitted and response saved.")
            st.session_state.submitted = True

    if st.session_state.submitted:
        download_path = os.path.join(OUTPUT_DIR, st.session_state.filename)
        with open(download_path, "r", encoding="utf-8") as f:
            st.download_button(
                label="💾 Download Output File",
                data=f.read(),
                file_name=st.session_state.filename,
                mime="text/plain"
            )

# --- CSV Download ---
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        st.download_button(
            label="📥 Download Log CSV",
            data=f.read(),
            file_name=LOG_FILE,
            mime="text/csv"
        )