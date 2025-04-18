import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai
from envVars.env_vars import openaiKey, geminiKey, claudeKey

# --- API Key Setup ---
openai.api_key = openaiKey
openai_client = OpenAI(api_key=openaiKey)
genai.configure(api_key=geminiKey)
gemini_model_instance = genai.GenerativeModel('gemini-2.0-pro')  # Default
anthropic_client = anthropic.Anthropic(api_key=claudeKey)

# --- Model options ---
model_options = {
    "Gemini": ["gemini-2.0-pro", "gemini-2.0-flash"],
    "Claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "OpenAI": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
}

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
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API call failed: {e}" 