import json
import re
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-2.5-flash"

def get_client():
    key = st.secrets.get("GEMINI_API_KEY", "")
    if not key:
        st.error("Gemini API key missing. Add GEMINI_API_KEY in Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=key)

def get_model():
    return st.secrets.get("GEMINI_MODEL", DEFAULT_MODEL)

def gemini_text(prompt: str, system_instruction: str = "", temperature: float = 0.2) -> str:
    client = get_client()
    response = client.models.generate_content(
        model=get_model(),
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            temperature=temperature,
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise ValueError("Gemini returned an empty response. Please retry.")
    return text.strip()

def extract_json(text: str):
    """Parse JSON, tolerating a surrounding Markdown code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise ValueError("The AI response was not valid JSON. Please retry.")

def safe_error(exc: Exception) -> str:
    # Avoid displaying request headers, API keys, or full provider payloads.
    return f"{type(exc).__name__}: {str(exc)[:350]}"
