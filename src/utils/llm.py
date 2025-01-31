import pdb
import google.genai as genai
import os

def get_llm_model(provider: str, **kwargs):
    """
    Get the LLM model.
    Currently only supports 'gemini' provider.
    """
    if provider == "gemini":
        api_key = kwargs.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key is required for Gemini models.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=kwargs.get("model_name", "gemini-2.0-flash-exp")) # default to gemini-2.0-flash-exp
        return model
    else:
        raise ValueError(f"Unsupported provider: {provider}. Only 'gemini' is supported.")