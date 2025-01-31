# File: test_gemini_chat.py
import os
import types
import pytest
import base64


from google import genai
from google.genai import types



def test_gemini_chat_with_image():
    """
    Example test verifying we can send a text + image prompt to Google Gemini
    using google.genai instead of google.generativeai.
    """
    api_key = "AIzaSyDjmcbRdSo_4xrnXt9JTzZVDoC4LX_-T6o"
    # 1. Create a ChatModel with the vision-capable model name.
    client = genai.Client(api_key=api_key)

    # 2. Load and encode a local image. Adjust the path as needed.
    img_path = "assets/examples/test.png"
    if not os.path.exists(img_path):
        pytest.fail(f"Image file not found: {img_path}")

    with open(img_path, "rb") as f:
        image_bytes = f.read()


    # 3. Generate the response.
    #    The library’s .generate() method will return a ChatResponse object with fields like 'generated_text'.
    try:
        response = client.models.generate_content(model='gemini-2.0-flash-exp',
            contents= ["Describe the following image in detail:" ,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")],
                config= types.GenerateContentConfig(
                temperature=1.5,
                top_p=0.95,
                top_k=40,
                candidate_count=1,
                seed=None,
                max_output_tokens=8192,
                stop_sequences=None,
                presence_penalty=0.0,
                frequency_penalty=0.0))
    except Exception as e:
        pytest.fail(f"Failed to call Gemini chat with image. Error: {str(e)}")

    # 4. Extract the model's answer and run assertions.
    answer = response.text
    assert isinstance(answer, str), "Gemini response content is not a string."
    assert len(answer.strip()) > 0, "Gemini response is empty or only whitespace."

    print("\nGemini response:\n", answer)