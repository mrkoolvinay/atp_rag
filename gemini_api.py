import os
import google.generativeai as genai

print(f"DEBUG: Type of genai after import: {type(genai)}") # DIAGNOSTIC PRINT

# Initialize the Gemini client
try:
    GEMINI_API_KEY = os.getenv("GEMINIAI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINIAI_API_KEY not found in environment variables.")
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
    print("Gemini client initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    gemini_model = None

def generate_from_gemini(prompt):
    """
    A generic function to generate content from the Gemini model.
    """
    if not gemini_model:
        return "Error: Gemini client is not initialized."
        
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"