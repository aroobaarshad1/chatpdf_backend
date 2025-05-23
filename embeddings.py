import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()

# Get and check API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing in .env file")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

def get_gemini_embedding(text: str, task_type: str = "retrieval_document") -> list:
    """
    Generates an embedding for the given text using Google's Gemini embedding model.

    Args:
        text (str): Input text to embed.
        task_type (str): Task type for the embedding (e.g., 'retrieval_document' or 'retrieval_query').

    Returns:
        list: A list representing the embedding vector.
    """
    if not text:
        raise ValueError("❌ Input text cannot be empty for embedding.")

    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type=task_type
        )
        return result["embedding"]
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return []
