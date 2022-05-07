"""Server file

Attributes:
    app (fastapi.applications.FastAPI): Fast API app
"""
import dotenv

dotenv.load_dotenv()

import os
import uvicorn
from fastapi import FastAPI, Response
from core.vectorizers import SentBERTVectorizer

app = FastAPI()


@app.get("/encode")
async def encode(text: str):
    """Encode given text
    
    Args:
        text (str): Text to encode
    
    Returns:
        dict: Dictionary containing encoded text
    """
    vector = SentBERTVectorizer().embed(text)
    return {"text": text, "vector": vector.tolist()}


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)
