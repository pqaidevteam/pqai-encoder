import os

import dotenv

dotenv.load_dotenv()

import uvicorn
from fastapi import FastAPI, Response
from core.vectorizers import SentBERTVectorizer

app = FastAPI()

PORT = int(os.environ["PORT"])


@app.get("/encode")
async def encode(text: str):
    vector = SentBERTVectorizer().embed(text)
    return {"text": text, "vector": vector.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
