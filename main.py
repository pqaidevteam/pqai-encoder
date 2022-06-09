"""Server file

Attributes:
    app (fastapi.applications.FastAPI): Fast API app
"""
from dotenv import load_dotenv
load_dotenv()

import os
import uvicorn
from fastapi import FastAPI, Response
from typing import Union, List, Literal
from pydantic import BaseModel
from core.vectorizers import SentBERTVectorizer

class EncodingRequest(BaseModel):
    data: Union[str, List[str]]
    encoder: Literal['sbert', 'boe']

app = FastAPI()


@app.post("/encode")
async def encode(req: EncodingRequest):
    """Encode given item(s)
    """
    if req.encoder == "sbert":
        if isinstance(req.data, list):
            vector = SentBERTVectorizer().encode_many(req.data)
            return {"text": req.data, "vector": vector.tolist()}
        elif isinstance(req.data, str):
            vectors = SentBERTVectorizer().encode_many(req.data)
            return {"text": req.data, "vector": vectors.tolist()}


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
