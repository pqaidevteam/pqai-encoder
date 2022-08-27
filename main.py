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
from core.vectorizers import SentBERTVectorizer, SIFTextVectorizer
from core.encoders import default_boe_encoder, default_embedding_matrix


class EncodingRequest(BaseModel):
    data: Union[str, List[str]]
    encoder: Literal["sbert", "boe", "emb", "sif"]


app = FastAPI()


@app.post("/encode")
async def encode(req: EncodingRequest):
    """Encode given item(s)"""
    if req.encoder == "sbert":
        if isinstance(req.data, list):
            vector = SentBERTVectorizer().encode_many(req.data)
            return {"original": req.data, "encoded": vector.tolist()}
        elif isinstance(req.data, str):
            vectors = SentBERTVectorizer().embed(req.data)
            return {"original": req.data, "encoded": vectors.tolist()}
    if req.encoder == "sif":
        if isinstance(req.data, list):
            vector = SIFTextVectorizer().encode_many(req.data)
            return {"original": req.data, "encoded": vector.tolist()}
        elif isinstance(req.data, str):
            vectors = SIFTextVectorizer().embed(req.data)
            return {"original": req.data, "encoded": vectors.tolist()}
    if req.encoder == "boe":
        if isinstance(req.data, list):
            entities = [default_boe_encoder.encode(text) for text in req.data]
            return {"original": req.data, "encoded": entities}
        elif isinstance(req.data, str):
            entities = default_boe_encoder.encode(req.data)
            return {"original": req.data, "encoded": entities}
    if req.encoder == "emb":
        if isinstance(req.data, list):
            embeddings = [default_embedding_matrix[e].tolist() for e in req.data]
            return {"original": req.data, "encoded": embeddings}
        elif isinstance(req.data, str):
            embedding = default_embedding_matrix[req.data].tolist()
            return {"original": req.data, "encoded": embedding}


if __name__ == "__main__":
    port = int(os.environ["PORT"])
    uvicorn.run(app, host="0.0.0.0", port=port)
