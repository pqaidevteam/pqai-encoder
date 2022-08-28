"""Server file

Attributes:
    app (fastapi.applications.FastAPI): Fast API app
"""
from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Response
from typing import Union, List, Literal
from pydantic import BaseModel

from core.encoders import BagOfEntitiesEncoder
from core.vectorizers import (
    SentBERTVectorizer,
    SIFTextVectorizer,
    EmbeddingMatrix
)

BASE_DIR = str(Path(__file__).parent.resolve())
ASSETS_DIR = f"{BASE_DIR}/assets"

txt_file = f"{ASSETS_DIR}/entities.txt"
npy_file = f"{ASSETS_DIR}/entities.npy"
blklst_file = f"{ASSETS_DIR}/entities_blacklist.txt"
default_boe_encoder = BagOfEntitiesEncoder.from_vocab_file(txt_file, blklst_file)
default_embedding_matrix = EmbeddingMatrix.from_txt_npy(txt_file, npy_file)

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
            vectors = SentBERTVectorizer().encode(req.data)
            return {"original": req.data, "encoded": vectors.tolist()}
    if req.encoder == "sif":
        if isinstance(req.data, list):
            vector = SIFTextVectorizer().encode_many(req.data)
            return {"original": req.data, "encoded": vector.tolist()}
        elif isinstance(req.data, str):
            vectors = SIFTextVectorizer().encode(req.data)
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
