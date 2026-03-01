import logging

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("deepvk/USER-bge-m3")

app = FastAPI()


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(..., max_length=3000)


@app.post("/embeddings")
def get_embeddings(request: EmbeddingRequest):
    embeddings = model.encode_document(request.texts, normalize_embeddings=False)
    return {"embeddings": embeddings.tolist()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
