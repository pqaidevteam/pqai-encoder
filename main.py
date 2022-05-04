import os

import dotenv
dotenv.load_dotenv()

import uvicorn
from fastapi import FastAPI, Response

app = FastAPI()

PORT = int(os.environ['PORT'])

@app.get('/encode')
async def encode(text:str):
    return {'success': True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
