from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from predictor import getPage
import re

app = FastAPI()
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent


@app.get("/")
def read_root():
    print("Got it")
    return {"Hello": "World"}


@app.get("/predict")
def pre():
    print("Got it")
    return {"Pages":getPage()}