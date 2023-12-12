from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from predictor import get_page, add_feedback
from pydantic import BaseModel
from json import dumps
import re

__base_URL__ = "https://en.wikipedia.org/wiki/"

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


class Pref(BaseModel):
    title: str
    score: str
    user: str

@app.get("/")
def read_root():
    print("Welcome")
    return {"Welcome": "to the api"}


@app.get("/predict")
def pre(usr: str, q:int):
    print("Requesting pages from :"+ usr)
    return {"Pages":get_page(usr, q)}

@app.post('/save_data')
def save_data(preference: Pref):
    print("Saving preference from :"+ preference.user)
    add_feedback(preference.user, preference.title, preference.score)