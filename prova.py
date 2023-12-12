import os
from backend.predictor import train

models = os.listdir("backend/models")
users  = [model.split(".")[0][5:] for model in models]

for user in users:
    train(user)
