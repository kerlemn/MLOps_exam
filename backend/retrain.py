import os
from supabase import create_client, Client
from predictor import train

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

response = supabase.table('Model') \
                   .select("user") \
                   .execute() \
                   .data

users = [user["user"] for user in response]

print(f"Something changed into the database, retrain the model for the {len(users)} users")
for user in users:
    train(user)