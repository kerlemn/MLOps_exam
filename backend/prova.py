import os
from supabase import create_client, Client
from helper import load_model
import pickle as pkl

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)



# response = supabase.table('Users') \
#                    .insert({"id": "prova"}) \
#                    .execute()
# print(response)

# try:
#     data = supabase.table('Preferences') \
#                    .insert({"Title": "prova3", "User": "prova", "Like": False}) \
#                    .execute()
#     print(data)
# except:
#     print("Duplicated Row, substituting value")
#     data = supabase.table('Preferences') \
#                    .update({"Title": "prova3", "User": "prova", "Like": False}) \
#                    .eq('Title', "prova3") \
#                    .eq('User', "prova") \
#                    .execute()


response = supabase.table('Preferences') \
                   .select('*') \
                   .eq('User', "5") \
                   .execute()
print(response.data)

# clf = load_model("")

# string = pkl.dumps(clf).hex()

# data = supabase.table('Model') \
#                .insert({"user": "34", "hex": string}) \
#                .execute()

response = supabase.table('Model') \
                   .select('*') \
                   .eq('user', "5") \
                   .execute() \
                   .data

string2 = bytes.fromhex(response[0]["hex"])
string = pkl.loads(string2)
print(string.coef_)