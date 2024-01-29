import pickle as pkl
import pandas as pd
import numpy  as np

from data_engineering.wiki_database import WikiDatabase
from supabase import create_client, Client

from pathlib import Path
import os

__path__ = Path(__file__).parent

# Open the database
__url__: str = os.environ.get("SUPABASE_URL").replace("\r", "")
__key__: str = os.environ.get("SUPABASE_KEY").replace("\r", "")

__supabase__: Client = create_client(__url__, __key__)


from time import time

def load_user_feedback(user:str) -> pd.DataFrame:
    """
    Function to load the "userX_feedback.csv" data.

    Parameters
    ----------
    user: str
        User id which gave the feedback

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the feedback of the specified user.
    """
    # Get the number of feedbacks given by the user
    response = __supabase__.table('Preferences') \
                       .select('*') \
                       .eq('User', user) \
                       .execute() \
                       .data
    
    # Obtain the titles and the scores of the pages
    titles = [row["Title"] for row in response]
    scores = [row["Like"] for row in response]
    times  = [row["TimeStamp"] for row in response]

    # Create the dataframe of the feedbacks
    user_database = pd.DataFrame(np.array([titles, scores, times]).T, columns=["title", "SCORE", "TIMES"])

    return user_database

def add_feedback(user:str, title:str, score:bool):
    """
    Function to add a feedback to the user's feedback.

    Parameters
    ----------
    user: str
        User id which gave the feedback
    title: str
        Title of the page rated
    score: int
        Score given to the page

    Returns
    -------
    n: int
        Number of feedbacks given by the user
    """    
    # Check if the User exists
    response = __supabase__.table('Users') \
                       .select('*') \
                       .eq('id', user) \
                       .execute() \
                       .data
    
    # If not, create it
    if len(response) == 0:
        response = __supabase__.table('Users') \
                           .insert({"id": user}) \
                           .execute()

    # Save the preference into the database
    response = __supabase__.table('Preferences') \
                           .upsert({"Title": title, "User": user, "Like": score}) \
                           .execute()
    
    # Get the number of feedbacks given by the user
    response = __supabase__.table('Preferences') \
                       .select('*') \
                       .eq('User', user) \
                       .execute() \
                       .data
    
    n = len(response)

    return n

def load_model(user:str):
    """
    Function to load a model from a pickle file specific for the given user id.

    Parameters
    ----------
    user: str
        The id to determine the model to use

    Returns
    -------
    model:
        Scikit-Learn model
    """
    # Get the user model
    response = __supabase__.table('Model') \
                       .select('*') \
                       .eq('user', user) \
                       .execute() \
                       .data

    # If the user has no model, return None
    if len(response) == 0:
        return None, None
    
    # Load the model converting from hex to bytes
    string = bytes.fromhex(response[0]["hex"])
    model = pkl.loads(string)
    
    return model, model.coef_[0]
    
def save_model(user:str, model):
    """
    Function to save a model to a pickle file specific for the given user id.

    Parameters
    ----------
    user: str
        The id to determine whose the model is
    model:
        Scikit-Learn model to save
    """

    # Convert the model to bytes and then to hex
    string = pkl.dumps(model).hex()

    # Insert the model into the database
    __supabase__.table('Model') \
                .upsert({"user": user, "hex": string}) \
                .execute()


def get_random_pages(n = 10) -> pd.DataFrame:
    """
    Function to get n random pages from the Wikipedia database.

    Parameters
    ----------
    n: int
        Number of pages to retrieve

    Returns
    -------
    pages: pd.DataFrame
        dataframe of information of the pages retrieved
    """
    wiki_database = WikiDatabase(f'{__path__}/data_engineering/data/wiki_database')

    # Get random titles of pages
    titles = wiki_database.get_random_pages(n)
    # Get random pages from the database
    pages  = wiki_database.get_training_data(titles)

    return pages

def get_training_data(user:str):
    """
    Function to get the training data for the specified user.
    
    Parameters
    ----------
    user: str
        User id which gave the feedback

    Returns
    -------
    X: np.array
        Array of the features of the pages
    y: np.array
        Array of the scores of the pages
    """
    # Load the data

    wiki_database  = WikiDatabase(f'{__path__}/data_engineering/data/wiki_database')
    feedback_df    = load_user_feedback(user)

    # Remove the duplicated titles from the feedback dataframe but removing the oldest ones
    feedback_title = feedback_df["title"]
    if feedback_title.duplicated().any():
        titles = feedback_title[feedback_title.duplicated()]
        idxs   = [feedback_title[feedback_title == title].index[0] for title in titles]
        feedback_df = feedback_df.drop(idxs)


    # Get the pages information
    rated_titles   = feedback_df["title"].values
    pages          = wiki_database.get_training_data(rated_titles)

    # Get the training data
    pages = pages.drop("title", axis=1)
    columns = pages.columns.values
    X = pages.values
    y = feedback_df["SCORE"].values

    return X, y, columns


def get_columns_name():
    """
    Function to get the training data for the specified user.

    Returns
    -------
    columns: list
        List of the features of the pages
    """
    wiki_database = WikiDatabase(f'{__path__}/data_engineering/data/wiki_database')

    # Get a random page to access to the columns of the pages
    tmp  = wiki_database.get_random_pages(1)
    page = wiki_database.get_training_data(tmp)

    # Get the columns names
    page.drop("title", axis=1, inplace=True)
    columns = page.columns.values

    return columns

def get_all_pages():
    """
    Function to get all the pages rated by the users.

    Returns
    -------
    df: pd.DataFrame
        Dataframe of the pages rated by the users
    grouped: pd.DataFrame
        Dataframe of the mean of the ratings grouped by the title
    """
    # Obtain the titles and the scores of the pages
    response = __supabase__.table('Preferences') \
                           .select('*') \
                           .execute() \
                           .data
    
    # Decompose the response
    titles = [row["Title"]             for row in response]
    scores = [1 if row["Like"] else -1 for row in response]

    # Create the dataframe of the feedbacks
    df = pd.DataFrame(np.array([titles, scores]).T, columns=["Title", "Score"])
    df["Score"] = df["Score"].astype(int)

    # Group the dataframe by the title and calculate the mean of the scores
    grouped  = df.groupby("Title")
    grouped  = (grouped.sum() + 1) / (grouped.count() + 2)

    return df, grouped