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
    # user_database = UserDatabase(f"{__path__}/data_engineering/data/user{user}.csv")
    # Get the number of feedbacks given by the user
    response = __supabase__.table('Preferences') \
                       .select('*') \
                       .eq('User', user) \
                       .execute() \
                       .data
    
    titles = [row["Title"] for row in response]
    score  = [row["Like"] for row in response]

    user_database = pd.DataFrame(np.array([titles, score]).T, columns=["TITLE", "SCORE"])

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

    try:
        # Save the preference into the database
        response = __supabase__.table('Preferences') \
                           .insert({"Title": title, "User": user, "Like": score}) \
                           .execute()
    except:
        # If already in the dataset, substitute the previous preference
        response = __supabase__.table('Preferences') \
                           .update({"Title": title, "User": user, "Like": score}) \
                           .eq('Title', title) \
                           .eq('User', user) \
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
    response = __supabase__.table('Model') \
                       .select('*') \
                       .eq('user', user) \
                       .execute() \
                       .data

    # If the user has no model, return None
    if len(response) == 0:
        return None
    
    # Load the model
    string = bytes.fromhex(response[0]["hex"])
    model = pkl.loads(string)
    
    return model
    
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

    string = pkl.dumps(model).hex()

    response = __supabase__.table('Model') \
                       .select('*') \
                       .eq('user', user) \
                       .execute() \
                       .data

    # If the user has no model, return None
    if len(response) == 0:
        __supabase__.table('Model') \
                .insert({"user": user, "hex": string}) \
                .execute()
    # Otherwise, update the model
    else:
        __supabase__.table('Model') \
                .update({"user": user, "hex": string}) \
                .eq('user', user) \
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

    titles = wiki_database.get_random_pages(n)
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
    feedback_title = feedback_df["TITLE"]
    if feedback_title.duplicated().any():
        titles = feedback_title[feedback_title.duplicated()]
        idxs   = [feedback_title[feedback_title == title].index[0] for title in titles]
        feedback_df = feedback_df.drop(idxs)

    # Get the pages information
    rated_titles   = feedback_df["TITLE"].values
    pages          = wiki_database.get_training_data(rated_titles)

    # Get the training data
    X = pages.drop("TITLE", axis=1).values
    y = feedback_df["SCORE"].values

    return X, y


def get_columns_name():
    """
    Function to get the training data for the specified user.

    Returns
    -------
    columns: list
        List of the features of the pages
    """
    wiki_database = WikiDatabase(f'{__path__}/data_engineering/data/wiki_database')

    tmp  = wiki_database.get_random_pages(1)
    page = wiki_database.get_training_data(tmp)

    page.drop("TITLE", axis=1, inplace=True)
    columns = page.columns.values

    return columns

def get_rated_pages(user:str):
    """
    Function to get the title of the pages rated from the user.
    
    Parameters
    ----------
    user: str
        User id which gave the feedback

    Returns
    -------
    titles: list
        Array of the titles of the pages
    """
    feedback_df   = load_user_feedback(user)

    titles = feedback_df["TITLE"].values
    return titles

def get_best_coefficients(coef:list, n:int=10):
    """
    Function to get the best coefficients for the model.

    Parameters
    ----------
    coef: list
        List of the coefficients

    Returns
    -------
    best_coefficients: list
        List of the best coefficients
    """
    # Get the index of the best coefficients
    best_coefficients_idx = np.argsort(np.abs(coef))
    columns_name          = get_columns_name()

    # Get the best coefficients
    best_coefficients     = coef[best_coefficients_idx]
    best_names            = columns_name[best_coefficients_idx]

    # Get the best n coefficients
    best_coefficients     = best_coefficients[-n:]
    best_names            = best_names[-n:]

    coefficients = {name: coefficient for name, coefficient in zip(best_names, best_coefficients)}

    return coefficients