import pickle as pkl
import pandas as pd
import numpy  as np

from data_engineering.wiki_database import WikiDatabase
from data_engineering.user_database import UserDatabase

from pathlib import Path

__path__ = Path(__file__).parent

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
    user_database = UserDatabase(f"{__path__}/data_engineering/data/user{user}.csv")

    return user_database.data

def add_feedback(user:str, title:str, score:int):
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
    user_database = UserDatabase(f"{__path__}/data_engineering/data/user{user}.csv")
    user_database.add_page(title, score)
    user_database.save()

    n = len(user_database.data)

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

    try:
        with open(f"{__path__}/models/model{user}.pkl", "rb") as f:
            model = pkl.load(f)
    except:
        model = None
    
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

    with open(f"{__path__}/models/model{user}.pkl", "wb") as f:
        pkl.dump(model, f)

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

def get_columns():
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
    columns_name          = get_columns()

    # Get the best coefficients
    best_coefficients     = coef[best_coefficients_idx]
    best_names            = columns_name[best_coefficients_idx]

    # Get the best n coefficients
    best_coefficients     = best_coefficients[-n:]
    best_names            = best_names[-n:]

    coefficients = {name: coefficient for name, coefficient in zip(best_names, best_coefficients)}

    return coefficients