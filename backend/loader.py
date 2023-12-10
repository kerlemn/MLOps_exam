import pickle as pkl
import pandas as pd

from data_engineering.wiki_database import WikiDatabase

from pathlib import Path

__path__ = Path(__file__).parent

def load_users_dataset() -> pd.DataFrame:
    """
    Function to load the "users.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the users.
    """

    users_df = pd.read_csv(f"{__path__}/datasets/users.csv", sep="\t")
    return users_df

def load_user_feedback(user="") -> pd.DataFrame:
    """
    Function to load the "userX_feedback.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the feedback of the specified user.
    """

    feedback_df = pd.read_csv(f"{__path__}/datasets/user{user}.csv", sep="\t")
    return feedback_df

def load_model(user:str):
    """
    Function to load a model from a pickle file specific for the given user id.

    Parameters
    ----------
    user: int
        The id to determine the model to use

    Returns
    -------
    model:
        Scikit-Learn model
    """

    with open(f"{__path__}/models/model{user}.pkl", "rb") as f:
        model = pkl.load(f)
    
    return model
    
def save_model(user:str, model):
    """
    Function to save a model to a pickle file specific for the given user id.

    Parameters
    ----------
    user: int
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
    pages: dataframe
        dataframe of information of the pages retrieved
    """
    wiki_database = WikiDatabase(f'{__path__}/data/wiki_database')

    titles = wiki_database.get_random_pages(n)
    pages  = wiki_database.get_training_data(titles)

    return pages

def get_training_data(user=""):
    """
    Function to get the training data for the specified user.
    
    Parameters
    ----------
    user: int
        User id which gave the feedback

    Returns
    -------
    X: np.array
        Array of the features of the pages
    y: np.array
        Array of the scores of the pages
    """
    wiki_database = WikiDatabase(f'{__path__}/data/wiki_database')
    feedback_df   = load_user_feedback(user)

    pages = wiki_database.get_training_data(feedback_df["TITLE"].values)
    
    X = pages.drop("TITLE", axis=1).values
    y = feedback_df["SCORE"].values

    return X, y

def get_rated_pages(user=""):
    """
    Function to get the title of the pages rated from the user.
    
    Parameters
    ----------
    user: int
        User id which gave the feedback

    Returns
    -------
    titles: list
        Array of the titles of the pages
    """
    feedback_df   = load_user_feedback(user)

    titles = feedback_df["TITLE"].values
    return titles