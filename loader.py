import pickle as pkl
import pandas as pd

def load_pages_dataset() -> pd.DataFrame:
    """
    Function to load the "pages.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the wikipedia pages.
    """

    pages_df = pd.read_csv("datasets/pages.csv", sep="\t")
    return pages_df

def load_users_dataset() -> pd.DataFrame:
    """
    Function to load the "users.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the users.
    """

    users_df = pd.read_csv("datasets/users.csv", sep="\t")
    return users_df

def load_user_feedback(user="") -> pd.DataFrame:
    """
    Function to load the "userX_feedback.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the feedback of the specified user.
    """

    feedback_df = pd.read_csv(f"datasets/user{user}.csv", sep="\t")
    return feedback_df

def load_model(user):
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

    with open(f"models/model{user}.pkl", "rb") as f:
        model = pkl.load(f)
    
    return model
    
def save_model(user, model):
    """
    Function to save a model to a pickle file specific for the given user id.

    Parameters
    ----------
    user: int
        The id to determine whose the model is
    """

    with open(f"models/model{user}.pkl", "wb") as f:
        pkl.dump(model, f)