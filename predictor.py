import pickle as pkl
import numpy as np
import pandas as pd

def load_dataset() -> pd.DataFrame:
    """
    Function to load the "pages.csv" data.

    Returns
    -------
    pd.Dataframe
        Pandas Dataframe containing the information about the wikipedia pages.
    """

    pages_df = pd.read_csv("datasets/pages.csv", sep="\t")
    return pages_df

def load_model(user):
    """
    Function to load a model from pickle file specific for the specified user id.

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

def predict(user="", k=10, best=True) -> np.array:
    """
    Function to predict the possible pages that the user specified could like based on its preferences (given from the model trained on its preferences).

    Parameters
    ----------
    user: int
        The id to determine the model to use
    k: int, optional
        The number of row to keep from the whole dataset
    best: bool, optional
        Boolean to select if return the most suggested page or just one of the several suggested pages with respect to their scores
    
    Returns
    -------
    reccomended_page: np.array
        The reccomended page based on the preferences of the user.
    """

    # Load the model
    model                     = load_model(user)
    n                         = X.shape[0]
    
    # Select k random elements from the probabilities
    selected_idxs             = np.random.choice(range(n), k)
    selected_rows             = X[selected_idxs]

    # Predict the probabilities
    probabilities             = model.predict_proba(selected_rows)[:, 1]

    if best:
        # Get the index of the most suggested page
        reccomended_idx_tmp       = np.argmax(probabilities)
    else:
        # Calculate the probability for each element to be chosen
        sum_selected              = np.sum(probabilities)
        reccomended_probabilities = probabilities/sum_selected

        # Select an element with respect to the probabilities
        reccomended_idx_tmp       = np.random.choice(range(k), 1, p=reccomended_probabilities)[0]

    # Get the original index of the page
    reccomended_idx           = selected_idxs[reccomended_idx_tmp]
    reccomended_page          = pages_df.values[reccomended_idx]

    return reccomended_page



if __name__ == '__main__':
    pages_df = load_dataset()
    X = pages_df.drop(["TITLE", "SCORE", "VISITS"], axis=1).values

    page = predict()
    print(page)
