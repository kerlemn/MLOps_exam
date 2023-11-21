import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from loader import load_pages_dataset 
from loader import load_users_dataset 
from loader import load_user_feedback 
from loader import load_model
from loader import save_model

import warnings
warnings.filterwarnings("ignore")

# Number of row in the sample dataset
__k__       = 10
# Number of feed for each user to trigger the re-train
__newfeed__ = 10 

def predict(user="", best=True) -> np.array:
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
    # Load the pages dataset
    pages_df = load_pages_dataset()
    X        = pages_df.drop("TITLE", axis=1).values

    # Load the model
    model                     = load_model(user)
    n                         = X.shape[0]
    
    # Select k random elements from the probabilities
    selected_idxs             = np.random.choice(range(n), __k__)
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
        reccomended_idx_tmp       = np.random.choice(range(__k__), 1, p=reccomended_probabilities)[0]

    # Get the original index of the page
    reccomended_idx           = selected_idxs[reccomended_idx_tmp]
    reccomended_page          = pages_df.values[reccomended_idx]

    return reccomended_page

def train(user=""):
    """
    Function to train a new model for the given user id based on its preferences and save it in a pickle file.

    Parameters
    ----------
    user: int
        User id to determine the preferences to use
    
    """
    # Load the pages dataset
    pages_df   = load_pages_dataset()
    # Load the pages of which the user gave a feedback
    feedback   = load_user_feedback(user)

    # Take the title of the pages in which we have a score
    title_list = feedback["TITLE"].tolist()

    # Take the index of the pages in the dataset
    pages_idxs = [pages_df[pages_df["TITLE"] == title].index[0] for title in title_list]
    
    # Get the info of the pages of which the user gave a feedback and theirs scores
    X          = pages_df.drop("TITLE", axis=1).values[pages_idxs]
    y          = feedback["SCORE"].values
    
    # Fit the model
    LR = LogisticRegression(max_iter=3000).fit(X, y)

    # Save the model
    save_model(user, LR)

def getPage():
    """
    Test shortcut
    """
    page_info = predict()
    page      = page_info[0].replace(' ', '_')
    return "https://en.wikipedia.org/wiki/"+page


def add_feedback(title_page, score, user=""):
    """
    Funtion to add a feedback about a page to the user's feedback .csv file.

    Parameters
    ----------
    user: id
        User id which gave the feedback
    title_page: string
        Title of the page
    score: bool
        Score given to the page (0: dislike, 1:like)
    """

    with open(f"datasets/user{user}.csv", "a") as f:
        string = f"{title_page}\t{score}\n"
        f.writelines(string)

    # Take the number of lines of the file
    with open(f"datasets/user{user}.csv", "r") as f:
        n = len(f.readlines())

    # Every __newfeed__ feedback re-train the model
    if n % __newfeed__ == 0:
        print(f"{n} feedback: Retrain model for user {user}")
        train(user)

if __name__ == '__main__':
    df_pages = load_pages_dataset()
    titles = df_pages["TITLE"].values[:11]

    for title in titles:
        add_feedback(title, np.random.randint(0, 1))
    

    page_info = predict()
    page      = page_info[0].replace(' ', '_')
    print(f"https://en.wikipedia.org/wiki/{page}")