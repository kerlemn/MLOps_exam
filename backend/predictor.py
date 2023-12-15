import numpy as np
import neptune

import os
from pathlib import Path
__path__ = Path(__file__).parent

from sklearn.linear_model import LogisticRegression

import helper

import warnings
warnings.filterwarnings("ignore")

"""
####################################
### Global variables declaration ###
####################################
"""
# Coefficient for number of row in the predict dataset
__k__               = 10
# Number of feed for each user to trigger the re-train
__newfeed__         = 10 
# # Neptune project name
__neptune_project__ = os.getenv('NEPTUNE_PROJECT').replace("\r", "")
# # Token for neptune.ai
__neptune_token__   = os.getenv('NEPTUNE_API_TOKEN').replace("\r", "")

"""
#############################
### Functions declaration ###
#############################
"""
def predict(user:str, n:int, best=True) -> np.array:
    """
    Function to predict the possible pages that the user specified could like based on its preferences (given from the model trained on its preferences).

    Parameters
    ----------
    user: str
        The id to determine the model to use
    n: int, optional
        The number of pages to return
    best: bool, optional
        Boolean to select if return the most suggested page or just one of the several suggested pages with respect to their scores
    
    Returns
    -------
    reccomended_page: np.array
        The reccomended pages based on the preferences of the user.
    """
    n_row = __k__ * n

    # Load the dataset
    pages = helper.get_random_pages(n_row)
    X     = pages.drop("TITLE", axis=1).values

    # Load the model
    model = helper.load_model(user)
    if model is not None:
        # Predict the probabilities
        probabilities = model.predict_proba(X)[:, 1]

        if best:
            # Get the index of the most suggested page
            reccomended_idx           = np.argsort(probabilities)[-n:]
        else:
            # Calculate the probability for each element to be chosen
            sum_selected              = np.sum(probabilities)
            reccomended_probabilities = probabilities/sum_selected

            # Select an element with respect to the probabilities
            reccomended_idx           = np.random.choice(range(n_row), n, p=reccomended_probabilities, replace=False)
    else:
        # If the model is not trained yet, return a random page
        reccomended_idx = np.random.randint(0, n_row, n)

    # Get the original index of the page
    reccomended_page = pages.values[reccomended_idx]

    return reccomended_page

def train(user:str):
    """
    Function to train a new model for the given user id based on its preferences and save it in a pickle file.

    Parameters
    ----------
    user: str
        User id to determine the preferences to use
    """
    X, y = helper.get_training_data(user)

    # If the user has only one class, don't train the model
    if np.unique(y).shape[0] == 1:
        print(f"User {user} has only one class, train when more feedback is given")
        return

    # Fit the model
    clf = LogisticRegression(max_iter=3000).fit(X, y)

    # Save the model
    helper.save_model(user, clf)

    # Save the user's feedback related to the page suggested on neptune to monitorate the prediction correctness
    run = neptune.init_run(
        project   = __neptune_project__,
        api_token = __neptune_token__,
    )

    run["user"] = user
    run["parameters"] = clf.get_params()
    run["coefficients"] = helper.get_best_coefficients(clf.coef_[0])
    run["intercept"] = clf.intercept_
    run["rows"] = len(y)

    run.stop()

def get_page(user:str, n=1, best=True) -> list:
    """
    Function to get the Wikipedia page predicting it for the specified user.

    Parameters
    ----------
    user: id
        User id which gave the feedback
    n: int, (optional)
        Number of pages to return

    Returns
    -------
    page: list
        List of dictionary containing the url and the title of the pages
    """

    pages_info = predict(user=user, n=n, best=best)
    pages      = pages_info[:, 0]
    urls       = [url.replace(' ', '_') for url in pages]

    suggested  =  [{"url": f"https://en.wikipedia.org/wiki/{url}", "title": page} 
                   for url, page in zip(urls, pages)]

    return suggested

def add_feedback(user:str, title_page: str, score: int):
    """
    Function to add a feedback about a page to the user's feedback .csv file.

    Parameters
    ----------
    user: id
        User id which gave the feedback
    title_page: string
        Title of the page
    score: bool
        Score given to the page (0: dislike, 1:like)
    """
    # # Save the user's feedback related to the page suggested on neptune to monitorate the prediction correctness
    # run = neptune.init_run(
    #     project  ="WikiTok/WikiTok"#__neptune_project__,
    #     # api_token=__neptune_token__,
    # )

    # run["user"] = user
    # run["predicted"] = title_page
    # run["score"] = score

    # run.stop()

    # Add the feedback to the user's feedback .csv file
    # And obtain the number of feedbacks given by the user
    n = helper.add_feedback(user, title_page, score)

    # Every __newfeed__ feedback re-train the model
    if n % __newfeed__ == 0:
        print(f"{n} feedback: Retrain model for user {user}")
        train(user)

def get_URLs(user:str) -> np.array:
    """
    Function to get the pages rated by the user.

    Parameters
    ----------
    user: id
        User id which gave the feedback

    Returns
    -------
    pages: np.array
        Array of the URL of the user's rated pages
    """
    # Get the titles rated by the user
    pages_df = helper.load_user_feedback(user)

    # Get the titles of the pages
    titles   = pages_df["TITLE"].values

    # Get the URL of the pages
    pages    = [{ "url": f"https://en.wikipedia.org/wiki/{title}", "title": title } for title in titles]
    return pages

def get_URLs_liked(user:str) -> np.array:
    """
    Function to get the pages liked by the user.

    Parameters
    ----------
    user: id
        User id which gave the feedback

    Returns
    -------
    pages: np.array
        Array of the URL of the user's liked pages
    """
    # Get the titles rated by the user
    pages_df = helper.load_user_feedback(user)
    
    # Get the titles of the pages liked by the user
    titles   = pages_df["TITLE"][pages_df["SCORE"] == "True"].values

    # Get the URL of the pages
    pages    = [{ "url": f"https://en.wikipedia.org/wiki/{title}", "title": title } for title in titles]
    return pages