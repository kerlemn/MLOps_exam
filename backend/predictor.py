import numpy as np
import neptune

import os
from pathlib import Path
__path__ = Path(__file__).parent

from sklearn.linear_model import LogisticRegression

import helper

import warnings
warnings.filterwarnings("ignore")

from time import time

"""
####################################
### Global variables declaration ###
####################################
"""
# Coefficient for number of row in the predict dataset
__k__               = 100
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
def predict_no_model(n:int):
    """
    Select the best pages based on the average score of the users.

    Parameters
    ----------
    n: int
        Number of pages to return
    best: bool
        Boolean to select if return the most suggested page or just one of the several suggested pages with respect to their scores
    
    Returns
    -------
    selected: np.array
        Array of the selected page's titles
    """
    # Obtain the average score of the users for each page
    df, grouped = helper.get_all_pages()

    # Select the pages with respect to the average score
    grouped     = grouped[grouped["Score"] > 0]

    # If there are pages with a score > 0
    if grouped.shape[0] > 0:
        # Calculate the probability for each element to be chosen
        proba    = grouped["Score"].values
        proba    = proba / sum(proba) 

        # Select the pages with respect to the probabilities
        titles   = grouped.index.values
        selected = np.random.choice(titles, n, p=proba)
    elif df.shape[0] > 0:
        # If there are no pages with a score > 0, select randomly
        titles   = df["Title"].values
        selected = np.random.choice(titles, n)
    else:
        # If there are no pages in the dataset, select randomly
        n_row    = __k__ * n
        titles   = helper.get_random_pages(n_row)["title"].values
        selected = np.random.choice(titles, n)

    return selected

def predict_model(model, n:int, best:bool):
    """
    Select the best pages based on the model trained on the user's preferences.

    Parameters
    ----------
    model: sklearn.linear_model.LogisticRegression
        Model trained on the user's preferences
    n: int
        Number of pages to return
    best: bool
        Boolean to select if return the most suggested page or just one of the several suggested pages with respect to their scores
    
    Returns
    -------
    reccomended_page: np.array
        Array of the selected page's titles
    """
    n_row = __k__ * n

    # Load the dataset
    pages = helper.get_random_pages(n_row)
    X     = pages.drop("title", axis=1).values

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
    
    # Get the original index of the page
    reccomended_page = pages.values[reccomended_idx, 0]

    return reccomended_page

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
    

    # Load the model
    model, _ = helper.load_model(user)

    if model is not None:
        # Predict the probabilities
        reccomended = predict_model(model, n, best)
    else:
        # If the model is not trained yet return the best rated
        reccomended = predict_no_model(n)

    return reccomended

def train(user:str):
    """
    Function to train a new model for the given user id based on its preferences and save it in a pickle file.

    Parameters
    ----------
    user: str
        User id to determine the preferences to use
    """
    _, old_coef = helper.load_model(user)
    X, y, columns = helper.get_training_data(user)

    # If the user has only one class, don't train the model
    if np.unique(y).shape[0] == 1:
        print(f"User {user} has only one class, train when more feedback is given")
        return

    # Fit the model
    clf = LogisticRegression(max_iter=3000).fit(X, y)

    if old_coef is not None:
        # Update the coefficient with the new one
        prev_importance = 0.5
        clf.coef_[0]    = prev_importance * np.array(clf.coef_[0]) + (1 - prev_importance) * np.array(old_coef)
        clf.coef_[0]    = clf.coef_[0].tolist()

    # Save the model
    helper.save_model(user, clf)

    # Save the user's feedback related to the page suggested on neptune to monitorate the prediction correctness
    run = neptune.init_run(
        project   = __neptune_project__,
        api_token = __neptune_token__,
    )

    parameters = {name: value for name, value in zip(columns, clf.coef_[0])}

    # Weight the feedback by the timestamp
    ordered_feed = get_ordered_feedback(user)
    weighted_feed = [score * np.exp(-i/10) for i, score in enumerate(ordered_feed)]

    run["user"]         = user
    run["parameters"]   = clf.get_params()
    run["coefficients"] = parameters
    run["intercept"]    = clf.intercept_
    run["rows"]         = len(y)
    run["likes"]        = np.mean(weighted_feed)

    run.stop()

    return parameters

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

    # Get the prediction for the user specified
    pages      = predict(user=user, n=n, best=best)

    # Get the URL of the pages
    urls       = [url.replace(' ', '_') for url in pages]

    # Create the list of the suggested pages conform to the frontend
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
    # Add the feedback to the user's feedback .csv file
    # And obtain the number of feedbacks given by the user
    n = helper.add_feedback(user, title_page, score)
    parameters = None

    # Every __newfeed__ feedback re-train the model
    if n % __newfeed__ == 0:
        print(f"{n} feedback: Retrain model for user {user}")
        parameters = train(user)

    return parameters

def get_ordered_feedback(user:str) -> np.array:
    """
    Function to get the feedback of the user ordered by the timestamp.

    Parameters
    ----------
    user: id
        User id which gave the feedback

    Returns
    -------
    feedback: np.array
        Array of the feedback of the user ordered by the timestemp
    """
    # Get the feedback of the user
    feedback_df = helper.load_user_feedback(user)

    # Get the sorted feedback
    feedback = feedback_df.sort_values(by="TIMES", ascending=False)["SCORE"]

    return feedback.values.astype(bool)

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
    titles   = pages_df["title"].values

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
    titles   = pages_df["title"][pages_df["SCORE"] == "True"].values

    # Get the URL of the pages
    pages    = [{ "url": f"https://en.wikipedia.org/wiki/{title}", "title": title } for title in titles]
    return pages
