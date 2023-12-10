import numpy as np
import neptune

import os
from pathlib import Path
__path__ = Path(__file__).parent

from sklearn.linear_model import LogisticRegression

import loader

import warnings
warnings.filterwarnings("ignore")

"""
####################################
### Global variables declaration ###
####################################
"""
# Number of row in the sample dataset
__k__               = 10
# Number of feed for each user to trigger the re-train
__newfeed__         = 10 
# # Neptune project name
# __neptune_project__ = str(os.getenv('NEPTUNE_PROJECT'))
# # Token for neptune.ai
# __neptune_token__   = str(os.getenv('NEPTUNE_TOKEN'))

"""
#############################
### Functions declaration ###
#############################
"""
def predict(user, best=True) -> np.array:
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
    # Load the dataset
    pages = loader.get_random_pages(__k__)
    X     = pages.drop("TITLE", axis=1).values

    # Load the model
    model = loader.load_model(user)
    if model is not None:
        # Predict the probabilities
        probabilities = model.predict_proba(X)[:, 1]

        if best:
            # Get the index of the most suggested page
            reccomended_idx           = np.argmax(probabilities)
        else:
            # Calculate the probability for each element to be chosen
            sum_selected              = np.sum(probabilities)
            reccomended_probabilities = probabilities/sum_selected

            # Select an element with respect to the probabilities
            reccomended_idx           = np.random.choice(range(__k__), 1, p=reccomended_probabilities)[0]
    else:
        # If the model is not trained yet, return a random page
        reccomended_idx = np.random.randint(0, __k__)

    # Get the original index of the page
    reccomended_page = pages.values[reccomended_idx]

    return reccomended_page

def train(user):
    """
    Function to train a new model for the given user id based on its preferences and save it in a pickle file.

    Parameters
    ----------
    user: int
        User id to determine the preferences to use
    
    """
    X, y = loader.get_training_data(user)

    # If the user has only one class, don't train the model
    if np.unique(y).shape[0] == 1:
        print(f"User {user} has only one class, train when more feedback is given")
        return

    # Fit the model
    clf = LogisticRegression(max_iter=3000).fit(X, y)

    # Save the model
    loader.save_model(user, clf)

    # Save the user's feedback related to the page suggested on neptune to monitorate the prediction correctness
    run = neptune.init_run(
        project  ="WikiTok/WikiTok"#__neptune_project__,
        # api_token=__neptune_token__,
    )

    run["user"] = user
    run["parameters"] = clf.get_params()
    run["coefficients"] = clf.coef_
    run["intercept"] = clf.intercept_
    run["rows"] = len(y)

    run.stop()

def get_page(user) -> str:
    """
    Function to get the Wikipedia page predicting it for the specified user.

    Parameters
    ----------
    user: id
        User id which gave the feedback
    """

    page_info = predict(user=user)
    page      = page_info[0]
    url       = page.replace(' ', '_')

    return {"url": f"https://en.wikipedia.org/wiki/{url}", "title": page}

def add_feedback(user:str, title_page: str, score: str):
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
    n = loader.add_feedback(user, title_page, score)

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
    titles = loader.get_rated_pages(user)

    pages = [{ "url": f"https://en.wikipedia.org/wiki/{title}", "title": title } for title in titles]
    return pages