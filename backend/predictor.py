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
    
    # Fit the model
    LR = LogisticRegression(max_iter=3000).fit(X, y)

    # Save the model
    loader.save_model(user, LR)

def get_page(user) -> str:
    """
    Function to get the Wikipedia page predicting it for the specified user.

    Parameters
    ----------
    user: id
        User id which gave the feedback
    """

    page_info = predict(user=user)
    page      = page_info[0].replace(' ', '_')

    return page

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
    # Save the user's feedback related to the page suggested on neptune to monitorate the prediction correctness
    run = neptune.init_run(
        project  ="WikiTok/WikiTok"#__neptune_project__,
        # api_token=__neptune_token__,
    )

    run["user"] = user
    run["predicted"] = title_page
    run["score"] = score

    run.stop()

    # Save the feedback in the user's feedback .csv file
    with open(f"{__path__}/datasets/user{user}.csv", "a") as f:
        string = f"{title_page}\t{score}\n"
        f.writelines(string)

    # Take the number of lines of the feedback .csv file
    with open(f"{__path__}/datasets/user{user}.csv", "r") as f:
        n = len(f.readlines())

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

"""
#####################
###     TESTS     ###
#####################
"""
if __name__ == '__main__':
    user = ""
    """
    ##########################################
    ### Creation of example user feedbacks ###
    ##########################################
    """
    # import pandas as pd
    # pages = get_random_pages(100)
    # df = pd.DataFrame(np.array([pages["TITLE"].values, [np.random.randint(0,2) for _ in range(100)]]).T, columns=["TITLE", "SCORE"])
    # df.to_csv(f"datasets/user{user}.csv", sep="\t", index=False)



    """
    ####################################################################
    ### Test for the retrain of the model based on the new feedbacks ###
    ####################################################################
    """
    # titles = pages_df["TITLE"].values[:__newfeed__]

    # for title in titles:
    #     add_feedback(user, title, np.random.randint(0, 1))
    
    """
    ##############################################################
    ### Test for the predict and the score of a suggested page ###
    ##############################################################
    """
    # suggested_page = get_page(user)
    # print("https://en.wikipedia.org/wiki/" + suggested_page)
    # score = input("Rate the page (0: dislike, 1: like): ")
    # add_feedback(user, suggested_page, score)

    """
    ######################################################
    ### Test to get all the URL for the specified user ###
    ######################################################
    """
    URLs = get_URLs(user)
    print(URLs)