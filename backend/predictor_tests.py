from predictor import add_feedback, get_page, get_URLs
import loader

"""
#####################
###     TESTS     ###
#####################
"""
if __name__ == '__main__':
    user = 3

    """
    ##########################################
    ### Creation of example user feedbacks ###
    ##########################################
    """
    # import pandas as pd
    # pages = loader.get_random_pages(100)
    # df = pd.DataFrame(np.array([pages["TITLE"].values, [np.random.randint(0,2) for _ in range(100)]]).T, columns=["TITLE", "SCORE"])
    # df.to_csv(f"{__path__}/datasets/user{user}.csv", index=False)



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
    # print(suggested_page["url"])
    # score = input("Rate the page (0: dislike, 1: like): ")
    # add_feedback(user, suggested_page["title"], int(score))


    """
    ######################################################
    ### Test to get all the URL for the specified user ###
    ######################################################
    """
    URLs = get_URLs(user)
    print(URLs)
    print((len(URLs)))