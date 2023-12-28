from predictor import get_page, add_feedback, get_URLs, get_URLs_liked, train
import helper

"""
#####################
###     TESTS     ###
#####################
"""
if __name__ == '__main__':
    user = "asd"

    # train(user)

    """
    ##########################################
    ### Creation of example user feedbacks ###
    ##########################################
    """
    # import pandas as pd
    # pages = helper.get_random_pages(100)
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
    suggested_pages = get_page(user=user, n=20, best=True)
    for suggested in suggested_pages:
        print(suggested["url"])
        score = input("Rate the page (0: dislike, 1: like): ")
        add_feedback(user, suggested["title"], int(score))

    """
    ######################################################
    ### Test to get all the URL for the specified user ###
    ######################################################
    """
    # URLs = get_URLs(user)
    # print(URLs)
    # print((len(URLs)))

    """
    ####################################################
    ### Test to get liked URL for the specified user ###
    ####################################################
    """
    # URLs = get_URLs_liked(user)
    # print(URLs)
    # print((len(URLs)))

    """
    #########################
    ### Delete all models ###
    #########################
    """

    # response = __supabase__.table('Model') \
    #                     .select("user") \
    #                     .execute() \
    #                     .data
    
    # users = [row["user"] for row in response]
    # print(users)

    # for user in users:
    #     data = __supabase__.table('Model') \
    #                     .delete() \
    #                     .eq('user', user) \
    #                     .execute()
    
    # response = __supabase__.table('Model') \
    #                    .select("user") \
    #                    .execute() \
    #                    .data

    # print(response) 

    """
    #########################
    ### Obtain accuracies ###
    #########################
    """

    # from sklearn.linear_model import LogisticRegression
    # from sklearn.tree         import DecisionTreeClassifier
    # from sklearn.model_selection import cross_val_score
    # from sklearn.metrics import f1_score
    # from sklearn.metrics import make_scorer
    # import numpy as np


    # X, y = helper.get_training_data(user)

    # LR = LogisticRegression(max_iter=3000)
    # LR_accuracy_mean = np.mean(cross_val_score(LR, X, y, cv=10))
    # LR_f1_scores     = np.mean(cross_val_score(LR, X, y,
    #                            cv=10,
    #                            scoring=make_scorer(f1_score, average='weighted')))
    # DT = DecisionTreeClassifier(max_depth=2)
    # DT_accuracy_mean = np.mean(cross_val_score(DT, X, y, cv=10))
    # DT_f1_scores     = np.mean(cross_val_score(DT, X, y,
    #                            cv=10,
    #                            scoring=make_scorer(f1_score, average='weighted')))
    
    # DT = DT.fit(X, y)
    
    # print(f"LR Accuracy: {LR_accuracy_mean}")
    # print(f"LR F1 score: {LR_f1_scores}")
    # print(f"DT Accuracy: {DT_accuracy_mean}")
    # print(f"DT F1 score: {DT_f1_scores}")
    # print(DT.predict_proba(X[:10]))
