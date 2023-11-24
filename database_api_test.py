from database_api import *
from misc import get_keywords
from random import randrange


def test_add_review():
    """download data and convert to BoW"""

    # choose keywords (necessary only the first time)
    keywords = get_keywords('data/english10k.txt')

    # initialize pipeline (keywords are necessary only the first time)
    handler = DatabaseAPI(keywords=keywords, verbose=True)

    # add review
    title = f'Test title {randrange(10**3)}'
    binary_list = [False] * len(keywords)       # given by BoW
    handler.add_review(title, score=randrange(2), binary_list=binary_list)

    # save file
    handler.dump_data()

    # show data
    print(handler.df.head())


def test_get_data():

    # choose keywords (necessary only the first time)
    keywords = get_keywords('data/english10k.txt')

    # initialize pipeline (keywords are necessary only the first time)
    handler = DatabaseAPI(keywords=keywords)

    # extract data
    x, y = handler.get_training_data()

    # show data
    print(x)
    print(y)


if __name__ == '__main__':
    test_add_review()
    # test_get_data()
