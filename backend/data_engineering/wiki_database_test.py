from wiki_database import WikiDatabase


def test_get_random_pages():

    # initialize the database
    wiki_database = WikiDatabase('data/wiki_database')

    # get 10 random titles
    titles = wiki_database.get_random_pages(10)
    print(titles)


def test_get_training_data():

    # initialize the database
    wiki_database = WikiDatabase('data/wiki_database')

    # get your titles (here I choose at random)
    titles = wiki_database.get_random_pages(10)

    # get the training data
    data = wiki_database.get_training_data(titles)
    print(data)


if __name__ == '__main__':
    # test_get_random_pages()
    test_get_training_data()
