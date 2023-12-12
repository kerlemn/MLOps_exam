from user_database import UserDatabase


def test():

    # create a user database (default path = 'data\\user_database.csv')
    user_database = UserDatabase('data\\user_database.csv')

    # add some pages
    user_database.add_page('Linear algebra', 1)
    user_database.add_page('Python', 1)
    user_database.add_page('Soccer', 0)
    print(user_database.data)

    # save the database
    user_database.save()


if __name__ == '__main__':
    test()
