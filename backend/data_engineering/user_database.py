"""
This class is used to interact with the database of the user's preferences about Wikipedia pages.
The databased is stored in a csv file with the following columns:
- TITLE: the title of the wikipedia page
- SCORE: the score of the wikipedia page

Useful methods:
- add_page: add a page to the database
- save: save database to csv file

"""
import pandas as pd
import os


class UserDatabase:

    COLUMNS = ['TITLE', 'SCORE']
    ALLOWED_SCORES = (0, 1)

    def __init__(self, path):
        self.path = path
        self.data = pd.DataFrame(columns=UserDatabase.COLUMNS)
        self._load()

    def _health_check(self):
        """ check if there are duplicate titles """
        if len(self.data) != len(self.data.TITLE.unique()):
            raise ValueError('Invalid database, duplicate titles found!')

    def _load(self):
        """ load database from file """
        if os.path.exists(self.path):
            self.data = pd.read_csv(self.path)
            self._health_check()
        else:
            # create directory if it doesn't exist
            dir_path = os.path.dirname(self.path)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            # create empty database
            self.data = pd.DataFrame(columns=UserDatabase.COLUMNS)

    def add_page(self, page_title, score):
        """
        add a page to the database
        score must be 0 or 1
        """
        if score not in UserDatabase.ALLOWED_SCORES:
            raise ValueError('score must be 0 or 1')

        if page_title in self.data.TITLE.values:
            self.data.loc[self.data.TITLE == page_title, 'SCORE'] = score
        else:
            self.data.loc[len(self.data)] = [page_title, score]

    def save(self):
        """ save database to file """
        self.data.to_csv(self.path, index=False)
