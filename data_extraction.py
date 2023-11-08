import requests
import pickle
import os
import random


WIKI_URL = 'https://en.wikipedia.org/w/api.php'
REMOVE_CHARS = '[]{}()<>|\\^~`@#$%&*_-+=;:\'",.?/!\t\n'


def clean_text(text: str):
    """ clean text from html tags and other stuff """
    # remove special chars
    for c in REMOVE_CHARS:
        text = text.replace(c, ' ')

    # lower case
    text = text.lower()

    # split
    v = [x for x in text.split(' ') if x != '']
    return v


def get_random_id(max_ids=10**15):
    """ return a random id """
    return random.randint(0, max_ids)


def request_to_wiki(title):
    """ return response from wikipedia api given some url and title """
    params = {'action': 'query',
              'format': 'json',
              'titles': title,
              'prop': 'extracts',
              'exintro': True,
              'explaintext': True}
    return requests.get(WIKI_URL, params=params).json()


def request_text_from_wiki(title):
    """ extract text from wikipedia page given some title """
    pages = request_to_wiki(title)['query']['pages']
    k = list(pages.keys())[0]
    try:
        return pages[k]['extract']
    except KeyError:
        raise ValueError(f'page "{title}" not found')


class WebScrapingPipeline:
    """
    pipeline for building a dataset of bags of words from wikipedia pages
    """
    def __init__(self, bag_creator):
        self.bag_creator = bag_creator
        self.data_dir_path = 'data'
        self._build_env()

    def _build_env(self):
        # build environment
        root = self.data_dir_path
        paths = [root, root + '/user_data']

        # create missing directories
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

        # create wiki_bags.pkl if it doesn't exist
        if not os.path.exists(self.get_wiki_bags_path()):
            with open(self.get_wiki_bags_path(), 'wb') as f:
                pickle.dump(dict(), f)

    def get_wiki_bags_path(self):
        """return the path where the wiki bag of words are saved"""
        return self.data_dir_path + '/wiki_bags.pkl'

    def get_user_data_path(self, user_id):
        """return the path where the user data is saved"""
        return self.data_dir_path + f'/user_data/user_{user_id}.pkl'

    def convert_text_to_bag(self, text: str):
        """return bag of words given some text"""
        text = clean_text(text)
        return self.bag_creator(text)

    def save_bag(self, title, bag: list):
        """append bag to the dictionary saved in the file"""
        with open(self.get_wiki_bags_path(), 'rb') as f:
            wiki_bags = pickle.load(f)
        wiki_bags[title] = bag
        with open(self.get_wiki_bags_path(), 'wb') as f:
            pickle.dump(wiki_bags, f)

    def add_bag(self, title):
        """add bag to the dictionary saved in the file"""
        text = request_text_from_wiki(title)
        bag = self.convert_text_to_bag(text)
        self.save_bag(title, bag)

    def add_user(self):
        """add user to the database and return his id"""
        user_id = get_random_id()
        path = self.get_user_data_path(user_id)
        if not os.path.exists(path):
            with open(path, 'wb') as f:
                pickle.dump({}, f)
        return user_id

    def get_bags(self) -> dict:
        """return dictionary of all the bags in the database"""
        with open(self.get_wiki_bags_path(), 'rb') as f:
            return pickle.load(f)


def test_1():
    """
    test use of pipeline:
    - create new user
    - download some bags
    - show results
    """

    # chose bag of wards
    words = ['python', 'programming', 'matrix', 'batteries', 'budget', 'engineering', 'science']

    def bag_creator(text):
        return [word in text for word in words]

    # initialize pipeline
    pipeline = WebScrapingPipeline(bag_creator)

    # dd user (just for testing)
    new_user_id = pipeline.add_user()
    print('\nOwO! Welcome to the party, user', new_user_id)

    # download this titles
    titles = ['Python (programming language)', 'The Matrix', 'Linear algebra']
    for title in titles:
        pipeline.add_bag(title)

    # show the results
    print(f'\nwords = {words}')
    print('\nall bags:')
    bags = pipeline.get_bags()
    for title in bags:
        print(f'{bags[title]} {title}')


if __name__ == '__main__':
    test_1()
