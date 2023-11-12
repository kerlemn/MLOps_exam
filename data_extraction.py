"""
Web scraping pipeline for building a dataset of bags of words from wikipedia pages
- download wikipedia pages
- extract text
- clean text
- build bag of words
- save bag of words

The pipeline is used in model_training.py to build the dataset

The wiki_bags.pkl file contains a dictionary:
{'bags': dict(...), 'keywords': [...]}

"""
import requests
import pickle
import os


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
    def __init__(self):
        self.data_dir_path = 'data'
        self.keywords = None
        self._build_env()
        self.get_bags()  # sets keywords

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
                pickle.dump({'bags': dict(), 'keywords': []}, f)  # initial empty dictionary

    def text_to_bag(self, text):
        assert self.keywords is not None, 'keywords not set'
        return [word in text for word in self.keywords]

    def get_wiki_bags_path(self):
        """return the path where the wiki bag of words are saved"""
        return self.data_dir_path + '/wiki_bags.pkl'

    def get_user_data_path(self, user_id):
        """return the path where the user data is saved"""
        return self.data_dir_path + f'/user_data/user_{user_id}.pkl'

    def get_bags(self) -> dict:
        """return dictionary of all the bags in the database"""
        with open(self.get_wiki_bags_path(), 'rb') as f:
            dct = pickle.load(f)
            new_keywords = dct['keywords']
            if new_keywords:
                self.keywords = dct['keywords']
            return dct['bags']

    def _save_bag(self, title, bag):
        """append bag to the dictionary saved in the file"""
        wiki_bags = self.get_bags()
        wiki_bags[title] = tuple(bag)
        with open(self.get_wiki_bags_path(), 'wb') as f:
            pickle.dump({'keywords': self.keywords, 'bags': wiki_bags}, f)

    def set_keywords(self, keywords: list):
        """set the keywords for the pipeline"""
        if not self.keywords:
            self.keywords = keywords
        elif self.keywords != keywords:
            raise ValueError('Keywords already set! Use: get_bags()')

    def add_bags(self, titles: list):
        """add bag to the dictionary saved in the file"""
        if not self.keywords:
            raise ValueError('Keywords not set! Use: set_keywords(keywords)')
        for title in titles:
            text = request_text_from_wiki(title)
            text = clean_text(text)
            bag = self.text_to_bag(text)
            self._save_bag(title, bag)


def test_make_dataset():
    """download data and convert to BoW"""

    # choose keywords
    keywords = ['python', 'programming', 'matrix', 'batteries', 'budget', 'engineering', 'science']

    # initialize pipeline
    pipeline = WebScrapingPipeline()

    # download Wiki pages
    titles = ['Python (programming language)', 'The Matrix', 'Linear algebra']
    pipeline.set_keywords(keywords)
    pipeline.add_bags(titles)

    print('done')


def test_read_dataset():
    """show the dataset"""

    # initialize pipeline
    pipeline = WebScrapingPipeline()

    # show the BoWs
    print('\nall bags:')
    bags = pipeline.get_bags()
    for title in bags:
        print(f'{bags[title]} {title}')

    # show keywords
    print(f'\nwords = {pipeline.keywords}')


if __name__ == '__main__':
    # test_make_dataset()
    test_read_dataset()
