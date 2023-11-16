"""
----------------------------------------------------------------
                            WikiTok
----------------------------------------------------------------

This file contains the pipeline for building a dataset of bags
of words from wikipedia pages
Web scraping pipeline for building a dataset of bags of words
from wikipedia pages
- download wikipedia pages
- extract text
- clean text
- build bag of words
- save bag of words

The pipeline is used in model_training.py to build the dataset

----------------------------------------------------------------
The database wiki_bags.pkl file contains a DataFrame with the
following columns:
- TITLE: the title of the wikipedia page
- SCORE: the score of the wikipedia page
- VISITS: the number of visits of the wikipedia page
- <keyword1>: is <keyword1> contained in the page?
- <keyword2>: is <keyword2> contained in the page?
...

----------------------------------------------------------------
"""
import wikipediaapi
import pandas as pd
import os


WIKI_URL = 'https://en.wikipedia.org/w/api.php'
KEEP_CHARS = 'abcdefghijklmnopqrstuvwxyz '
SPECIAL_COLUMNS = ('TITLE', 'SCORE', 'VISITS')


def clean_text(text: str):
    """ clean text from html tags and other stuff """
    # lower case
    text = text.lower()

    # keep only some chars
    text = ''.join([x if (x in KEEP_CHARS) else ' ' for x in text])

    # split
    return [x for x in text.split(' ') if x != '']


def process_text(s, n=5):
    """cleans and extracts only the first chapter of the text"""
    s = s.replace('  ', '')
    v = s.split('\n')
    v = [x for x in v if len(x) != 1]
    s = '\n'.join(v)
    v = s.split('\n\n')[:n]
    s = '\n'.join(v)
    return s


def request_text_from_wiki(title):
    """ extract text from wikipedia page given some title """
    print(f'downloading "{title}"...')
    wiki = wikipediaapi.Wikipedia(
        user_agent='UniTS DSAI MLOps Project Team Buktu',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    page = wiki.page(title)

    if page.exists():
        return page.text
    else:
        raise ValueError(f'page {title} does not exist!')


def get_keywords(path) -> list:
    """ return the list of keywords from file """
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        v = text.split('\n')

        # remove empty strings
        v = [x for x in v if x]
        return v


def text_to_bag(text, keywords):
    """convert text to bag of words"""
    return [word in text for word in keywords]


class WebScrapingPipeline:
    """
    pipeline for building a dataset of bags of words from wikipedia pages
    """
    def __init__(self, keywords=None, path='data'):
        self.data_dir_path = path
        self.keywords = keywords
        self.df = None
        self._build_env(keywords)
        self.load_data()  # sets keywords

    def get_keywords(self):
        """return the list of keywords"""
        return self.keywords

    def _build_env(self, keywords=None):
        """ build environment """
        root = self.data_dir_path
        paths = [root, root + '/user_data']

        # create missing directories
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)

        # create pages.csv if it doesn't exist
        if not os.path.exists(self.get_wiki_bags_path()):
            assert keywords is not None, 'keywords must be provided when initializing WebScrapingPipeline'

            # create empty dataframe and save it to file
            self.keywords = keywords
            columns = list(SPECIAL_COLUMNS) + list(keywords)
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.get_wiki_bags_path(), sep='\t', index=False)

    def get_wiki_bags_path(self):
        """return the path where the wiki bag of words are saved"""
        return self.data_dir_path + '/pages.csv'

    def load_data(self) -> pd.DataFrame:
        """
        overwrite self.keywords with the keywords of the df
        return the content of the wiki_bags.pkl file
        """
        print('Loading data from ', self.get_wiki_bags_path())
        self.df = pd.read_csv(self.get_wiki_bags_path(), encoding='UTF-8', sep='\t')
        new_keywords = list(self.df.keys())[len(SPECIAL_COLUMNS):]
        if self.keywords is not None:
            if self.keywords != new_keywords:
                print('WARNING: keywords are overwritten by the ones in wiki_bags.pkl')
        self.keywords = new_keywords
        return self.df

    def dump_data(self):
        """save the dataframe to file"""
        print(f'\nSaving {len(self.df)} entries...\n')
        self.df.to_csv(self.get_wiki_bags_path(), sep='\t', index=False)

    def add_page_to_df(self, title, bag_of_words, score=0, visits=1):
        """add entry to the dataframe"""
        if title not in self.df['TITLE'].values:
            first_part = [title, score, visits]
            assert len(first_part) == len(SPECIAL_COLUMNS)
            self.df.loc[len(self.df)] = first_part + bag_of_words
        else:
            # get index of the entry
            index = self.df[self.df['TITLE'] == title].index[0]
            # update score and visits
            self.df.loc[index, 'SCORE'] += score
            self.df.loc[index, 'VISITS'] += visits

    def get_data_copy(self):
        """return the dataframe"""
        return self.df.copy()


def change_database_keywords(keywords):
    """recreate database with new pipeline"""
    wsp = WebScrapingPipeline()

    # save page names and scores
    page_titles = wsp.df['TITLE']
    scores = wsp.df['SCORE']

    # delete pages.csv
    os.remove(wsp.get_wiki_bags_path())

    # re-create wsp with new keywords
    wsp = WebScrapingPipeline(keywords=keywords)

    # iterate over titles to add pages
    for i in range(len(page_titles)):
        text = request_text_from_wiki(page_titles[i])
        bag_of_words = text_to_bag(clean_text(text), wsp.keywords)
        wsp.add_page_to_df(page_titles[i], bag_of_words, score=scores[i])

    # save
    wsp.dump_data()


# --------------------------------
#             TESTS
# --------------------------------


def test_change_keywords():
    kw = get_keywords('data/english10k.txt')
    change_database_keywords(kw)


def test_make_database():
    """download data and convert to BoW"""

    # choose keywords (necessary only the first time)
    keywords = get_keywords('data/english1k.txt')

    # initialize pipeline
    pipeline = WebScrapingPipeline(keywords=keywords)

    # download Wiki pages
    while True:

        title = input('\n\npage title: ')
        if title == 'quit':
            break

        try:
            # download text
            text = request_text_from_wiki(title)
            bag_of_words = text_to_bag(clean_text(text), pipeline.keywords)

            text = process_text(text)
            print(text)

            while True:
                score = input('\nscore (0 or 1): ')
                if score in ('0', '1'):
                    score = int(score)
                    break

            # add page to dataframe
            pipeline.add_page_to_df(title, bag_of_words, score)

            # save dataframe
            pipeline.dump_data()

        except ValueError:
            # page does not exist
            print(f'page {title} does not exist!')


def test_read_database(max_len=60):
    """show the dataset"""

    # initialize pipeline
    pipeline = WebScrapingPipeline()

    # show keywords
    n = len(pipeline.keywords)
    print(f'\nkeywords = {str(pipeline.keywords)[:max_len]} {"..."*(n >= max_len)} ({n})')

    # show dataframe
    if len(pipeline.df):
        print(f'\n{pipeline.df.describe()}')
        print(f'\n{pipeline.df.head()}')
        print(f'\n{pipeline.df.sample(5)}')
    print(f'\nshape = {pipeline.df.shape}')


if __name__ == '__main__':
    # test_make_database()
    test_read_database()

