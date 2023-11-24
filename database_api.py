"""
----------------------------------------------------------------
                            WikiTok API
----------------------------------------------------------------

The DatabaseAPI class allows you to create a database of reviews

Note: Remember to run .dump_data() to save the DataFrame to a file

----------------------------------------------------------------
The database wiki_bags.pkl file contains a DataFrame with the
following columns:
- TITLE: the title of the wikipedia page
- SCORE: total number of positive reviews
- <keyword1>: is <keyword1> contained in the page?
- <keyword2>: is <keyword2> contained in the page?
...

----------------------------------------------------------------
"""
import pandas as pd
import os


SPECIAL_COLUMNS_DEFAULTS = {'TITLE': '-', 'SCORE': 0}


class DatabaseAPI:
    """
    pipeline for building a dataset of
    bags of words from wikipedia pages
    """
    def __init__(self, keywords=None, dir_path='data', file_name='pages.csv', verbose=True):
        self.data_dir_path = dir_path
        self.file_name = file_name
        self.keywords = keywords
        self.verbose = verbose
        self.df = None
        self._build_env()
        self.load_data()    # sets keywords

    def report(self, text, start='\n>>> ', end='\n'):
        if self.verbose:
            print(f'{start}{text}', end='\n'+end)

    def get_keywords(self):
        """return the list of keywords"""
        return self.keywords

    def _build_env(self):
        """ build environment """

        # create missing directories
        if not os.path.exists(self.data_dir_path):
            os.mkdir(self.data_dir_path)

        # create missing files
        if not os.path.exists(self.get_database_path()):
            assert self.keywords is not None, 'keywords must be provided when initializing pipeline'
            columns = list(SPECIAL_COLUMNS_DEFAULTS.keys()) + list(self.keywords)
            self.df = pd.DataFrame(columns=columns)
            self.dump_data()

    def get_database_path(self):
        """return the path where the wiki bag of words are saved"""
        return self.data_dir_path + f'/{self.file_name}'

    def load_data(self) -> pd.DataFrame:
        """
        overwrite keywords with the keywords of the df
        return the content of the wiki_bags.pkl file
        """
        self.report(f'Loading data from {self.get_database_path()}')
        self.df = pd.read_csv(self.get_database_path(), encoding='latin1', sep='\t')
        new_keywords = list(self.df.keys())[len(SPECIAL_COLUMNS_DEFAULTS):]
        if self.keywords is not None:
            if self.keywords != new_keywords:
                self.report('WARNING: keywords have been overwritten by the ones in wiki_bags.pkl')
        self.keywords = new_keywords
        return self.df

    def dump_data(self):
        """save the dataframe to file"""
        self.report(f'Saving {len(self.df)} entries...')
        self.df.to_csv(self.get_database_path(), sep='\t', index=False)

    def _add_page(self, title, binary_list: list):
        """
        Add new title to dataframe
        check that the title isn't already present first!

        title: page title
        binary_list: list of boolean values that represents whether
                     each keyword is present or not in the file text
        """
        assert title not in self.df['TITLE'].values, f'title "{title}" not found!'
        assert len(binary_list) == len(self.keywords)
        index = len(self.df)
        default = list(SPECIAL_COLUMNS_DEFAULTS.values())
        self.df.loc[index] = default + binary_list
        self.df.loc[index, 'TITLE'] = title

    def add_review(self, title, score, binary_list=None):
        """
        Add user review to dataframe (set score)
        Also add page first if not already present

        title: page title
        score: user's review (semantic: 0=bad, 1=good)
        binary_list: list of boolean values that represents whether
                     each keyword is present or not in the file text
        """
        assert score in (0, 1)

        # add page if not already present
        if title not in self.df['TITLE'].values:
            assert binary_list is not None, 'please provide binary_list!'
            self._add_page(title, binary_list)

        # get index of the entry
        index = self.df[self.df['TITLE'] == title].index[0]

        # update score
        self.df.loc[index, 'SCORE'] = score

    def get_training_data(self):
        """ Get dataset of user feedback to train the model on """
        x = self.df[self.keywords].to_numpy()
        y = self.df['SCORE'].to_numpy()
        return x, y
